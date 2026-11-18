# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 KDA with fused input projections and a full-rank output gate."""

import torch
from b12x.sequence import gdn_decode, kda_prefill, short_conv

from ignition.components.gated_norm import PreparedGatedRMSNorm
from ignition.components.recurrent import prepare_sequence_binding
from ignition.components.kda_speculation import _record, register_recording

from .projections import prepare_projection
from .kda_input import prepare_input_copy
from .kda_overlap import prepare_forget_overlap


class PreparedKDA:
    def __init__(self, weights, source, state, metadata, *, mode, eps, session, name, workspace, buffers, records=None,
                 allocator=None):
        projection_mode = mode
        mode = "decode" if mode == "verify" else mode
        self.records, self.state, self.metadata = records, state, metadata
        tokens, hidden = source.shape
        heads, width = state.recurrent.shape[1], state.recurrent.shape[1] * 128
        device = source.device

        def buffer(key, channels):
            if key not in buffers or buffers[key].shape[0] < tokens:
                buffers[key] = (torch.zeros((tokens, channels), dtype=torch.bfloat16, device=device) if allocator is None
                                else allocator.zeros((tokens, channels), torch.bfloat16))
            return buffers[key][:tokens]

        self.forget = buffer("forget_input", 128)
        raw_g = buffer("raw_g", width)
        mixed = buffer("mixed", 3 * width)
        recurrence, gated = buffer("recurrence", width), buffer("gated", width)
        columns = (4 * width + 128 + heads + 7) // 8 * 8
        self.input = buffer("input", columns)
        qkv, self.forget_source, gate, beta, _ = self.input.split(
            (3 * width, 128, width, heads, columns - 4 * width - 128 - heads), dim=-1)
        self.projections = {key: prepare_projection(x, weights[key], y, session=session,
            name=name + "." + key, workspace=workspace) for key, x, y in (
                ("input", source, self.input), ("forget", self.forget, raw_g),
                ("output", gated, buffer("output", hidden)))}
        self.input_copy = prepare_input_copy(self.projections['input'], source, self.forget,
            mode=projection_mode, session=session, workspace=workspace)
        sequences = metadata.query_start_loc.numel() - 1
        conv_args = dict(x=qkv, conv_weight=weights["convolution"], query_start_loc=metadata.query_start_loc,
            state_slot_ids=metadata.convolution_slots, state_is_fresh=metadata.fresh,
            num_accepted_tokens=metadata.accepted_tokens, num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens,
            conv_state=state.convolution, out=mixed)
        caps = short_conv.Caps(device=device, mode=mode, channels=3 * width, kernel_size=4,
            max_tokens=tokens, max_seqs=sequences, max_state_slots=state.convolution.shape[0],
            max_speculative_tokens=state.convolution.shape[-1] - 3)
        plan = short_conv.plan(caps, invocation=short_conv.invocation_from_tensors(**conv_args))
        prepare_sequence_binding(plan, conv_args, session=session, name=name + ".conv", mode=mode,
            public_bind=short_conv.bind, bind_plan=True, workspace=workspace,
            set_binding=lambda binding: setattr(self, "convolution", binding))
        self.forget_overlap = prepare_forget_overlap(source, self.projections['forget'], plan,
            mode=projection_mode, heads=heads, session=session, workspace=workspace)
        recurrent = state.recurrent if records is None else state.recurrent[:records[0].shape[0]]
        common = dict(raw_g=raw_g.view(tokens, heads, 128), raw_beta=beta, A_log=weights["A_log"],
            dt_bias=weights["dt_bias"].view(heads, 128), recurrent_state=recurrent,
            num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens,
            output=(gated if mode == "decode" else recurrence).view(tokens, heads, 128))
        self.options, self.mode = {"lower_bound": -5.0}, mode
        if mode == "decode":
            self.options["eps"] = eps
            caps = gdn_decode.Caps(device=device, max_tokens=tokens, max_seqs=sequences,
                max_state_slots=recurrent.shape[0], key_heads=heads, value_heads=heads,
                state_index_columns=metadata.state_indices.shape[1], gate_activation="sigmoid", null_state_index=0,
                recover_speculative_state=records is not None)
            args = dict(common, mixed_qkv=mixed, z=gate.view(tokens, heads, 128), norm_weight=weights["norm"],
                query_start_loc=metadata.query_start_loc, num_accepted_tokens=metadata.accepted_tokens,
                state_indices=metadata.state_indices)
            if records is not None:
                args.update(correction_cache=records[0], kg_cache=records[1])
                self.handle = register_recording(self)
                self.reads = [mixed, raw_g, beta, gate, recurrent, metadata.query_start_loc,
                              metadata.state_indices, metadata.num_seqs, metadata.num_tokens]
            plan = gdn_decode.plan(caps, invocation=gdn_decode.invocation_from_tensors(caps, **args))
            bind, method = gdn_decode.bind_kda, "bind_kda"
        else:
            caps = kda_prefill.Caps(device=device, max_tokens=tokens, max_seqs=sequences,
                max_state_slots=state.recurrent.shape[0], heads=heads, null_state_index=0)
            q, k, v = (x.view(tokens, heads, 128) for x in mixed.split(width, dim=-1))
            args = dict(common, q=q, k=k, v=v, cu_seqlens=metadata.query_start_loc,
                initial_state_indices=metadata.initial_state_indices, final_state_indices=metadata.final_state_indices,
                checkpoint_state_indices=metadata.checkpoint_state_indices, checkpoint_offsets=metadata.checkpoint_offsets)
            plan = kda_prefill.plan(caps, invocation=kda_prefill.invocation_from_tensors(**args))
            bind, method = kda_prefill.bind, "bind"
            self.norm = PreparedGatedRMSNorm(recurrence.view(tokens, heads, 128), gate.view(tokens, heads, 128),
                weights["norm"], gated.view(tokens, heads, 128), metadata.num_tokens, eps=eps)
        self.plan = plan
        prepare_sequence_binding(plan, args, session=session, name=name + ".recurrence", mode=mode,
            public_bind=bind, bind_method=method, workspace=workspace, run_options=self.options,
            set_binding=lambda binding: setattr(self, "recurrence", binding))
        self.gated = gated

    def __call__(self, source):
        p = self.projections
        if self.input_copy is not None and self.input_copy.enabled:
            self.input_copy(source)
        else:
            p["input"](source)
            self.forget.copy_(self.forget_source)
        if self.forget_overlap is None:
            p["forget"](self.forget)
            short_conv.run(self.convolution)
        else:
            self.forget_overlap.start(self.forget)
            short_conv.run(self.convolution)
            self.forget_overlap.finish(self.forget, self.convolution.out)
        if self.records is not None:
            _record(self.reads, self.gated, *self.records, self.handle)
        elif self.mode == "decode":
            gdn_decode.run_kda(self.recurrence, **self.options)
        else:
            kda_prefill.run(self.recurrence, **self.options)
            self.norm()
        return p["output"](self.gated)
