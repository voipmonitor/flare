# SPDX-License-Identifier: Apache-2.0
"""Partial tensor-parallel tuning compares complete local components before distributing a race."""

from types import SimpleNamespace

import pytest
import torch
from b12x._lib.compile_pool import CompileJob
from b12x._lib.scratch import ScratchBufferSpec
from b12x.preparation import DetectedDevice, DeviceIdentity, FinalistPolicy, Knob, MemoryRequirements, Plan, TuningContract

from ignition.preparation.collectives import CollectivePreparation
from ignition.preparation.tuning_shards import component_key, local_components, materialization_key


DEVICE = DetectedDevice(0, DeviceIdentity('nvidia', (12, 0), 188, 'RTX PRO 6000 Blackwell'))


def request(name, size, *, dependencies=(), memory=0, factory_value=0, shared=False, policy=None, override=None):
    contract = TuningContract(component_id='test.projection', query_schema_version=1, config_schema_version=1,
        query_fields=frozenset({'size'}), config_fields=frozenset({'width'}),
        encode_query=lambda q: {'size': q}, encode_config=lambda c: {'width': c}, decode_config=lambda d: d['width'],
        validate_query=lambda q, d: None, validate_config=lambda q, c, d: None, default_config=lambda q, d: 1,
        knobs=(Knob(name='width', values=(1, 2)),))
    plan = Plan(contract=contract, query=size, shared=shared, override=override,
        _compile_jobs=lambda c, d: (CompileJob.create('example:compile', size, c, factory_value if c == 2 else 0),),
        _memory_requirements=lambda c, d: MemoryRequirements(scratch=(
            ScratchBufferSpec(name='tiles', shape=(size + (memory if c == 2 else 0),), dtype=torch.uint8, device=torch.device('cuda', d.ordinal)),)),
        _materialize=lambda s, d: s.config)
    return plan.request(name=name, dependencies=dependencies, prepare_call=lambda s: None,
        benchmark_call=lambda s: None, finalist_policy=policy)


def test_local_components_keep_aliases_dependencies_and_collective_neighbors():
    from dataclasses import replace
    a, b, alias = request('input', 32), request('consume', 64, dependencies=('input',)), request('input-alias', 32)
    collective = replace(request('exchange', 128), collective=SimpleNamespace(ranks=(0,1)))
    downstream = request('after-exchange', 256, dependencies=('exchange',))
    groups = local_components((b, downstream, collective, alias, a))
    assert [[r.name for r in group] for group in groups] == [['consume', 'input', 'input-alias']]
    assert component_key(groups[0]) == component_key(local_components((a, alias, b))[0])
    unequal_input = request('input', 48)
    assert component_key(groups[0]) != component_key(local_components((unequal_input, b))[0])
    with pytest.raises(ValueError, match='missing'):
        local_components((request('broken', 32, dependencies=('missing',)),))


def test_component_keys_include_pins_and_finalists_and_every_candidate_binding():
    def key(**kwargs):
        return component_key([request('projection', 32, **kwargs)])
    assert key(override=2) != key()
    assert key(shared=True) != key()
    assert key(policy=FinalistPolicy(family_fields=('width',))) != key()
    session=SimpleNamespace(device=DEVICE)
    fingerprint=lambda **kwargs: materialization_key(session, [request('projection',32,**kwargs)])
    assert fingerprint(memory=4) != fingerprint()
    assert fingerprint(factory_value=8) != fingerprint()
    shared=''.join(['repeated', ' factory argument'])
    assert fingerprint(factory_value=(shared,shared)) == fingerprint(factory_value=(shared,(' '+shared)[1:]))


def test_metadata_failure_is_reported_from_the_failing_rank(monkeypatch):
    coordinator=object.__new__(CollectivePreparation)
    coordinator.group,coordinator.ranks=object(),(2,5)
    def gather(output,local,*,group):
        assert group is coordinator.group
        output[:] = [('ok',None),local]
    monkeypatch.setattr(torch.distributed,'all_gather_object',gather)
    def fail():
        raise ValueError('invalid dependency')
    with pytest.raises(RuntimeError,match='5.*ValueError: invalid dependency') as error:
        coordinator.gather_tuning_metadata(fail)
    assert isinstance(error.value.__cause__,ValueError)
