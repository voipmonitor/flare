/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Retained duration estimates; zero denotes an unmeasured graph cost.

use super::ScheduledStep;
use crate::{
    Result,
    execution::{ExecutionTiming, GraphDescriptor, GraphKind},
};

#[cfg(test)]
thread_local! {
    /// Cost-table lookups on this thread, for tests that bound planning work.
    pub(super) static ENTRY_LOOKUPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(crate) fn observed_duration_ns(timing: ExecutionTiming) -> Option<u64> {
    timing
        .gpu_ns
        .into_iter()
        .flatten()
        .max()
        .filter(|ns| *ns > 0)
        .map(|ns| if timing.queued { ns } else { ns.max(timing.wall_ns) })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Key {
    pub drafts: usize,
    pub kind: GraphKind,
    pub batch: usize,
    pub tokens: usize,
    pub prefill: usize,
    pub packed: usize,
}

impl Key {
    pub fn from_step(step: &ScheduledStep) -> Self {
        Self { drafts: step.draft_tokens, kind: step.kind, batch: step.batch_capacity, tokens: step.token_capacity,
            prefill: step.prefill_capacity(), packed: step.packed_token_capacity }
    }
}

pub(super) struct PackedRoundShape {
    pub batch: usize,
    pub drafts: usize,
    pub width: usize,
    pub tokens: usize,
    pub nanos: u64,
}

const PREFILL_HISTORY_BANDS: usize = 6;

/// Logarithmic KV-history bands for prefill duration estimation:
/// 0: [0, 16k), 1: [16k, 32k), 2: [32k, 64k), 3: [64k, 128k), 4: [128k, 256k), 5: [256k, ..).
fn prefill_history_band(start_position: u64) -> usize {
    if start_position < 16_384 {
        0
    } else {
        (1 + (start_position / 16_384).ilog2() as usize).min(PREFILL_HISTORY_BANDS - 1)
    }
}

#[derive(Default)]
struct BandEstimate {
    nanos: u64,
    replays: ReplayTimes,
    observed_at_ns: u64,
}

impl BandEstimate {
    fn fresh(&self, now_ns: u64, ttl_ns: u64) -> bool {
        self.nanos != 0 && now_ns.saturating_sub(self.observed_at_ns) < ttl_ns
    }
}

struct Estimate {
    nanos: u64,
    replays: ReplayTimes,
    observed_at_ns: u64,
    last_attempt_ns: u64,
    history_bands: [BandEstimate; PREFILL_HISTORY_BANDS],
}

#[derive(Default)]
struct ReplayTimes {
    samples: [u64; 9],
    len: usize,
    next: usize,
}

impl ReplayTimes {
    fn observe(&mut self, sample: u64, fresh: bool) -> u64 {
        if !fresh {
            *self = Self::default();
        }
        self.samples[self.next] = sample;
        self.next = (self.next + 1) % self.samples.len();
        self.len = (self.len + 1).min(self.samples.len());
        let mut ordered = self.samples;
        ordered[..self.len].sort_unstable();
        let middle = self.len / 2;
        if self.len % 2 == 0 {
            ((u128::from(ordered[middle - 1]) + u128::from(ordered[middle]) + 1) / 2) as u64
        } else {
            ordered[middle]
        }
    }
}

impl Estimate {
    fn fresh(&self, now_ns: u64, ttl_ns: u64) -> bool {
        self.nanos != 0 && now_ns.saturating_sub(self.observed_at_ns) < ttl_ns
    }
}

#[derive(Default)]
pub(super) struct ActionCosts {
    entries: Vec<(Key, Estimate)>,
    index: std::collections::HashMap<Key, usize>,
    packed_round_durations: [Vec<Option<u64>>; 7],
    now_ns: u64,
    ttl_ns: u64,
}

impl ActionCosts {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub fn set_measured(&mut self, step: &ScheduledStep, nanos: u64) {
        let key = Key {
            drafts: step.draft_tokens,
            kind: step.kind,
            batch: step.batch_capacity,
            tokens: step.token_capacity,
            prefill: step.prefill_capacity(),
            packed: step.packed_token_capacity,
        };
        let estimate = &mut self.entries[self.index[&key]].1;
        estimate.nanos = nanos;
        estimate.replays.observe(nanos, false);
        estimate.observed_at_ns = self.now_ns;
        estimate.history_bands = Default::default();
        if key.kind.is_prefill() && nanos > 0 {
            let band = &mut estimate.history_bands[0];
            band.nanos = nanos;
            band.replays.observe(nanos, false);
            band.observed_at_ns = self.now_ns;
        }
        self.refresh_packed_durations();
    }

    pub fn prepared(graphs: &[GraphDescriptor], token_capacity: usize) -> Result<Self> {
        let mut entries = Vec::with_capacity(graphs.len());
        let mut index = std::collections::HashMap::with_capacity(graphs.len());
        for graph in graphs {
            let key = Key {
                drafts: graph.draft_tokens,
                kind: graph.kind,
                batch: graph.batch_capacity,
                tokens: graph.token_capacity,
                prefill: graph.prefill_capacity,
                packed: graph.packed_token_capacity,
            };
            if index.insert(key, entries.len()).is_some() {
                return Err("prepared graphs require unique shapes".into());
            }
            entries.push((
                key,
                Estimate {
                    nanos: 0,
                    replays: ReplayTimes::default(),
                    observed_at_ns: 0,
                    last_attempt_ns: 0,
                    history_bands: Default::default(),
                },
            ));
        }
        let index = entries.iter().enumerate().map(|(index, (key, _))| (*key, index)).collect();
        let mut costs = Self { entries, index,
            packed_round_durations: std::array::from_fn(|_| vec![None; token_capacity + 1]),
            ttl_ns: super::PolicyConfig::default().estimate_ttl_ns, ..Default::default() };
        costs.refresh_packed_durations();
        Ok(costs)
    }

    pub fn advance_to(&mut self, now_ns: u64, ttl_ns: u64) {
        self.now_ns = now_ns;
        self.ttl_ns = ttl_ns;
    }

    pub fn duration(&self, kind: GraphKind, batch: usize, tokens: usize) -> Option<u64> {
        self.shape_duration(kind, batch, tokens, 0)
    }

    pub fn prefill_duration(
        &self,
        kind: GraphKind,
        tokens: usize,
        start_position: u64,
    ) -> Option<u64> {
        let estimate = self.entry(Key {
            drafts: 0,
            kind,
            batch: 1,
            tokens,
            prefill: 0,
            packed: 0,
        })?;
        if estimate.nanos == 0 {
            return Some(0);
        }
        let band = prefill_history_band(start_position);
        if estimate.history_bands[band].nanos > 0 {
            return Some(estimate.history_bands[band].nanos);
        }
        let mut src: Option<(usize, u64)> = None;
        for b in 0..PREFILL_HISTORY_BANDS {
            let ns = estimate.history_bands[b].nanos;
            if ns > 0 && src.is_none_or(|(old_b, _)| b.abs_diff(band) < old_b.abs_diff(band)) {
                src = Some((b, ns));
            }
        }
        let (src_band, src_ns) = src.unwrap_or((0, estimate.nanos));
        if src_band == band {
            return Some(src_ns);
        }
        // When scaling upward (`band > src_band`), only peers with strictly larger capacity
        // supply a valid slowdown lower bound (smaller chunks amortize per-launch KV overhead
        // less effectively at long context). This prevents both a downward spiral onto stale
        // short-context smaller chunks AND trapping on a smaller chunk without testing larger ones.
        let mut peer: Option<(usize, u64, u64)> = None;
        for (peer_key, peer_est) in &self.entries {
            if peer_key.kind != kind || peer_key.batch != 1 || peer_key.prefill != 0 {
                continue;
            }
            if band > src_band && peer_key.tokens <= tokens {
                continue;
            }
            let target_ns = peer_est.history_bands[band].nanos;
            let base_ns = if src_band == 0 && peer_est.history_bands[0].nanos == 0 {
                peer_est.nanos
            } else {
                peer_est.history_bands[src_band].nanos
            };
            if target_ns > 0
                && base_ns > 0
                && peer.is_none_or(|(old_tokens, _, _)| peer_key.tokens > old_tokens)
            {
                peer = Some((peer_key.tokens, target_ns, base_ns));
            }
        }
        let Some((_, target_ns, base_ns)) = peer else {
            return Some(src_ns);
        };
        let (num, den) = if band > src_band {
            (target_ns.max(base_ns), base_ns)
        } else {
            (target_ns.min(base_ns), base_ns)
        };
        let scaled = ((u128::from(src_ns) * u128::from(num) + u128::from(den) / 2)
            / u128::from(den))
            .max(1)
            .min(u128::from(u64::MAX)) as u64;
        Some(scaled)
    }

    fn refresh_packed_durations(&mut self) {
        for row in &mut self.packed_round_durations { row.fill(None); }
        for (key, estimate) in &self.entries {
            if key.kind != GraphKind::Speculative || key.packed == 0 { continue; }
            let maximum = (key.drafts + 1).min(key.tokens).min(key.packed);
            let nanos = estimate.nanos;
            for duration in self.packed_round_durations[key.batch.trailing_zeros() as usize].iter_mut().take(maximum + 1).skip(1) {
                *duration = Some(duration.map_or(nanos, |old| old.min(nanos)));
            }
        }
    }

    pub fn packed_round_duration(&self, batch: usize, width: usize) -> Option<u64> {
        if !batch.is_power_of_two() || batch > super::REQUEST_LANES { return None; }
        self.packed_round_durations.get(batch.trailing_zeros() as usize)?.get(width).copied().flatten()
    }

    pub fn packed_round_shapes(&self) -> impl Iterator<Item = PackedRoundShape> + '_ {
        self.entries.iter().filter(|(key, _)| key.kind == GraphKind::Speculative && key.packed != 0)
            .map(|(key, estimate)| PackedRoundShape {
                batch: key.batch, drafts: key.drafts, width: key.tokens, tokens: key.packed,
                nanos: estimate.nanos,
            })
    }

    pub fn shape_duration(
        &self,
        kind: GraphKind,
        batch: usize,
        tokens: usize,
        prefill: usize,
    ) -> Option<u64> {
        self.exact_duration(kind, batch, tokens, prefill, 0)
    }

    fn exact_duration(&self, kind: GraphKind, batch: usize, tokens: usize, prefill: usize, packed: usize) -> Option<u64> {
        self.entry(Key {
            drafts: if kind == GraphKind::Speculative { tokens - 1 } else { 0 },
            kind,
            batch,
            tokens,
            prefill,
            packed: if kind == GraphKind::Speculative { batch * tokens } else { packed },
        })
        .map(|estimate| estimate.nanos)
    }

    pub fn duration_key(&self, key: Key) -> Option<u64> {
        self.entry(key).map(|estimate| estimate.nanos)
    }

    pub fn is_fresh(&self, key: Key) -> bool {
        self.entry(key).is_some_and(|estimate| estimate.fresh(self.now_ns, self.ttl_ns))
    }

    pub fn fresh_duration(&self, kind: GraphKind, tokens: usize) -> Option<u64> {
        let key = Key { drafts: 0, kind, batch: 1, tokens, prefill: 0, packed: 0 };
        self.entry(key).filter(|estimate| estimate.fresh(self.now_ns, self.ttl_ns))
            .map(|estimate| estimate.nanos)
    }

    pub fn stale_keys(&self) -> impl Iterator<Item = Key> + '_ {
        self.entries.iter().filter(|(_, estimate)| estimate.nanos != 0
            && !estimate.fresh(self.now_ns, self.ttl_ns)).map(|(key, _)| *key)
    }

    pub fn refresh_order(&self, key: Key) -> u64 {
        self.entry(key).map_or(0, |estimate| estimate.observed_at_ns.max(estimate.last_attempt_ns))
    }

    pub fn record_attempt(&mut self, key: Key) {
        if let Some(&index) = self.index.get(&key) {
            let estimate = &mut self.entries[index].1;
            estimate.last_attempt_ns = self.now_ns;
        }
    }

    pub fn observation(&self, key: Key) -> Option<(u64, Option<u64>, bool)> {
        self.entry(key).map(|estimate| (estimate.nanos,
            (estimate.nanos != 0).then_some(estimate.observed_at_ns),
            estimate.fresh(self.now_ns, self.ttl_ns)))
    }

    fn entry(&self, key: Key) -> Option<&Estimate> {
        #[cfg(test)]
        ENTRY_LOOKUPS.set(ENTRY_LOOKUPS.get() + 1);
        self.index.get(&key).map(|&index| &self.entries[index].1)
    }

    pub fn get(&self, step: &ScheduledStep) -> Option<u64> {
        self.duration_key(Key::from_step(step))
    }

    pub fn observe(&mut self, step: &ScheduledStep, timing: ExecutionTiming) {
        let Some(raw_sample) = timing.gpu_ns.into_iter().flatten().max().filter(|n| *n > 0) else {
            return;
        };
        let sample = raw_sample.saturating_sub(timing.readiness.map_or(0, |intervals| intervals.wait_ns()));
        if sample == 0 {
            return;
        }
        let key = Key {
            drafts: step.draft_tokens,
            kind: step.kind,
            batch: step.batch_capacity,
            tokens: step.token_capacity,
            prefill: step.prefill_capacity(),
            packed: step.packed_token_capacity,
        };
        if let Some(&index) = self.index.get(&key) {
            let estimate = &mut self.entries[index].1;
            let fresh = estimate.fresh(self.now_ns, self.ttl_ns);
            if key.kind.is_prefill() {
                let start_position = step.lanes.first().map_or(0, |lane| lane.start_position);
                let band = prefill_history_band(start_position);
                let band_estimate = &mut estimate.history_bands[band];
                let band_fresh = band_estimate.fresh(self.now_ns, self.ttl_ns);
                band_estimate.nanos = band_estimate.replays.observe(sample, band_fresh);
                band_estimate.observed_at_ns = self.now_ns;
                if band == 0 || estimate.nanos == 0 {
                    estimate.nanos = estimate.replays.observe(sample, fresh);
                }
                estimate.observed_at_ns = self.now_ns;
            } else {
                estimate.nanos = estimate.replays.observe(sample, fresh);
                estimate.observed_at_ns = self.now_ns;
                self.refresh_packed_durations();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ActionCosts, ReplayTimes, prefill_history_band};
    use crate::{
        execution::{ExecutionTiming, GraphDescriptor, GraphKind, SamplingMode},
        scheduler::{RequestHandle, ScheduledLane, ScheduledStep},
    };

    fn prefill_graph(kind: GraphKind, tokens: usize) -> GraphDescriptor {
        GraphDescriptor {
            tuned_bindings: vec![],
            confidence_width: 0,
            draft_tokens: 0,
            packed_token_capacity: 0,
            name: format!("{kind:?}-{tokens}"),
            kind,
            sampling_mode: SamplingMode::General,
            handle: 1,
            batch_capacity: 1,
            token_capacity: tokens,
            prefill_capacity: 0,
            estimated_duration_ns: 0,
            input_bindings: vec![],
        }
    }

    fn prefill_step(kind: GraphKind, tokens: usize, start_position: u64) -> ScheduledStep {
        ScheduledStep {
            draft_tokens: 0,
            packed_token_capacity: 0,
            kind,
            batch_capacity: 1,
            token_capacity: tokens,
            mixed: None,
            lanes: vec![ScheduledLane {
                request: RequestHandle { slot: 0, generation: 1 },
                start_position,
                tokens,
            }],
            requires_feedback: true,
        }
    }

    fn gpu_timing(ns: u64) -> ExecutionTiming {
        let mut gpu_ns = [None; 16];
        gpu_ns[0] = Some(ns);
        ExecutionTiming { gpu_ns, readiness: None, wall_ns: ns, queued: false }
    }

    #[test]
    fn long_context_prefill_observations_do_not_pollute_short_band_and_scale_unmeasured_peers() {
        assert_eq!(prefill_history_band(0), 0);
        assert_eq!(prefill_history_band(16_383), 0);
        assert_eq!(prefill_history_band(16_384), 1);
        assert_eq!(prefill_history_band(32_768), 2);
        assert_eq!(prefill_history_band(65_536), 3);
        assert_eq!(prefill_history_band(131_072), 4);

        let graphs = vec![
            prefill_graph(GraphKind::Prefill, 4096),
            prefill_graph(GraphKind::Prefill, 8192),
        ];
        let mut costs = ActionCosts::prepared(&graphs, 16).unwrap();
        costs.observe(&prefill_step(GraphKind::Prefill, 4096, 0), gpu_timing(110_000_000));
        costs.observe(&prefill_step(GraphKind::Prefill, 8192, 0), gpu_timing(210_000_000));
        for _ in 0..9 {
            costs.observe(&prefill_step(GraphKind::Prefill, 8192, 80_000), gpu_timing(252_000_000));
        }
        assert_eq!(costs.prefill_duration(GraphKind::Prefill, 8192, 0), Some(210_000_000));
        assert_eq!(costs.prefill_duration(GraphKind::Prefill, 4096, 0), Some(110_000_000));
        assert_eq!(costs.prefill_duration(GraphKind::Prefill, 8192, 80_000), Some(252_000_000));
        assert_eq!(costs.prefill_duration(GraphKind::Prefill, 4096, 80_000), Some(132_000_000));

        costs.observe(&prefill_step(GraphKind::Prefill, 4096, 140_000), gpu_timing(160_000_000));
        assert_eq!(costs.prefill_duration(GraphKind::Prefill, 4096, 140_000), Some(160_000_000));
        assert_eq!(costs.prefill_duration(GraphKind::Prefill, 8192, 140_000), Some(252_000_000));
    }

    #[test]
    fn isolated_slow_replay_does_not_displace_the_steady_cost() {
        let mut replays = ReplayTimes::default();
        for _ in 0..9 {
            assert_eq!(replays.observe(8_000_000, true), 8_000_000);
        }
        assert_eq!(replays.observe(16_000_000, true), 8_000_000);
        for _ in 0..9 {
            assert_eq!(replays.observe(8_000_000, true), 8_000_000);
        }
    }

    #[test]
    fn sustained_cost_changes_replace_the_window_and_staleness_resets_it() {
        let mut replays = ReplayTimes::default();
        for _ in 0..9 {
            replays.observe(8, true);
        }
        for _ in 0..4 {
            assert_eq!(replays.observe(16, true), 8);
        }
        assert_eq!(replays.observe(16, true), 16);
        assert_eq!(replays.observe(32, false), 32);
        assert_eq!(replays.observe(16, true), 24);
    }

    #[test]
    fn even_sample_median_does_not_overflow() {
        let mut replays = ReplayTimes::default();
        assert_eq!(replays.observe(u64::MAX, true), u64::MAX);
        assert_eq!(replays.observe(u64::MAX - 1, true), u64::MAX);
    }
}
