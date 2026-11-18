// SPDX-License-Identifier: Apache-2.0
//! Select one of two preinstantiated speculative graph handles on a single-host C1 benchmark.
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

pub const CONTROL_PREFIX: &str = "paired-control--";

/// The controller changes this selector only between drained requests. The root
/// executor additionally latches it per step so every rank launches the same role.
#[derive(Debug)]
pub struct GraphComparison {
    names: Vec<String>,
    candidate: AtomicBool,
    launches: Vec<Vec<[AtomicU64; 2]>>,
}

impl GraphComparison {
    pub(crate) fn new(names: Vec<String>, ranks: usize) -> Arc<Self> {
        let launches = (0..ranks)
            .map(|_| {
                (0..names.len())
                    .map(|_| [AtomicU64::new(0), AtomicU64::new(0)])
                    .collect()
            })
            .collect();
        Arc::new(Self {
            names,
            candidate: AtomicBool::new(false),
            launches,
        })
    }
    pub(crate) fn names(&self) -> &[String] {
        &self.names
    }
    pub(crate) fn candidate(&self) -> bool {
        self.candidate.load(Ordering::Acquire)
    }
    pub(crate) fn select(&self, candidate: bool) {
        self.candidate.store(candidate, Ordering::Release);
    }
    pub(crate) fn counts(&self) -> Vec<Vec<[u64; 2]>> {
        self.launches
            .iter()
            .map(|rank| {
                rank.iter()
                    .map(|pair| {
                        [
                            pair[0].load(Ordering::Relaxed),
                            pair[1].load(Ordering::Relaxed),
                        ]
                    })
                    .collect()
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct RankGraphComparison {
    pub(crate) session: Arc<GraphComparison>,
    rank: usize,
    // Logical serving index -> (alternative handle, observable graph ordinal).
    control: Vec<Option<(usize, usize)>>,
}

impl RankGraphComparison {
    pub(crate) fn new(
        session: Arc<GraphComparison>,
        rank: usize,
        control: Vec<Option<(usize, usize)>>,
    ) -> Self {
        Self {
            session,
            rank,
            control,
        }
    }

    pub(crate) fn compatible_peer(&self, peer: &Self) -> bool {
        self.rank != peer.rank
            && Arc::ptr_eq(&self.session, &peer.session)
            && self.control.len() == peer.control.len()
            && self
                .control
                .iter()
                .zip(&peer.control)
                .all(|(a, b)| a.map(|(_, id)| id) == b.map(|(_, id)| id))
    }

    pub(crate) fn handle(&self, index: usize, candidate_handle: usize, candidate: bool) -> usize {
        if candidate {
            return candidate_handle;
        }
        self.control[index].map_or(candidate_handle, |(handle, _)| handle)
    }

    pub(crate) fn record(&self, index: usize, candidate: bool) {
        if let Some((_, graph)) = self.control[index] {
            self.session.launches[self.rank][graph][usize::from(candidate)]
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latched_role_preserves_all_rank_choices_and_counts_only_paired_launches() {
        let session = GraphComparison::new(vec!["speculative4".into()], 2);
        let ranks = [
            RankGraphComparison::new(session.clone(), 0, vec![None, Some((100, 0))]),
            RankGraphComparison::new(session.clone(), 1, vec![None, Some((200, 0))]),
        ];
        let first_step = session.candidate();
        session.select(true);
        let next_step = session.candidate();
        assert!(!first_step && next_step);
        for (i, rank) in ranks.iter().enumerate() {
            let candidate = 10 + i;
            assert_eq!(rank.handle(1, candidate, first_step), (i + 1) * 100);
            rank.record(1, first_step);
            assert_eq!(rank.handle(1, candidate, next_step), candidate);
            rank.record(1, next_step);
            assert_eq!(rank.handle(0, candidate, first_step), candidate);
            rank.record(0, first_step);
        }
        assert_eq!(session.counts(), vec![vec![[1, 1]], vec![[1, 1]]]);
    }
}
