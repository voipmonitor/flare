// SPDX-License-Identifier: Apache-2.0
//! Select one of two preinstantiated speculative graph handles on a single-host C1 benchmark.
use std::sync::{
    Arc,Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

pub const CONTROL_PREFIX: &str = "paired-control--";

/// The controller changes this selector only between drained requests. The root
/// executor additionally latches it per step so every rank launches the same role.
#[derive(Debug)]
pub struct GraphComparison {
    names: Vec<String>,
    candidate: AtomicBool,
    launches: Vec<Vec<[AtomicU64; 2]>>,
    overrides: Vec<Vec<AtomicUsize>>,
    loaded: Mutex<Option<String>>,
    available: AtomicBool,
    gate: Mutex<Gate>,

}

#[derive(Debug,Default)]
struct Gate { closed:bool, outstanding:usize, broken:bool }

pub(crate) struct Quiescent<'a>(&'a GraphComparison);
impl Drop for Quiescent<'_> {
    fn drop(&mut self) { self.0.gate.lock().unwrap_or_else(|e|e.into_inner()).closed=false; }
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
        let overrides=(0..ranks).map(|_|(0..names.len()).map(|_|AtomicUsize::new(0)).collect()).collect();
        Arc::new(Self {
            overrides, loaded: Mutex::new(None), available:AtomicBool::new(true), gate:Mutex::default(),
            names,
            candidate: AtomicBool::new(false),
            launches,
        })
    }
    pub(crate) fn try_begin_frame(&self) -> crate::Result<bool> {
        let mut g=self.gate.lock().unwrap_or_else(|e|e.into_inner());
        if g.broken { return Err("comparison CUDA drain could not be confirmed".into()); }
        if g.closed { return Ok(false); }
        g.outstanding+=1;Ok(true)
    }
    pub(crate) fn finish_frame(&self) -> crate::Result<()> {
        let mut g=self.gate.lock().unwrap_or_else(|e|e.into_inner());
        g.outstanding=g.outstanding.checked_sub(1).ok_or("unexpected comparison completion")?;Ok(())
    }
    pub(crate) fn quiesce(&self) -> crate::Result<Quiescent<'_>> {
        let mut g=self.gate.lock().unwrap_or_else(|e|e.into_inner());
        if g.closed||g.outstanding!=0||g.broken {return Err("comparison has pending frames or an unconfirmed drain".into());}
        g.closed=true;Ok(Quiescent(self))
    }
    pub(crate) fn healthy(&self)->bool{!self.gate.lock().unwrap_or_else(|e|e.into_inner()).broken}
    pub(crate) fn poison(&self){self.gate.lock().unwrap_or_else(|e|e.into_inner()).broken=true;}
    pub(crate) fn initial(&self,id:String){*self.loaded.lock().unwrap_or_else(|e|e.into_inner())=Some(id);}
    pub(crate) fn disable(&self){
        self.select(false);self.available.store(false,Ordering::Release);
        *self.loaded.lock().unwrap_or_else(|e|e.into_inner())=None;
    }
    pub(crate) fn available(&self)->bool{self.available.load(Ordering::Acquire)}
    pub(crate) fn loaded(&self)->Option<String>{self.loaded.lock().unwrap_or_else(|e|e.into_inner()).clone()}
    pub(crate) fn install(&self,id:String,handles:&[Vec<usize>]) {
        for (rank,new) in self.overrides.iter().zip(handles) {
            for (slot,&handle) in rank.iter().zip(new) { slot.store(handle,Ordering::Release); }
        }
        *self.loaded.lock().unwrap_or_else(|e|e.into_inner())=Some(id);
        self.available.store(true,Ordering::Release);
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
        let Some((control,ordinal))=self.control[index] else {return candidate_handle;};
        if !candidate || !self.session.available() {return control;}
        let handle=self.session.overrides[self.rank][ordinal].load(Ordering::Acquire);
        if handle==0 {candidate_handle} else {handle}
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
    fn a_replacement_gate_requires_positive_completion_and_commits_one_generation() {
        let shared=GraphComparison::new(vec!["speculative4".into()],2);
        let r0=RankGraphComparison::new(shared.clone(),0,vec![Some((11,0))]);
        let r1=RankGraphComparison::new(shared.clone(),1,vec![Some((12,0))]);
        assert!(shared.try_begin_frame().unwrap());
        assert!(shared.quiesce().is_err());
        shared.finish_frame().unwrap();
        let ready=shared.quiesce().unwrap();
        assert!(!shared.try_begin_frame().unwrap());
        shared.disable();
        assert!(!shared.available());
        assert_eq!(r0.handle(0,15,true),11);
        assert_eq!(r1.handle(0,16,true),12);
        shared.install("tested".into(),&[vec![21],vec![22]]);
        drop(ready);
        assert_eq!(r0.handle(0,15,true),21);
        assert_eq!(r1.handle(0,16,true),22);
        assert!(shared.try_begin_frame().unwrap());
        shared.finish_frame().unwrap();
        shared.poison();
        assert!(shared.try_begin_frame().is_err());
        assert!(shared.quiesce().is_err());
    }

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
