// SPDX-License-Identifier: Apache-2.0
//! Each instantiated CUDA graph has one owner even when two logical roles share it.
use super::{driver::Handle, format::invalid};
use crate::Result;

#[derive(Debug)]
enum Entry {
    Owned(Handle),
    Reference(usize),
}

#[derive(Debug,Default)]
pub(super) struct GraphHandles(Vec<Entry>);

impl GraphHandles {
    pub fn len(&self) -> usize { self.0.len() }
    pub fn push(&mut self, handle:Handle) { self.0.push(Entry::Owned(handle)); }

    pub fn get(&self,index:usize)->Result<Handle>{
        match self.0.get(index) {
            Some(Entry::Owned(handle))=>Ok(*handle),
            Some(Entry::Reference(owner))=>match self.0.get(*owner) {
                Some(Entry::Owned(handle))=>Ok(*handle),
                _=>Err(invalid("borrowed CUDA graph has no permanent owner")),
            },
            None=>Err(invalid("CUDA graph index is out of range")),
        }
    }

    /// Move ownership of the initial handle to the immutable control entry.
    /// The candidate can then be replaced without touching the control.
    pub fn append_owning_control(&mut self, candidate:usize)->Result<()> {
        let owner=self.0.len();
        let Some(Entry::Owned(handle))=self.0.get(candidate) else {
            return Err(invalid("initial comparison graph must have a unique owner"));
        };
        let handle=*handle;
        self.0[candidate]=Entry::Reference(owner);
        self.0.push(Entry::Owned(handle));
        Ok(())
    }

    pub fn retire(&mut self,candidate:usize,control:usize,release:impl FnOnce(Handle)->Result<()>)->Result<()> {
        if candidate==control || !matches!(self.0.get(control),Some(Entry::Owned(_))) {
            return Err(invalid("a resident control graph must own its executable"));
        }
        match self.0.get(candidate) {
            Some(Entry::Owned(value))=>release(*value)?,
            Some(Entry::Reference(source)) if *source==control=>return Ok(()),
            _=>return Err(invalid("invalid candidate graph ownership")),
        }
        self.0[candidate]=Entry::Reference(control);
        Ok(())
    }

    pub fn install(&mut self,candidate:usize,control:usize,handle:Handle) {
        // The complete installation is validated before the first publication.
        assert!(matches!(self.0[candidate],Entry::Reference(i) if i==control));
        self.0[candidate]=Entry::Owned(handle);
    }

    pub fn take_prepared(&mut self,start:usize)->Vec<Handle> {
        self.0.split_off(start).into_iter().map(|entry|match entry {
            Entry::Owned(h)=>h,
            Entry::Reference(_)=>unreachable!("only ordinary candidate graphs are independently constructed"),
        }).collect()
    }

    pub fn owned(&self)->impl DoubleEndedIterator<Item=Handle> + '_ {
        self.0.iter().filter_map(|entry|match entry {
            Entry::Owned(h)=>Some(*h),Entry::Reference(_)=>None
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repeated_install_retires_only_the_candidate_and_preserves_reference_on_error() {
        let h=|n:usize|n as Handle;
        let mut table=GraphHandles::default();
        table.push(h(101));table.append_owning_control(0).unwrap();
        assert_eq!(table.owned().collect::<Vec<_>>(),[h(101)]);
        assert_eq!(table.get(0).unwrap(),table.get(1).unwrap());
        let mut freed=Vec::new();
        table.retire(0,1,|v|{freed.push(v);Ok(())}).unwrap();
        assert!(freed.is_empty());
        table.install(0,1,h(202));
        assert!(table.retire(0,1,|_|Err(invalid("driver failure"))).is_err());
        assert_eq!(table.get(0).unwrap(),h(202));
        table.retire(0,1,|v|{freed.push(v);Ok(())}).unwrap();
        table.retire(0,1,|v|{freed.push(v);Ok(())}).unwrap();
        assert_eq!(freed,[h(202)]);assert_eq!(table.owned().collect::<Vec<_>>(),[h(101)]);
    }
}
