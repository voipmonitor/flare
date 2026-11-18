// SPDX-License-Identifier: Apache-2.0
//! Authenticate executable-only replacements for one paired native residency.
use super::format::{invalid, Core, Placement};
use super::payload::{ExecutableFiles, RecordReader};
use super::protobuf;
use super::Publisher;
use crate::core::{protection, reader::Metadata, schema::wire};
use crate::execution::graph_comparison::CONTROL_PREFIX;
use crate::Result;
use std::{collections::{BTreeMap,BTreeSet,HashMap},io::Read,path::{Path,PathBuf}};

#[derive(Clone)]
pub(super) struct Origin {
    pub path: PathBuf,
    pub envelope_digest: [u8;32],
    pub core_id:[u8;32],
    pub publisher: Publisher,
}

pub(super) struct Images(pub HashMap<String,Vec<u8>>);
impl ExecutableFiles for Images {
    fn image(&self,_: &Core,name:&str)->Result<Vec<u8>> {
        self.0.get(name).cloned().ok_or_else(||invalid(format!("missing verified program image {name}")))
    }
    fn graph_records<'a>(&'a self,core:&'a Core,name:&str)->Result<RecordReader<'a>> {
        Ok(RecordReader::Slice{bytes:self.image(core,name)?,position:0})
    }
}

pub(super) struct RankPlan {
    pub core: Core,
    pub images: Images,
    /// Challenger function ID and resident function ID with identical signed definition.
    pub reused: Vec<(usize,usize)>,
    pub load_functions:BTreeSet<usize>,
}
pub(super) struct Plan {
    pub id:String,
    pub ranks:Vec<RankPlan>,
    pub executable_bytes:u64,
    pub companions:Vec<String>,
}

pub(super) fn sha_hex(bytes:&[u8])->String { bytes.iter().map(|x|format!("{x:02x}")).collect() }

fn resource_same(a:&wire::ExecutableResource,b:&wire::ExecutableResource)->bool {
    let mut q=a.clone();q.resource_id=b.resource_id;q==*b
}
fn matching_function(a:&wire::RankBody, ai:usize,b:&wire::RankBody,bi:usize)->bool {
    let (Some(x),Some(y))=(a.functions.get(ai),b.functions.get(bi)) else{return false};
    if x!=y {return false}
    let (Some(m),Some(n))=(a.modules.iter().find(|m|m.name==x.module),b.modules.iter().find(|m|m.name==y.module)) else{return false};
    if m!=n {return false}
    let same=|i:u32,j:u32| a.resources.get(i as usize).zip(b.resources.get(j as usize)).is_some_and(|(s,t)|resource_same(s,t));
    same(m.cubin,n.cubin)
}
fn underlying<'a>(r:&'a wire::RankBody,name:&str)->Option<&'a wire::Graph>{
    let alias=format!("{CONTROL_PREFIX}{name}");
    r.graphs.iter().find(|g|g.name==alias).or_else(||r.graphs.iter().find(|g|g.name==name))
}

fn check_rank_state(a:&wire::RankBody,b:&wire::RankBody)->Result<()> {
    let mut m=b.clone();
    m.graphs.clone_from(&a.graphs);m.modules.clone_from(&a.modules);m.functions.clone_from(&a.functions);
    m.resources.clone_from(&a.resources);m.payloads.clone_from(&a.payloads);
    if m!=*a {return Err(invalid("replacement changes frozen weights, state, allocation, peer or execution bindings"));}
    let names=|r:&wire::RankBody|r.graphs.iter().filter(|g|!g.name.starts_with(CONTROL_PREFIX)).map(|g|g.name.clone()).collect::<BTreeSet<_>>();
    for r in [a,b] {
        let unique:BTreeSet<_>=r.graphs.iter().map(|g|g.name.as_str()).collect();
        if unique.len()!=r.graphs.len(){return Err(invalid("duplicate named replacement graph"));}
        for alias in r.graphs.iter().filter_map(|g|g.name.strip_prefix(CONTROL_PREFIX)) {
            if !unique.contains(alias) || !r.execution.as_ref().is_some_and(|e|e.graphs.iter().any(|g|g.name==alias && g.kind=="speculative")){
                return Err(invalid("replacement control lacks its signed speculative operation"));
            }
        }
    }
    if names(a)!=names(b){return Err(invalid("replacement serving graph names differ"));}
    let mut checked=BTreeSet::new();
    for n in names(a) {
        let x=underlying(a,&n).ok_or_else(||invalid("resident control missing"))?;
        let y=underlying(b,&n).ok_or_else(||invalid("replacement control missing"))?;
        if (x.flags,x.node_count,x.edge_count,&x.functions)!=(y.flags,y.node_count,y.edge_count,&y.functions)
            || !resource_same(&a.resources[x.resource as usize],&b.resources[y.resource as usize]){
            return Err(invalid(format!("signed control frame differs for {n}")));
        }
        for &i in &x.functions { if checked.insert(i) && !matching_function(a,i as usize,b,i as usize) {
            return Err(invalid(format!("signed control program differs at {i}")));
        }}
    }
    Ok(())
}

pub(super) fn prepare(origin:&Origin,new_root:&Path,expected_id:&str,slots:&[String],available:&[BTreeSet<usize>])->Result<Plan> {
    let old_meta=Metadata::open(&origin.path)?;
    if protection::sha256(&old_meta.envelope)!=origin.envelope_digest{return Err(invalid("resident signed core changed on disk"));}
    let select=protection::Selection::Deployment{deployment:0,rank:None};
    let keys=std::slice::from_ref(&origin.publisher);
    let old=protection::verify(&old_meta.envelope,keys,select)?;
    let meta=Metadata::open(new_root)?;
    let new=protection::verify(&meta.envelope,keys,select)?;
    let id=sha_hex(&new.core_id());
    if id!=expected_id || !origin.publisher.development || new.deployment_count()!=1 || new.ranks().len()!=16 || old.ranks().len()!=16 {
        return Err(invalid("replacement must be the requested signed local TP16 comparison core"));
    }
    let mut top=new.core().clone();top.core_id.clone_from(&old.core().core_id);
    if top.deployments.len()!=1{return Err(invalid("replacement deployment count differs"));}
    top.deployments[0].body_digest.clone_from(&old.core().deployments[0].body_digest);
    if top!=*old.core(){return Err(invalid("replacement model, hardware, publisher or sampling contract differs"));}
    let mut deployment=new.deployment().ok_or_else(||invalid("replacement deployment missing"))?.clone();
    let baseline=old.deployment().ok_or_else(||invalid("resident deployment missing"))?;
    if deployment.ranks.len()!=baseline.ranks.len(){return Err(invalid("replacement rank count differs"));}
    for (n,o) in deployment.ranks.iter_mut().zip(&baseline.ranks) {n.body_digest.clone_from(&o.body_digest);}
    if &deployment!=baseline{return Err(invalid("replacement deployment contract differs"));}
    let arch:[i32;2]=deployment.architecture.as_slice().try_into().map_err(|_|invalid("invalid architecture"))?;
    let mut placements=BTreeMap::new();
    for key in new.catalog().ids(){
        placements.insert(key.to_vec(),Placement{token:sha_hex(key),offset:0,nbytes:new.catalog().length(key)?});
    }
    let allowed:BTreeSet<_>=slots.iter().cloned().collect();
    let mut needs=BTreeSet::new();
    let mut descriptors=Vec::new();let mut specs=Vec::new();let mut catalog=None;
    for (rank,(a,b)) in old.ranks().iter().zip(new.ranks()).enumerate() {
        check_rank_state(a,b)?;
        let changed:BTreeSet<_>=b.graphs.iter().filter_map(|g|g.name.strip_prefix(CONTROL_PREFIX).map(str::to_owned)).collect();
        if !changed.is_subset(&allowed) || catalog.as_ref().is_some_and(|prior|prior!=&changed){
            return Err(invalid("replacement graph names exceed reserved slots or differ between ranks"));
        }
        catalog=Some(changed.clone());
        let mut core=protobuf::payload_rank(b,arch,&placements)?;
        core.graphs.retain(|g|changed.contains(&g.name));
        let mut need_images:BTreeSet<String>=core.graphs.iter().map(|g|g.resource.clone()).collect();
        let used:BTreeSet<usize>=core.graphs.iter().flat_map(|g|g.functions.iter().copied()).collect();
        let mut reused=Vec::new();let mut fresh=BTreeSet::new();
        for &i in &used {
            if available[rank].contains(&i) && matching_function(a,i,b,i) {
                reused.push((i,i));
            } else {
                fresh.insert(i);
            }
        }
        // Graph validation needs the parameter ABI of every referenced function.
        // Only the subset with no resident match gets a new CUfunction.
        core.functions.retain(|f|used.contains(&f.id));
        let needed_modules:BTreeSet<String>=core.functions.iter().filter(|f|fresh.contains(&f.id)).map(|f|f.module.clone()).collect();
        core.modules.retain(|m|needed_modules.contains(&m.name));
        for m in &core.modules {
            if !m.globals.is_empty(){return Err(invalid("a new comparison module requires persistent initialized globals"));}
            need_images.insert(m.file.clone());
        }
        let resources:BTreeMap<_,_>=b.resources.iter().filter_map(|r|{
            let name=format!("executable/{}",r.resource_id);
            need_images.contains(&name).then_some((name,r.clone()))
        }).collect();
        if resources.len()!=need_images.len(){return Err(invalid("replacement executable resource closure is incomplete"));}
        for res in resources.values(){needs.insert(res.stored.as_ref().ok_or_else(||invalid("missing stored program bytes"))?.payload_id.clone());}
        descriptors.push(resources);specs.push((core,reused,fresh));
    }
    let companions:BTreeSet<String>=meta.locations.locations.iter()
        .filter(|l|needs.contains(&l.payload_id) && !l.file.is_empty()).map(|l|l.file.clone()).collect();
    let payloads=meta.payloads(new.catalog(),needs.iter().map(Vec::as_slice))?;
    let authorized=new.authorize(&origin.publisher.namespace)?;
    let mut ranks=Vec::new();let mut executable_bytes=0u64;
    for ((core,reused,load_functions),resources) in specs.into_iter().zip(descriptors) {
        let mut images=HashMap::new();
        for (name,res) in resources {
            let stored=res.stored.as_ref().unwrap();
            let slice=payloads.get(&stored.payload_id)?;
            let mut encoded=vec![0u8;usize::try_from(stored.nbytes).map_err(|_|invalid("program extent overflow"))?];
            slice.read_at(stored.offset,&mut encoded)?;
            let mut reader=authorized.open_reader(&res,std::io::Cursor::new(encoded))?;
            let mut decoded=Vec::new();
            reader.read_to_end(&mut decoded).map_err(|e|invalid(format!("read signed executable: {e}")))?;
            if decoded.len() as u64!=res.decoded_nbytes{return Err(invalid("signed executable length differs"));}
            executable_bytes=executable_bytes.checked_add(stored.nbytes).ok_or_else(||invalid("total program size overflow"))?;
            images.insert(name,decoded);
        }
        let images=Images(images);
        for g in &core.graphs {
            let mut reader=images.graph_records(&core,&g.resource)?;
            let declared:BTreeSet<_>=g.functions.iter().copied().collect();
            let rank=ranks.len();
            for ordinal in 0..g.node_count {
                let node=protobuf::node(reader.read()?)?;
                let function=if let super::format::Node::Kernel{function,..}=&node{Some(*function)}else{None};
                if function.is_some_and(|id|!declared.contains(&id)) {
                    return Err(invalid(format!("rank {rank} graph {} node {ordinal} undeclared function {function:?}",g.name)));
                }
                core.validate_node(&node).map_err(|err|invalid(format!("rank {rank} graph {} node {ordinal} function {function:?}: {err}",g.name)))?;
            }
            let mut deps=super::format::Dependencies::new(g.node_count);
            for ordinal in 0..g.edge_count {
                let edge=protobuf::edge(reader.read()?)?;
                deps.add(&edge).map_err(|err|invalid(format!("rank {rank} graph {} edge {ordinal}: {err}",g.name)))?;
            }
            deps.finish()?;reader.finish()?;
        }
        ranks.push(RankPlan{core,images,reused,load_functions});
    }
    Ok(Plan{id,ranks,executable_bytes,companions:companions.into_iter().collect()})
}

struct PendingImages {
    names:Vec<String>,
    executables:Vec<super::driver::Handle>,
    templates:Vec<super::driver::Handle>,
    modules:Vec<super::driver::Handle>,
}

// SAFETY: Rank workers only construct handles in their bound context. The
// coordinator returns each result to that same rank for publication or cleanup.
unsafe impl Send for PendingImages {}

impl super::NativeResources {
    fn bind_drained(&self)->Result<()> {
        // SAFETY: The session owns the retained context and closes submission
        // before these context-wide drains.
        unsafe {
            self.driver.check("cuCtxSetCurrent",(self.driver.context_set)(self.context))?;
            self.driver.check("cuCtxSynchronize",(self.driver.context_sync)())
        }
    }
    fn hot_free_bytes(&self)->Result<u64> {
        unsafe {
            self.driver.check("cuCtxSetCurrent",(self.driver.context_set)(self.context))?;
            let (mut free,mut total)=(0usize,0usize);
            self.driver.check("cuMemGetInfo_v2",(self.driver.memory_info)(&mut free,&mut total))?;
            Ok(free as u64)
        }
    }
    fn slots(&self,names:&[String])->Result<Vec<(usize,usize)>>{
        names.iter().map(|name|{
            let idx=self.names.iter().position(|n|n==name).ok_or_else(||invalid("resident target handle missing"))?;
            let alias=format!("{CONTROL_PREFIX}{name}");
            let ctrl=self.names.iter().position(|n|n==&alias).ok_or_else(||invalid("resident control handle missing"))?;
            Ok((idx,ctrl))
        }).collect()
    }
    fn retire(&mut self,slots:&[(usize,usize)])->Result<()> {
        self.bind_drained()?;
        // SAFETY: All rank fences have been observed and new submissions are blocked.
        unsafe {
            for &(idx,ctrl) in slots {
                if self.executables[idx]!=self.executables[ctrl] {
                    self.driver.check("cuGraphExecDestroy",(self.driver.executable_destroy)(self.executables[idx]))?;
                    self.executables[idx]=self.executables[ctrl];
                }
            }
            while let Some(&module)=self.hot_modules.last(){
                self.driver.check("cuModuleUnload",(self.driver.module_unload)(module))?;
                self.hot_modules.pop();
            }
        }
        Ok(())
    }
    fn discard(&mut self, mut images:PendingImages)->Result<()> {
        let result=(|| {
            self.bind_drained()?;
            // SAFETY: These are private handles that were never submitted by a rank.
            unsafe {
                while let Some(&e)=images.executables.last() {
                    self.driver.check("cuGraphExecDestroy",(self.driver.executable_destroy)(e))?;
                    images.executables.pop();
                }
                while let Some(&t)=images.templates.last() {
                    self.driver.check("cuGraphDestroy",(self.driver.graph_destroy)(t))?;
                    images.templates.pop();
                }
                while let Some(&m)=images.modules.last() {
                    self.driver.check("cuModuleUnload",(self.driver.module_unload)(m))?;
                    images.modules.pop();
                }
            }
            Ok(())
        })();
        // Retain anything whose destruction could not be confirmed under
        // the normal native context owner's final drain.
        self.hot_retained.append(&mut images.executables);
        self.templates.append(&mut images.templates);
        self.hot_modules.append(&mut images.modules);
        result
    }
    fn stage(&mut self,plan:&RankPlan,gate:&crate::execution::graph_comparison::GraphComparison)->Result<PendingImages>{
        let mut functions=HashMap::new();
        for (new,old) in &plan.reused {
            let f=*self.functions.get(old).ok_or_else(||invalid("resident function disappeared"))?;
            functions.insert(*new,f);
        }
        let locked=functions.values().map(|&h|h as usize).collect();
        let frozen=std::mem::replace(&mut self.functions,functions);
        let (m,t,n,e)=(self.modules.len(),self.templates.len(),self.names.len(),self.executables.len());
        let progress=crate::startup::StartupProgress::silent();
        let phase=progress.phase("Prepare signed comparison graphs",None);
        let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: CPU verification proved the unchanged allocation/peer/control ABI.
            unsafe {
                self.prepare_kernel_subset(&plan.images,&plan.core,&phase,Some(&plan.load_functions))?;
                self.prepare_graphs_bounded(&plan.images,&plan.core,&phase,&locked,true)
            }
        })).unwrap_or_else(|_|Err(invalid("replacement graph construction panicked")));
        self.functions=frozen;
        let staged=PendingImages{modules:self.modules.split_off(m),templates:self.templates.split_off(t),
            names:self.names.split_off(n),executables:self.executables.split_off(e)};
        if let Err(error)=result {
            if let Err(cleanup)=self.discard(staged){gate.poison();return Err(invalid(format!("replacement cleanup failed: {cleanup}; construction: {error}")));}
            return Err(error);
        }
        Ok(staged)
    }
}

impl super::NativeModel {
    pub(crate) fn reload_comparison(&mut self,gate:&crate::execution::graph_comparison::GraphComparison,path:&Path,
                                    requested:&str) -> Result<serde_json::Value> {
        let origin=self.comparison_origin.as_ref().ok_or_else(||invalid("a signed paired native residency is required"))?;
        let available=self.ranks.iter().map(|r|r.functions.keys().copied().collect()).collect::<Vec<_>>();
        let plan=prepare(origin,path,requested,gate.names(),&available)?;
        let slots:Vec<_>=self.ranks.iter().map(|r|r.slots(gate.names())).collect::<Result<_>>()?;
        let _idle=gate.quiesce()?;
        for rank in &self.ranks { if let Err(err)=rank.bind_drained() {gate.poison();return Err(err);} }
        let free_before=self.ranks.iter().map(|r|r.hot_free_bytes()).collect::<Result<Vec<_>>>()?;
        gate.disable();
        for (rank,slots) in self.ranks.iter_mut().zip(&slots) {
            if let Err(err)=rank.retire(slots){gate.poison();return Err(err);}
        }
        let free_retired=self.ranks.iter().map(|r|r.hot_free_bytes()).collect::<Result<Vec<_>>>()?;
        let outcomes=std::thread::scope(|scope| {
            let jobs:Vec<_>=self.ranks.iter_mut().zip(&plan.ranks)
                .map(|(r,spec)|scope.spawn(move||r.stage(spec,gate))).collect();
            jobs.into_iter().map(|job|job.join().unwrap_or_else(|_| {
                gate.poison();Err(invalid("replacement worker did not return its resource ownership"))
            })).collect::<Vec<_>>()
        });
        let mut staged=Vec::new();let mut first_error=None;
        for (rank,result) in outcomes.into_iter().enumerate(){
            match result { Ok(bank)=>staged.push((rank,bank)), Err(e) if first_error.is_none()=>first_error=Some(e), Err(_)=>{} }
        }
        if let Some(err)=first_error {
            let mut cleanup_error=None;
            for (idx,bank) in staged {
                if let Err(cleanup)=self.ranks[idx].discard(bank){gate.poison();cleanup_error=Some(cleanup);}
            }
            return Err(invalid(match cleanup_error {
                Some(cleanup)=>format!("replacement could not drain: {cleanup}; construction: {err}"),
                None=>format!("candidate installation failed; control selected: {err}"),
            }));
        }
        let changed:Vec<_>=plan.ranks[0].core.graphs.iter().map(|g|g.name.clone()).collect();
        let mut bank=Vec::new();
        for ((rank,slots),(_,mut stage)) in self.ranks.iter_mut().zip(&slots).zip(staged) {
            debug_assert_eq!(stage.names,changed);
            debug_assert!(stage.templates.is_empty());
            for (name,exec) in stage.names.into_iter().zip(stage.executables) {
                // All names were checked before closing submission, and the
                // source's graph catalogue is immutable.
                let i=gate.names().iter().position(|n|n==&name).unwrap();
                rank.executables[slots[i].0]=exec;
            }
            rank.hot_modules.append(&mut stage.modules);
            bank.push(slots.iter().map(|&(idx,_)|rank.executables[idx] as usize).collect());
        }
        let id=plan.id;
        gate.install(id.clone(),&bank);
        // Observing free memory is diagnostic. A failed query after publication
        // must not report that an already committed install failed.
        let free_installed:Vec<Option<u64>>=self.ranks.iter().map(|r|r.hot_free_bytes().ok()).collect();
        Ok(serde_json::json!({"loaded_comparison":id,"changed_graphs":changed,
             "stored_executable_bytes":plan.executable_bytes,"role":"control",
             "frozen_allocations":true,"graphs":gate.names(),"launches":gate.counts(),
             "free_bytes_before":free_before,"free_bytes_after_retire":free_retired,
             "free_bytes_after_install":free_installed,
             "loaded_cubins_per_rank":plan.ranks.iter().map(|r|r.core.modules.len()).collect::<Vec<_>>(),
             "loaded_functions_per_rank":plan.ranks.iter().map(|r|r.load_functions.len()).collect::<Vec<_>>()}))
    }
}

/// Parse and authenticate the programs for a paired native graph package on a CPU-only host.
/// Unused producer functions may be pruned by serving options on an actual host.
pub fn inspect_plan(old_root:&Path,next_root:&Path,id:&str,publishers:&[Publisher])->Result<serde_json::Value>{
    let meta=Metadata::open(old_root)?;
    let verified=protection::verify(&meta.envelope,publishers,protection::Selection::Deployment{deployment:0,rank:None})?;
    if verified.ranks().len()!=16 {return Err(invalid("expected a local TP16 graph comparison"));}
    let names:BTreeSet<_>=verified.ranks()[0].graphs.iter().filter_map(|g|g.name.strip_prefix(CONTROL_PREFIX).map(str::to_owned)).collect();
    if names.is_empty(){return Err(invalid("resident input has no paired graph slots"));}
    let names=names.into_iter().collect::<Vec<_>>();
    let available=verified.ranks().iter().map(|r|r.graphs.iter().flat_map(|g|g.functions.iter().map(|i|*i as usize)).collect()).collect::<Vec<_>>();
    let origin=Origin{path:old_root.to_owned(),envelope_digest:verified.envelope_digest(),core_id:verified.core_id(),publisher:verified.publisher().clone()};
    let plan=prepare(&origin,next_root,id,&names,&available)?;
    Ok(serde_json::json!({"candidate":plan.id,"stored_executable_bytes":plan.executable_bytes,"slots":names,"executable_companions":plan.companions,
        "ranks":plan.ranks.iter().map(|r|serde_json::json!({"changed":r.core.graphs.iter().map(|g|&g.name).collect::<Vec<_>>(),
                "reused_functions":r.reused.len(),"additional_functions":r.load_functions.len(),"additional_modules":r.core.modules.len(),
                "programs":r.images.0.len()})).collect::<Vec<_>>() }))
}
