//! Pair signed speculative graph alternatives over a shared native model allocation layout.
use flare_helios::core::{
    container::{self, Source, Writer},
    protection::{self, EnvelopeBuilder, Selection},
    reader::{self, Metadata},
    schema::wire,
};
use flare_helios::runtime_core::Publisher;
use openssl::pkey::PKey;
use prost::Message;
use ring::{
    digest,
    signature::{Ed25519KeyPair, KeyPair},
};
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::BufWriter;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};

type AnyResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
fn hex(data: &[u8]) -> String {
    data.iter().map(|x| format!("{x:02x}")).collect()
}
fn regular_path(s: &str) -> AnyResult<&Path> {
    let p = Path::new(s);
    if p.as_os_str().is_empty() || !p.components().all(|c| matches!(c, Component::Normal(_))) {
        return Err(format!("invalid package relative path {s:?}").into());
    }
    Ok(p)
}

fn pinned_same(a: &Path, b: &Path) -> AnyResult<bool> {
    let a = fs::symlink_metadata(a)?;
    let b = fs::symlink_metadata(b)?;
    Ok(
        a.is_file()
            && b.is_file()
            && a.dev() == b.dev()
            && a.ino() == b.ino()
            && a.len() == b.len(),
    )
}

const PREFIX: &str = "paired-control--";

fn compare_contracts(a: &protection::VerifiedCore, b: &protection::VerifiedCore) -> AnyResult<()> {
    if a.deployment_count() != 1
        || b.deployment_count() != 1
        || a.ranks().len() != 16
        || b.ranks().len() != 16
    {
        return Err("the native comparison requires one TP16 deployment".into());
    }
    let mut top = a.core().clone();
    top.core_id.clone_from(&b.core().core_id);
    top.deployments[0]
        .body_digest
        .clone_from(&b.core().deployments[0].body_digest);
    if top != *b.core() {
        return Err("model, hardware, frontend or sampling identities differ".into());
    }
    let mut deployment = a.deployment().ok_or("missing control deployment")?.clone();
    let candidate = b.deployment().ok_or("missing candidate deployment")?;
    if deployment.ranks.len() != candidate.ranks.len() {
        return Err("rank commitments differ".into());
    }
    for (old, new) in deployment.ranks.iter_mut().zip(&candidate.ranks) {
        old.body_digest.clone_from(&new.body_digest);
    }
    if deployment != *candidate {
        return Err("deployment contract differs".into());
    }
    Ok(())
}

fn pair_rank(
    control: &wire::RankBody,
    candidate: &wire::RankBody,
    next_id: &mut u64,
    mirror: &BTreeSet<String>,
    borrow: bool,
    only: Option<&BTreeSet<String>>,
) -> AnyResult<(wire::RankBody, Vec<String>)> {
    if control.graphs.len() != candidate.graphs.len()
        || !candidate.functions.starts_with(&control.functions)
        || !candidate.modules.starts_with(&control.modules)
        || !candidate.payloads.starts_with(&control.payloads)
        || candidate.resources.len() < control.resources.len()
    {
        return Err("candidate must retain the control's original executable definitions".into());
    }
    let mut paired = candidate.clone();
    let mut check = candidate.clone();
    check.modules.truncate(control.modules.len());
    check.functions.truncate(control.functions.len());
    check.payloads.truncate(control.payloads.len());
    check.resources.truncate(control.resources.len());
    let mut names = Vec::new();
    for (ordinal,(old, new)) in control.graphs.iter().zip(&candidate.graphs).enumerate() {
        if old.name != new.name || old.resource != new.resource || old.name.starts_with(PREFIX) {
            return Err("source graph names or frame indices differ".into());
        }
        let idx = old.resource as usize;
        let src = control
            .resources
            .get(idx)
            .ok_or("missing control graph resource")?;
        if src == &candidate.resources[idx] && old == new && !mirror.contains(&old.name) {
            continue;
        }
        if !old.name.starts_with("speculative") {
            return Err(format!("non-speculative graph differs: {}", old.name).into());
        }
        if src.resource_id != candidate.resources[idx].resource_id
            || src.kind != candidate.resources[idx].kind
        {
            return Err("candidate rewrote an underlying resource identity".into());
        }
        if only.is_some_and(|wanted|!wanted.contains(&old.name)) {
            paired.graphs[ordinal]=old.clone();
            paired.resources[idx]=src.clone();
            check.resources[idx]=src.clone();
            continue;
        }
        let mut graph = old.clone();
        graph.name = format!("{PREFIX}{}", old.name);
        if borrow {
            if src!=&candidate.resources[idx] || old!=new {return Err("borrowed reference bank requires identical source frames".into());}
        } else {
            let mut resource=src.clone();
            resource.resource_id=*next_id;
            *next_id=next_id.checked_add(1).ok_or("resource ID overflow")?;
            graph.resource=u32::try_from(paired.resources.len())?;
            paired.resources.push(resource);
        }
        paired.graphs.push(graph);
        check.resources[idx] = src.clone();
        names.push(old.name.clone());
    }
    check.graphs.clone_from(&control.graphs);
    if check != *control {
        return Err(
            "allocation, peer, weight, state, execution or existing resource differs".into(),
        );
    }
    if only.is_some_and(|wanted| wanted.len()!=names.len() || !names.iter().all(|name|wanted.contains(name))) {
        return Err("requested graph has no distinct signed alternate".into());
    }
    if names.is_empty() {
        return Err("no alternative speculative graph found".into());
    }
    Ok((paired, names))
}

fn run(control_path: &Path, source: &Path, destination: &Path, identity: &Path, mirror: &BTreeSet<String>, borrow: bool, only:Option<&BTreeSet<String>>) -> AnyResult<()> {
    if destination.exists() {
        return Err("package destination already exists".into());
    }
    let control = Metadata::open(control_path)?;
    let base = Metadata::open(source)?;
    let profiles: Vec<Publisher> =
        serde_json::from_slice(&fs::read(identity.join("development-publisher.json"))?)?;
    let selection = Selection::Deployment {
        deployment: 0,
        rank: None,
    };
    let original = protection::verify(&control.envelope, &profiles, selection)?;
    let verified = protection::verify(&base.envelope, &profiles, selection)?;
    let publisher = verified.publisher();
    if !publisher.development
        || publisher.namespace != "local-development"
        || publisher.public_key != original.publisher().public_key
    {
        return Err(
            "comparison requires source cores signed by the same development publisher".into(),
        );
    }
    compare_contracts(&original, &verified)?;
    let mut next_id = verified
        .ranks()
        .iter()
        .flat_map(|r| r.resources.iter().map(|s| s.resource_id))
        .max()
        .ok_or("rank has no executable resources")?
        .checked_add(1)
        .ok_or("resource ID overflow")?;
    let parent = destination.parent().ok_or("missing destination parent")?;
    fs::create_dir_all(parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".paired-core-")
        .tempdir_in(parent)?;
    let root = stage.path();
    let locations = base.locations.clone();
    let mut existing = BTreeSet::new();
    for loc in &locations.locations {
        if !loc.file.is_empty() {
            let rel = regular_path(&loc.file)?;
            if existing.insert(rel.to_owned()) {
                let (original, _) = container::open_relative(source, &loc.file)?;
                let dst = root.join(rel);
                fs::create_dir_all(dst.parent().unwrap())?;
                fs::hard_link(&original, &dst)?;
                if !pinned_same(&original, &dst)? {
                    return Err("original companion inode changed".into());
                }
            }
        }
    }
    let mut deploy = verified
        .deployment()
        .ok_or("missing deployment body")?
        .clone();
    let mut rank_bodies = Vec::new();
    let mut package_id = digest::Context::new(&digest::SHA256);
    package_id.update(b"Flare resident speculative graph comparison v1\0");
    package_id.update(&protection::sha256(&control.envelope));
    package_id.update(&protection::sha256(&base.envelope));
    let mut variants = None;
    for (i, (a, b)) in original.ranks().iter().zip(verified.ranks()).enumerate() {
        let (paired, names) = pair_rank(a, b, &mut next_id, mirror, borrow,only)?;
        if variants.as_ref().is_some_and(|v| *v != names) {
            return Err("alternate graph names differ between ranks".into());
        }
        variants = Some(names);
        let data = paired.encode_to_vec();
        deploy.ranks[i].body_digest = protection::sha256(&data).to_vec();
        package_id.update(&deploy.ranks[i].body_digest);
        rank_bodies.push(data);
    }
    let id: [u8; 32] = package_id.finish().as_ref().try_into()?;
    let dep = deploy.encode_to_vec();
    let mut package = verified.core().clone();
    package.core_id = id.to_vec();
    package.deployments[0].body_digest = protection::sha256(&dep).to_vec();
    let mut builder = EnvelopeBuilder::new(package.encode_to_vec(), publisher.public_key, id)?;
    builder.deployment(0, dep)?;
    for (i, body) in rank_bodies.into_iter().enumerate() {
        builder.rank(0, i as u32, body)?;
    }
    let mut encoded = fs::read(identity.join("development-signing-key.pem"))?;
    let native = PKey::private_key_from_pem(&encoded);
    encoded.fill(0);
    let mut seed = native?.raw_private_key()?;
    let key = Ed25519KeyPair::from_seed_unchecked(&seed);
    seed.fill(0);
    let key = key.map_err(|_| "invalid local development signing key")?;
    if key.public_key().as_ref() != publisher.public_key {
        return Err("development key does not match the verified source identity".into());
    }
    let envelope = builder.sign(&key)?;
    protection::verify(&envelope, &profiles, selection)?;
    let primary = root.join("model.safetensors");
    let mut w = Writer::new(
        BufWriter::new(
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&primary)?,
        ),
        container::PAYLOAD_FORMAT,
    )?;
    for name in base.primary().tensor_names() {
        if name == reader::MANIFEST || name == reader::LOCATIONS || name.starts_with("align.") {
            continue;
        }
        let slice = base.primary().tensor(name)?;
        w.add(name, slice.len(), Source::Slice(slice))?;
    }
    let encoded_locations = locations.encode_to_vec();
    w.add(
        reader::MANIFEST,
        envelope.len() as u64,
        Source::Bytes(envelope),
    )?;
    w.add(
        reader::LOCATIONS,
        encoded_locations.len() as u64,
        Source::Bytes(encoded_locations),
    )?;
    w.finish()?;
    fs::set_permissions(&primary, fs::Permissions::from_mode(0o444))?;
    let installed = Metadata::open(root)?;
    let verification = protection::verify(&installed.envelope, &profiles, selection)?;
    let required = reader::required_payloads(&verification)?;
    installed.payloads(verification.catalog(), required.iter().map(Vec::as_slice))?;
    flare_helios::core::export::publish_staged_directory(stage.path(), destination)?;
    println!(
        "Comparison {} uses {} unchanged payload files at {}. Paired operations: {}",
        hex(&id),
        existing.len(),
        destination.display(),
        variants.unwrap().join(", ")
    );
    Ok(())
}
fn main() -> AnyResult<()> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if let [mode, source, new, id, identity] = &args[..] {
        if mode == Path::new("--check-load") {
            let keys:Vec<Publisher>=serde_json::from_slice(&fs::read(identity.join("development-publisher.json"))?)?;
            let info=flare_helios::runtime_core::inspect_resident_comparison_plan(source,new,id.to_str().ok_or("invalid core ID")?,&keys)?;
            println!("{}",serde_json::to_string_pretty(&info)?);
            return Ok(());
        }
    }
    if let [mode,source,dest,identity,names @ ..]=&args[..] {
        if mode==Path::new("--reference-bank") || mode==Path::new("--shared-reference-bank") {
            let wanted:BTreeSet<_>=names.iter().map(|v|v.to_str().ok_or("invalid graph name").map(str::to_owned)).collect::<Result<_,_>>()?;
            if wanted.is_empty() || wanted.len()!=names.len() {return Err("specify distinct speculative graph names".into());}
            let meta=Metadata::open(source)?;
            let keys:Vec<Publisher>=serde_json::from_slice(&fs::read(identity.join("development-publisher.json"))?)?;
            let v=protection::verify(&meta.envelope,&keys,Selection::Deployment{deployment:0,rank:None})?;
            if !wanted.iter().all(|n|n.starts_with("speculative") && v.ranks().iter().all(|r|r.graphs.iter().any(|g|&g.name==n))) {
                return Err("requested reference-bank graph missing".into());
            }
            return run(source,source,dest,identity,&wanted,mode==Path::new("--shared-reference-bank"),None);
        }
    }
    if let [mode,base,source,dest,identity,names @ ..]=&args[..] {
        if mode==Path::new("--only") {
            let wanted:BTreeSet<_>=names.iter().map(|v|v.to_str().ok_or("invalid graph name").map(str::to_owned)).collect::<Result<_,_>>()?;
            if wanted.is_empty()||wanted.len()!=names.len(){return Err("specify distinct comparison graph names".into());}
            return run(base,source,dest,identity,&BTreeSet::new(),false,Some(&wanted));
        }
    }
    let [control,source,dest,identity]=&args[..] else {
        return Err("usage: kimi-paired-package CONTROL CANDIDATE OUTPUT IDENTITY | --reference-bank/--shared-reference-bank SOURCE OUTPUT IDENTITY GRAPH... | --only CONTROL CANDIDATE OUTPUT IDENTITY GRAPH... | --check-load RESIDENT NEXT NEXT_CORE_ID IDENTITY".into());
    };
    run(control,source,dest,identity,&BTreeSet::new(),false,None)
}
