//! Publish a separately signed local development package for fixed prepared PCIe transports.
#[path = "../../kimi31_dspark_release/envelope.rs"]
mod envelope;

use flare_helios::package::{
    container::{self, Slice, Source, Writer},
    protection::{self, Selection},
    reader::{self, Metadata},
    schema::wire,
};
use flare_helios::runtime_package::Publisher;
use prost::Message;
use ring::digest;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::BufWriter;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

type AnyResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[derive(Deserialize)]
struct Asset {
    id: String,
    filename: String,
    relative: String,
}
#[derive(Deserialize)]
struct Inputs {
    source_manifest: String,
    files: Vec<Asset>,
}

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

fn compare_rank(old: &wire::RankBody, updated: &wire::RankBody) -> AnyResult<()> {
    if updated.graphs.len() != old.graphs.len()
        || updated.modules.len() <= old.modules.len()
        || updated.functions.len() <= old.functions.len()
        || updated.resources.len() <= old.resources.len()
        || updated.payloads.len() != old.payloads.len() + 2
    {
        return Err("unexpected prepared transport resource counts".into());
    }
    let mut check = updated.clone();
    let mut graph_frames = BTreeSet::new();
    for (source, candidate) in old.graphs.iter().zip(&updated.graphs) {
        let mut same = candidate.clone();
        same.functions = source.functions.clone();
        if same != *source {
            return Err("transport replacement changed graph shape, size or edge count".into());
        }
        let index = source.resource as usize;
        if updated.resources[index] != old.resources[index] {
            if candidate.functions == source.functions || !graph_frames.insert(index) {
                return Err("changed graph frame lacks a unique replacement".into());
            }
            if updated.resources[index].kind != old.resources[index].kind
                || updated.resources[index].resource_id != old.resources[index].resource_id
            {
                return Err("captured graph resource identity changed".into());
            }
            check.resources[index] = old.resources[index].clone();
        } else if source != candidate {
            return Err("unmodified graph frame has different function IDs".into());
        }
    }
    if graph_frames.is_empty() {
        return Err("no changed graph resources".into());
    }
    check.graphs = old.graphs.clone();
    check.modules.truncate(old.modules.len());
    check.functions.truncate(old.functions.len());
    check.resources.truncate(old.resources.len());
    check.payloads.truncate(old.payloads.len());
    if check != *old {
        return Err("modified rank fields outside the executable closure".into());
    }
    Ok(())
}

fn run(
    source: &Path,
    inputs: &Path,
    destination: &Path,
    identity: &Path,
    signatures: Option<&Path>,
) -> AnyResult<()> {
    if destination.exists() {
        return Err("package destination already exists".into());
    }
    let base = Metadata::open(source)?;
    let profiles: Vec<Publisher> =
        serde_json::from_slice(&fs::read(identity.join("development-publisher.json"))?)?;
    let selection = Selection::Deployment {
        deployment: 0,
        rank: None,
    };
    let verified = protection::verify(&base.envelope, &profiles, selection)?;
    let publisher = verified.publisher();
    if !publisher.development
        || publisher.namespace != "local-development"
        || verified.deployment_count() != 1
        || verified.ranks().len() != 16
    {
        return Err("source must be the single, local-development TP16 deployment".into());
    }
    let spec: Inputs = serde_json::from_slice(&fs::read(inputs.join("resources.json"))?)?;
    if spec.source_manifest != hex(&protection::sha256(&base.envelope)) {
        return Err("source manifest identity differs".into());
    }
    let parent = destination.parent().ok_or("missing destination parent")?;
    fs::create_dir_all(parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".pcie-package-")
        .tempdir_in(parent)?;
    let root = stage.path();
    let mut locations = base.locations.clone();
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
    for a in &spec.files {
        let file = inputs.join(regular_path(&a.filename)?);
        let bytes = fs::read(&file)?;
        let id = protection::sha256(&bytes);
        if hex(&id) != a.id {
            return Err("new compressed executable payload digest differs".into());
        }
        let rel = regular_path(&a.relative)?;
        if !existing.insert(rel.to_owned())
            || locations.locations.iter().any(|l| l.payload_id == id)
        {
            return Err("duplicate executable placement".into());
        }
        let dst = root.join(rel);
        fs::create_dir_all(dst.parent().unwrap())?;
        let raw = Slice::new(Arc::new(File::open(&file)?), 0, bytes.len() as u64)?;
        let mut companion = OpenOptions::new().write(true).create_new(true).open(&dst)?;
        container::write_companion(&mut companion, &raw)?;
        companion.sync_all()?;
        fs::set_permissions(&dst, fs::Permissions::from_mode(0o444))?;
        locations.locations.push(wire::PayloadLocation {
            payload_id: id.to_vec(),
            file: a.relative.clone(),
            tensor: "flare.payload".into(),
        });
    }
    let mut deploy = verified
        .deployment()
        .ok_or("missing deployment body")?
        .clone();
    let mut rank_bodies = Vec::new();
    let mut package_id = digest::Context::new(&digest::SHA256);
    package_id.update(b"Flare local Kimi prepared PCIe transport package v1\0");
    package_id.update(&protection::sha256(&base.envelope));
    for (i, old) in verified.ranks().iter().enumerate() {
        let data = fs::read(inputs.join(format!("rank-{i}.pb")))?;
        let updated = wire::RankBody::decode(&*data)?;
        compare_rank(old, &updated)?;
        deploy.ranks[i].body_digest = protection::sha256(&data).to_vec();
        package_id.update(&deploy.ranks[i].body_digest);
        rank_bodies.push(data);
    }
    let id: [u8; 32] = package_id.finish().as_ref().try_into()?;
    let dep = deploy.encode_to_vec();
    let mut package = verified.package().clone();
    package.package_id = id.to_vec();
    package.deployments[0].body_digest = protection::sha256(&dep).to_vec();
    let envelope = envelope::assemble(
        &publisher,
        id,
        package.encode_to_vec(),
        dep,
        rank_bodies,
        identity,
        signatures,
    )?;
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
    let pinned = installed.payloads(verification.catalog(), required.iter().map(Vec::as_slice))?;
    for a in &spec.files {
        let id = protection::sha256(&fs::read(inputs.join(&a.filename))?);
        let slice = pinned.get(&id)?;
        let mut bytes = vec![0; usize::try_from(slice.len())?];
        slice.read_at(0, &mut bytes)?;
        if protection::sha256(&bytes) != id {
            return Err("published executable payload verification failed".into());
        }
    }
    flare_helios::package::export::publish_staged_directory(stage.path(), destination)?;
    println!(
        "Published development package {} with {} original hard links and {} new companions at {}",
        hex(&id),
        existing.len() - spec.files.len(),
        spec.files.len(),
        destination.display()
    );
    Ok(())
}
fn main() -> AnyResult<()> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    match &args[..] {
        [source, inputs, dest, identity] => run(source, inputs, dest, identity, None),
        [source, inputs, dest, profiles, signatures] => {
            run(source, inputs, dest, profiles, Some(signatures))
        }
        _ => Err(
            "usage: publisher SOURCE INPUTS DEST IDENTITY_DIRECTORY [DETACHED_SIGNATURES]".into(),
        ),
    }
}
