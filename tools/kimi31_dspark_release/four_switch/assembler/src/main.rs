//! Reconstruct prepared TP16 graph packages under detached public signatures.
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
    old_payload: Option<String>,
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
        || updated.resources.len() < old.resources.len()
        || !updated.modules.starts_with(&old.modules)
        || !updated.functions.starts_with(&old.functions)
        || !updated.payloads.starts_with(&old.payloads)
    {
        return Err("incompatible source executable catalog".into());
    }
    let mut restored = updated.clone();
    for (i, original) in old.graphs.iter().enumerate() {
        let graph = &updated.graphs[i];
        if graph.resource != original.resource {
            return Err("changed root graph resource".into());
        }
        let index = graph.resource as usize;
        if graph == original && restored.resources[index] == old.resources[index] {
            continue;
        }
        let width = graph
            .name
            .strip_prefix("speculative1x8-tokens")
            .and_then(|value| value.parse::<usize>().ok());
        if !matches!(width, Some(2..=8)) {
            return Err(format!("unexpected changes to {}", graph.name).into());
        }
        let mut metadata = graph.clone();
        metadata.functions.clone_from(&original.functions);
        metadata.node_count = original.node_count;
        metadata.edge_count = original.edge_count;
        if &metadata != original {
            return Err(format!("changed physical metadata for {}", original.name).into());
        }
        restored.graphs[i] = original.clone();
        restored.resources[index] = old.resources[index].clone();
    }
    if restored.resources[old.resources.len()..]
        .iter()
        .any(|r| r.kind != wire::ExecutableKind::Cubin as i32)
    {
        return Err("expected only executable CUDA images in the resource suffix".into());
    }
    restored.resources.truncate(old.resources.len());
    restored.functions.truncate(old.functions.len());
    restored.modules.truncate(old.modules.len());
    restored.payloads.truncate(old.payloads.len());
    if &restored != old {
        return Err("weights, allocations or unrelated rank data changed".into());
    }
    Ok(())
}

fn compare_hidden_row_cubins(
    old: &wire::RankBody,
    updated: &wire::RankBody,
    previous: &str,
    replacement: &str,
) -> AnyResult<()> {
    if old.resources.len() != updated.resources.len()
        || old.payloads.len() != updated.payloads.len()
    {
        return Err("substitution changed the executable catalog size".into());
    }
    let mut restored = updated.clone();
    let mut changed = 0;
    for (index, (before, after)) in old.resources.iter().zip(&updated.resources).enumerate() {
        if !before
            .stored
            .as_ref()
            .is_some_and(|s| hex(&s.payload_id) == previous)
        {
            continue;
        }
        if before.kind != wire::ExecutableKind::Cubin as i32
            || before.kind != after.kind
            || before.resource_id != after.resource_id
            || before.chunk_nbytes != after.chunk_nbytes
            || !after
                .stored
                .as_ref()
                .is_some_and(|s| hex(&s.payload_id) == replacement)
        {
            return Err(
                "substitution changed a cubin's identity or left the replacement payload".into(),
            );
        }
        restored.resources[index] = before.clone();
        changed += 1;
    }
    let payloads: Vec<_> = old
        .payloads
        .iter()
        .enumerate()
        .filter(|(_, p)| hex(&p.id) == previous)
        .collect();
    if changed != 25 || payloads.len() != 1 {
        return Err("expected exactly 25 cubins in one original payload".into());
    }
    let (index, prior) = payloads[0];
    if hex(&updated.payloads[index].id) != replacement {
        return Err("rank's replacement payload ID differs".into());
    }
    restored.payloads[index] = prior.clone();
    if restored != *old {
        return Err(
            "substitution modified another rank field, graph, function or allocation".into(),
        );
    }
    Ok(())
}

fn run(
    source: &Path,
    inputs: &Path,
    destination: &Path,
    identity: &Path,
    step: &str,
) -> AnyResult<()> {
    let (domain, detached): (&[u8], &str) = match step {
        "attention" => (
            b"Flare local Kimi prepared PCIe transport package v1\0",
            "attention.pb",
        ),
        "latent-norm" => (
            b"Flare local Kimi prepared PCIe transport package v1\0",
            "latent_norm.pb",
        ),
        "hidden-residual" => (
            b"Flare local Kimi hidden-row IEEE positive-zero reduction v1\0",
            "hidden_residual.pb",
        ),
        "experts" => (
            b"Kimi DSpark TP16 collective graph preparation v1\0",
            "experts.pb",
        ),
        "proposals" => (
            b"Kimi DSpark packed causal proposal prefix v1\0",
            "bounded.pb",
        ),
        "transport" => (
            b"Kimi DSpark TP16 collective graph preparation v1\0",
            "transport.pb",
        ),
        _ => return Err(format!("unknown graph reconstruction stage {step:?}").into()),
    };
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
    if (step == "hidden-residual") != spec.old_payload.is_some() {
        return Err("replacing a companion requires the signed hidden-row substitution".into());
    }
    if let Some(previous) = &spec.old_payload {
        let [asset] = &spec.files[..] else {
            return Err("expected a single hidden-row cubin companion".into());
        };
        let old: Vec<_> = locations
            .locations
            .iter()
            .filter(|l| hex(&l.payload_id) == *previous)
            .collect();
        if old.len() != 1
            || old[0].file != asset.relative
            || asset.relative != "deployments/gb202-tp16/row-kernels.safetensors"
            || locations
                .locations
                .iter()
                .any(|l| l.file == asset.relative && hex(&l.payload_id) != *previous)
        {
            return Err("the hidden-row cubins do not identify an exclusive companion".into());
        }
        locations
            .locations
            .retain(|l| hex(&l.payload_id) != *previous);
    }
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
    package_id.update(domain);
    package_id.update(&protection::sha256(&base.envelope));
    for (i, old) in verified.ranks().iter().enumerate() {
        let data = fs::read(inputs.join(format!("rank-{i}.pb")))?;
        let updated = wire::RankBody::decode(&*data)?;
        if let Some(previous) = &spec.old_payload {
            compare_hidden_row_cubins(old, &updated, previous, &spec.files[0].id)?;
        } else {
            compare_rank(old, &updated)?;
        }
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
        id,
        package.encode_to_vec(),
        dep,
        rank_bodies,
        &identity.join(detached),
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
        "Reconstructed signed package {} with {} shared companions and {} generated companions at {}",
        hex(&id),
        existing.len() - spec.files.len(),
        spec.files.len(),
        destination.display()
    );
    Ok(())
}
fn main() -> AnyResult<()> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let [source, inputs, dest, identity, step] = &args[..] else {
        return Err(
            "usage: kimi-pcie-package SOURCE INPUTS DEST PUBLIC_SIGNATURE_DIRECTORY STEP".into(),
        );
    };
    run(
        source,
        inputs,
        dest,
        identity,
        step.to_str().ok_or("invalid reconstruction step")?,
    )
}
