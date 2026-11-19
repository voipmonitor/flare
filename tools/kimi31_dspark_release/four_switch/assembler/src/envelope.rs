// SPDX-License-Identifier: Apache-2.0
//! Attach the public signatures for the reconstructed package, deployment and rank bodies.
use flare_helios::package::schema::wire;
use prost::Message;
use std::{fs, path::Path};

pub fn assemble(
    id: [u8; 32],
    package: Vec<u8>,
    deployment: Vec<u8>,
    ranks: Vec<Vec<u8>>,
    signatures: &Path,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let mut envelope = wire::ProtectedPackage::decode(fs::read(signatures)?.as_slice())?;
    if envelope.package_id != id
        || !envelope.package_body.is_empty()
        || envelope.deployments.len() != 1
        || envelope.ranks.len() != ranks.len()
    {
        return Err("signature set does not identify the reconstructed package".into());
    }
    let entry = &mut envelope.deployments[0];
    if entry.index != 0 || !entry.body.is_empty() {
        return Err("expected one detached deployment signature".into());
    }
    envelope.package_body = package;
    entry.body = deployment;
    for (i, (entry, bytes)) in envelope.ranks.iter_mut().zip(ranks).enumerate() {
        if entry.deployment != 0 || entry.rank as usize != i || !entry.body.is_empty() {
            return Err("rank signature indices do not match the reconstructed deployment".into());
        }
        entry.body = bytes;
    }
    // The caller verifies every rebuilt body with the trusted publisher
    // profile before it writes a primary or publishes the directory.
    Ok(envelope.encode_to_vec())
}
