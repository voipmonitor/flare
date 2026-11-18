// SPDX-License-Identifier: Apache-2.0
//! Assemble reconstructed bodies under the release publisher's detached Ed25519 signatures.
use flare_helios::package::{protection::EnvelopeBuilder, schema::wire};
use flare_helios::runtime_package::Publisher;
use openssl::pkey::PKey;
use prost::Message;
use ring::signature::{Ed25519KeyPair, KeyPair};
use std::{fs, path::Path};

pub fn assemble(
    publisher: &Publisher,
    id: [u8; 32],
    package: Vec<u8>,
    deployment: Vec<u8>,
    ranks: Vec<Vec<u8>>,
    identity: &Path,
    signatures: Option<&Path>,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(file) = signatures {
        let mut envelope = wire::ProtectedPackage::decode(fs::read(file)?.as_slice())?;
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
                return Err(
                    "rank signature indices do not match the reconstructed deployment".into(),
                );
            }
            entry.body = bytes;
        }
        // The caller verifies every rebuilt body with the trusted publisher
        // profile before it writes a primary or publishes the directory.
        return Ok(envelope.encode_to_vec());
    }
    let mut builder = EnvelopeBuilder::new(package, publisher.public_key, id)?;
    builder.deployment(0, deployment)?;
    for (i, body) in ranks.into_iter().enumerate() {
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
    Ok(builder.sign(&key)?)
}
