// SPDX-License-Identifier: Apache-2.0
//! Select a verification-row budget or a signed speculative graph role between native benchmark requests.
use std::{fs, io, path::PathBuf, time::Duration};

pub(super) struct Control {
    dir: PathBuf,
    last: String,
}

impl Control {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir, last: String::new() }
    }

    pub fn poll(&mut self, server: &flare_helios::server::Server) {
        let command = match fs::read_to_string(self.dir.join("request")) {
            Ok(s) => s,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return,
            Err(e) => { eprintln!("Verification benchmark command read failed: {e}"); return; }
        };
        if self.last == command { return; }
        let words: Vec<_> = command.split_whitespace().collect();
        server.pause_admission();
        let outcome = if !server.wait_for_requests(Duration::ZERO) {
            server.resume_admission();
            return;
        } else {
            match words.as_slice() {
                [_, "role", role @ ("control" | "candidate")] => server.set_benchmark_graph_role(*role == "candidate")
                    .map_err(|e| e.to_string()),
                [_, rows] => rows.parse::<usize>().map_err(|_| "invalid verification row budget".to_owned())
                    .and_then(|n| server.set_benchmark_rows(n).map(|()| serde_json::Value::Null).map_err(|e| e.to_string())),
                _ => Err("expected TRANSACTION ROWS or TRANSACTION role control|candidate".to_owned()),
            }
        };
        server.resume_admission();
        let receipt = serde_json::json!({
            "command": command.trim(), "applied": outcome.is_ok(), "comparison": outcome.as_ref().ok(),
            "error": outcome.as_ref().err(), "pid": std::process::id()
        });
        let temporary = self.dir.join("ack.tmp");
        let write = fs::write(&temporary, receipt.to_string()).and_then(|()| fs::rename(temporary, self.dir.join("ack.json")));
        if let Err(e) = write { eprintln!("Verification benchmark receipt failed: {e}"); return; }
        self.last = command;
    }
}
