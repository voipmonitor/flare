// SPDX-License-Identifier: Apache-2.0
//! Local commands for changing the set of eligible C1 verification graphs between requests.
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
        let budget = if let [_, budget] = words.as_slice() { budget.parse::<usize>().ok() } else { None };
        server.pause_admission();
        let outcome = if !server.wait_for_requests(Duration::ZERO) {
            server.resume_admission();
            return;
        } else if let Some(budget) = budget {
            server.set_benchmark_rows(budget).map_err(|e| e.to_string())
        } else {
            Err("expected a unique transaction ID and a nonnegative verification-row budget".to_owned())
        };
        server.resume_admission();
        let receipt = serde_json::json!({
            "command": command.trim(), "applied": outcome.is_ok(),
            "error": outcome.err(), "pid": std::process::id()
        });
        let temporary = self.dir.join("ack.tmp");
        let write = fs::write(&temporary, receipt.to_string()).and_then(|()| fs::rename(temporary, self.dir.join("ack.json")));
        if let Err(e) = write { eprintln!("Verification benchmark receipt failed: {e}"); return; }
        self.last = command;
    }
}
