"""Send a transaction to a drained native graph comparison and verify its acknowledgement."""

import argparse
import json
from pathlib import Path
import sys
import time
import uuid


def submit(control, operation, timeout):
    command = uuid.uuid4().hex + " " + operation
    tmp = control / (command.split()[0] + ".tmp")
    tmp.write_text(command + "\n")
    tmp.replace(control / "request")
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            receipt = json.loads((control / "ack.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            receipt = {}
        if receipt.get("command") == command:
            receipt["elapsed_seconds"] = time.monotonic() - start
            return receipt
        time.sleep(0.03)
    raise TimeoutError(f"no acknowledgement for {command} within {timeout}s")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=480)
    p.add_argument("--out", type=Path)
    p.add_argument("--expect-rejection", action="store_true")
    sub = p.add_subparsers(dest="op", required=True)
    sub.add_parser("status")
    r = sub.add_parser("role")
    r.add_argument("role", choices=["candidate", "control"])
    q = sub.add_parser("load")
    q.add_argument("core_id")
    q.add_argument("path", type=Path)
    a = p.parse_args()
    if a.op == "role":
        op = f"role {a.role}"
    elif a.op == "load":
        op = f"load {a.core_id} {a.path.absolute()}"
    else:
        op = "status"
    receipt = submit(a.control, op, a.timeout)
    data = json.dumps(receipt, indent=2) + "\n"
    if a.out:
        a.out.write_text(data)
    view = dict(receipt)
    if view.get("comparison"):
        view["comparison"] = dict(view["comparison"])
        launch = view["comparison"].pop("launches", None)
        if launch is not None:
            view["comparison"]["rank_launch_totals"] = [
                [sum(q[0] for q in row), sum(q[1] for q in row)] for row in launch
            ]
    print(json.dumps(view))
    if bool(receipt.get("applied")) == a.expect_rejection:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
