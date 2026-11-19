"""Publish an authenticated executable-only native graph view from a signed comparison core."""

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile


def core_plan(packager, reference, candidate, core_id, profiles):
    proc = subprocess.run(
        [str(packager), "--check-load", str(reference), str(candidate), core_id, str(profiles)],
        text=True,
        capture_output=True,
    )
    if proc.returncode:
        raise ValueError(proc.stderr.strip() or "the native executable plan was rejected")
    plan = json.loads(proc.stdout)
    if plan.get("candidate") != core_id:
        raise ValueError("the verifier returned a different signed core ID")
    return plan


def input_paths(plan):
    yield Path("model.safetensors")
    seen = set()
    for name in plan["executable_companions"]:
        rel = PurePosixPath(name)
        if (
            rel.is_absolute()
            or name != rel.as_posix()
            or ".." in rel.parts
            or len(rel.parts) < 2
            or rel.parts[0] != "deployments"
            or name in seen
        ):
            raise ValueError(f"invalid or repeated executable companion path {name!r}")
        seen.add(name)
        yield Path(rel)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def materialize(source, destination, force_copy):
    metadata = source.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"the companion must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not force_copy and not (metadata.st_mode & 0o222):
        try:
            os.link(source, destination, follow_symlinks=False)
            return "linked"
        except OSError as error:
            if error.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP}:
                raise
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=4 << 20)
    destination.chmod(0o444)
    return "copied"


def run(args):
    reference = args.reference.resolve(strict=True)
    candidate = args.candidate.resolve(strict=True)
    packager = args.packager.resolve(strict=True)
    profiles = args.publisher_dir.resolve(strict=True)
    out = args.output.absolute()
    resolved_output = out.resolve()
    if resolved_output.is_relative_to(reference) or resolved_output.is_relative_to(candidate):
        raise ValueError("the output must be separate from both signed input directories")
    if len(args.core_id) != 64 or any(c not in "0123456789abcdef" for c in args.core_id):
        raise ValueError("the expected signed core ID must be 64 lowercase hex characters")
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    primary_digest = digest(candidate / "model.safetensors")
    plan = core_plan(packager, reference, candidate, args.core_id, profiles)
    counts = {"linked": 0, "copied": 0}
    paths = list(input_paths(plan))
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".prepare-executables-") as tmp:
        stage = Path(tmp)
        for rel in paths:
            original = candidate / rel
            resolved = original.resolve(strict=True)
            if not resolved.is_relative_to(candidate):
                raise ValueError(f"signed source placement leaves the input directory: {rel}")
            action = materialize(resolved, stage / rel, args.copy)
            counts[action] += 1
        if digest(stage / "model.safetensors") != primary_digest:
            raise ValueError("the signed primary changed during staging")
        if plan != core_plan(packager, reference, stage, args.core_id, profiles):
            raise ValueError("the staged signed executable plan differs")
        # An exclusive destination gives competing writers a deterministic outcome.
        # Readers can open a core only once the verified primary is installed.
        out.mkdir(mode=0o700)
        try:
            for entry in stage.iterdir():
                if entry.name != "model.safetensors":
                    entry.rename(out / entry.name)
            (stage / "model.safetensors").rename(out / "model.safetensors")
        except BaseException:
            # Only the installer owns this private output directory.
            shutil.rmtree(out)
            raise
    return dict(
        core_id=args.core_id,
        primary_sha256=primary_digest,
        directory=str(out),
        companion_files=len(paths) - 1,
        stored_program_bytes=plan["stored_executable_bytes"],
        changed_graphs=plan["ranks"][0]["changed"],
        files=counts,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packager", required=True, type=Path)
    p.add_argument(
        "--reference", required=True, type=Path, help="signed resident comparison reference directory"
    )
    p.add_argument("--candidate", required=True, type=Path, help="signed paired candidate directory")
    p.add_argument("--core-id", required=True, help="expected signed paired core ID")
    p.add_argument(
        "--publisher-dir", required=True, type=Path, help="directory containing development-publisher.json"
    )
    p.add_argument("--output", required=True, type=Path, help="a new destination directory")
    p.add_argument(
        "--copy", action="store_true", help="copy all files even when immutable hardlinks are possible"
    )
    args = p.parse_args()
    try:
        print(json.dumps(run(args), indent=2))
    except (OSError, ValueError, KeyError) as error:
        p.exit(1, f"executable-view preparation failed: {error}\n")


if __name__ == "__main__":
    main()
