"""Give the immutable experiment a short logical path on the same data disk."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import stat


SOCKET_SUFFIX = "/pymp-12345678/listener-12345678"


def socket_path_bytes(tmpdir: Path) -> int:
    return len(os.fsencode(str(tmpdir) + SOCKET_SUFFIX))


def prepare(root: Path, alias_parent: Path = Path("/tmp")) -> Path:
    root = root.resolve(strict=True)
    if not (root / ".git").is_dir():
        raise RuntimeError("The verified frozen Git checkout is required.")
    key = hashlib.sha256(os.fsencode(str(root))).hexdigest()[:12]
    # Keep the link private. The link lives in /tmp; its target, including all
    # temporary files, environments and checkpoints, stays on the data disk.
    private = alias_parent / f"cyp-{os.getuid()}-{key}"
    try:
        private.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = private.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise RuntimeError(f"Unexpected short-runtime directory; preserved: {private}")
    alias = private / "r"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except FileExistsError:
        pass
    if not alias.is_symlink() or alias.resolve(strict=True) != root:
        raise RuntimeError(f"Short-runtime link points elsewhere; preserved: {alias}")
    temporary = alias / ".runtime/tmp"
    # Deliberately do not resolve this pathname: AF_UNIX uses the pathname
    # supplied to bind(), not the length of the symlink's canonical target.
    if socket_path_bytes(temporary) >= 108:
        raise RuntimeError("Short-runtime socket pathname still exceeds Linux's limit.")
    real_temporary = root / ".runtime/tmp"
    if real_temporary.is_symlink():
        raise RuntimeError("Unexpected temporary-directory symlink; preserved.")
    real_temporary.mkdir(parents=True, exist_ok=True)
    return alias


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(prepare(args.root))
    except Exception as error:
        parser.exit(1, f"SHORT RUNTIME FAILED: {error}\n")
