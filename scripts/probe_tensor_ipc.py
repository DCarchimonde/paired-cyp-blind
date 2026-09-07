"""Fail before training if the actual temporary path cannot transfer tensors."""
from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import tempfile

from short_runtime import socket_path_bytes


def probe() -> None:
    expected = os.environ.get("TMPDIR")
    observed = tempfile.gettempdir()
    if expected is None or observed != expected:
        raise RuntimeError(f"TMPDIR was not used literally: expected={expected!r}, actual={observed!r}")
    length = socket_path_bytes(Path(observed))
    if length >= 108:
        raise RuntimeError(f"AF_UNIX path too long: expected {length} bytes, maximum 107.")
    with Listener(family="AF_UNIX", authkey=b"paired-cyp-ipc-preflight") as listener:
        if len(os.fsencode(listener.address)) >= 108:
            raise RuntimeError("The real multiprocessing listener pathname is too long.")
    print(f"IPC SOCKET PASS: TMPDIR={observed}; estimated listener path={length} bytes.", flush=True)
    evidence = {"checked_utc": datetime.now(timezone.utc).isoformat(),
                "tmpdir": observed, "canonical_tmpdir": str(Path(observed).resolve()),
                "estimated_socket_path_bytes": length, "socket_pass": True,
                "tensor_transfer_pass": None, "training_completed": False,
                "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save() -> None:
        runtime = Path(observed).resolve().parent
        destination = runtime / "ipc-preflight.json"
        with tempfile.NamedTemporaryFile(mode="w", dir=runtime, prefix="ipc-preflight-",
                                         suffix=".tmp", delete=False) as stream:
            json.dump(evidence, stream, indent=2)
            stream.write("\n")
            temporary = Path(stream.name)
        os.replace(temporary, destination)

    if importlib.util.find_spec("torch") is None:
        print("Tensor transfer check deferred: torch is not installed yet. Socket check passed.", flush=True)
        save()
        return
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # This CPU-only, ordered transfer exercises the same file-descriptor sharing
    # that failed in the user's four DataLoader workers. It is not a model run.
    expected_tensor = torch.arange(64, dtype=torch.int64)
    loader = DataLoader(TensorDataset(expected_tensor), batch_size=8,
                        num_workers=4, shuffle=False, timeout=15)
    actual = torch.cat([batch[0] for batch in loader])
    if not torch.equal(actual, expected_tensor):
        raise RuntimeError("Four-worker tensor transfer changed the data or its order.")
    evidence.update(tensor_transfer_pass=True, workers=4, values_transferred=64,
                    torch_version=torch.__version__)
    save()
    print("IPC PREFLIGHT PASS: 4 workers transferred all 64 values in order.", flush=True)


if __name__ == "__main__":
    probe()
