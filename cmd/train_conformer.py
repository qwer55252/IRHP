#!/usr/bin/env python3
"""Launch a historical Conformer recipe in an environment with the required NeMo patch."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    module = cfg.pop("entrypoint")
    if module not in ("irhp.conformer.train", "irhp.conformer.train_tedlium"):
        parser.error("Unsupported Conformer entry point")
    cfg.pop("status", None)
    command = [sys.executable, "-m", module, "--output_dir", args.output_dir]
    for key, value in cfg.items():
        command.extend(["--" + key, str(value)])
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)


if __name__ == "__main__":
    main()
