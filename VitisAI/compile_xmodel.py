#!/usr/bin/env python3
"""Compila o xmodel quantizado para o arch.json da DPU da ZCU104."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xmodel", required=True)
    parser.add_argument("--arch", required=True, help="arch.json que corresponde exatamente ao bitstream da placa.")
    parser.add_argument("--output-dir", default="build/vitis_ai/compiled_zcu104")
    parser.add_argument("--name", default="methane_segmentation")
    return parser.parse_args()


def main(args):
    compiler = shutil.which("vai_c_xir")
    if not compiler:
        raise SystemExit("vai_c_xir nao encontrado. Execute dentro do container Vitis AI.")
    xmodel, arch = Path(args.xmodel), Path(args.arch)
    for path in (xmodel, arch):
        if not path.is_file():
            raise SystemExit(f"Arquivo nao encontrado: {path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [compiler, "-x", str(xmodel), "-a", str(arch), "-o", str(output_dir), "-n", args.name]
    print("Executando:", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main(parse_args())

