"""Calibra e valida ambos os modelos; persiste estado e logs."""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from common import OFFICIAL_CALIBRATION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        parser.error("Use um diretorio vazio para preservar resultados anteriores.")
    root.mkdir(parents=True, exist_ok=True)
    state = dict(state="running", models={})

    def save():
        temporary = root / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(root / "status.json")

    save()
    try:
        models = ["attentiongates_dpu_easy_remaining", "attentiongates_dpu_only_remaining"]
        for stage in ["calib", "validation"]:
            jobs = []
            for model in models:
                if stage == "calib":
                    command = [sys.executable, "-u", "quantize_model.py", "--model", model,
                               "--quant-mode", "calib", "--csv", "/dataset_STARCOP/train.csv",
                               "--data-root", "/dataset_STARCOP", "--range-policy", "layer_mse",
                               "--max-clipping-percent", "0.1", "--num-workers", "2",
                               "--target", "DPUCZDX8G_ISA1_B4096", "--output-dir", str(root / "quantize")]
                    for key, value in OFFICIAL_CALIBRATION[model].items():
                        command.extend(["--" + key.replace("_", "-"), str(value)])
                else:
                    command = [sys.executable, "-u", "evaluate_quantized.py", "--model", model,
                               "--dataset", "test", "--quant-dir", str(root / "quantize"),
                               "--output-dir", str(root / "evaluation" / model)]
                log = open(root / (model + "_" + stage + ".log"), "w")
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                jobs.append((model, process, log))
                state["models"].setdefault(model, {})[stage] = dict(state="running", pid=process.pid, command=command)
                save()
            failed = []
            for model, process, log in jobs:
                code = process.wait()
                log.close()
                state["models"][model][stage].update(state="complete" if code == 0 else "failed", exit_code=code)
                save()
                if code:
                    failed.append(model)
            if failed:
                raise RuntimeError(f"Falha em {stage}: {failed}; consulte os logs.")
        import pandas as pd
        summary = pd.concat([pd.read_csv(root / "evaluation" / model / "summary.csv") for model in models])
        summary.to_csv(root / "summary.csv", index=False)
        state["state"] = "complete"
        save()
    except Exception as error:
        state.update(state="failed", error=str(error))
        (root / "error.log").write_text(traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
