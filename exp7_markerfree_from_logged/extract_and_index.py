import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_OUTPUTS = ROOT.parent / "experiments_markerfree" / "outputs"
RAW_OUTPUTS = ROOT / "outputs_raw"
INDEX_PATH = ROOT / "outputs_index.json"


def copy_outputs():
    if RAW_OUTPUTS.exists():
        shutil.rmtree(RAW_OUTPUTS)
    shutil.copytree(SRC_OUTPUTS, RAW_OUTPUTS)


def infer_exp_name(path: Path) -> str:
    name = path.name.lower()
    parts = []
    if "exp5" in name:
        parts.append("exp5")
    if "exp6" in name:
        parts.append("exp6")
    if "marker" in name:
        parts.append("marker")
    if "irrelevant" in name:
        parts.append("irrelevant")
    if not parts:
        parent = path.parent.name.lower()
        if parent.startswith("exp5"):
            return "exp5"
        if parent.startswith("exp6"):
            return "exp6"
        if "marker" in parent:
            return "marker"
    return "_".join(parts) if parts else "unknown"


def scan_csvs():
    entries = []
    for csv_path in RAW_OUTPUTS.rglob("*.csv"):
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                header = f.readline().strip()
        except UnicodeDecodeError:
            with csv_path.open("r", encoding="latin-1") as f:
                header = f.readline().strip()
        columns = [c.strip() for c in header.split(",")] if header else []
        entries.append(
            {
                "file_path": str(csv_path.relative_to(ROOT)),
                "available_columns": columns,
                "exp_name": infer_exp_name(csv_path),
            }
        )
    return entries


def main():
    copy_outputs()
    index = scan_csvs()
    with INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()

