import runpy
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve().parents[1] / "LLaMA-Factory" / "scripts" / "medical_benchmark.py"
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()

