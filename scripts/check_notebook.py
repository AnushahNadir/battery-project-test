import argparse
import json
import logging
import os
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

def execute_notebook(notebook_path: Path) -> None:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    global_ns = {"__name__": "__main__"}

    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        try:
            code = compile(source, f"{notebook_path}:cell_{idx}", "exec")
            exec(code, global_ns, global_ns)
        except Exception as exc:
            raise RuntimeError(f"Notebook execution failed at cell {idx}: {exc}") from exc

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a notebook smoke check by executing code cells.")
    parser.add_argument("notebook", nargs="?", default="battery_analysis.ipynb")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.environ.setdefault("MPLBACKEND", "Agg")
    warnings.filterwarnings(
        "ignore",
        message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
        category=UserWarning,
    )
    notebook_path = Path(args.notebook)
    execute_notebook(notebook_path)
    logger.info(f"Notebook smoke check passed: {notebook_path}")

if __name__ == "__main__":
    main()
