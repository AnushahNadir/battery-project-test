import logging
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.log import setup_logging
from src.pipeline.mapper import standardize_columns

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_logging()
    path = "path/to/00001.csv"  # change to a real file
    df = pd.read_csv(path)

    logger.info("ORIGINAL COLS: %s", list(df.columns))

    df2 = standardize_columns(df, kind="ts", interactive=False)

    logger.info("RENAMED COLS : %s", list(df2.columns))
    logger.info(df2.head(3))
