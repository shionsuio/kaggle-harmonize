"""Entry point for the hybrid SDRF extraction pipeline.

This submission folder is intended to live under the competition repository's
`Submissions/` directory, but it is self-contained enough to rebuild the final
submission from bundled intermediate artifacts.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="Rebuild submission.csv from bundled llm_extractions and pride_anchors only.",
    )
    parser.add_argument(
        "--run-anchor-fetch",
        action="store_true",
        help="Refresh PRIDE anchors before rebuilding.",
    )
    parser.add_argument(
        "--run-llm-extract",
        action="store_true",
        help="Refresh LLM extraction outputs before rebuilding.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    pipeline_dir = root / "pipeline"
    os.chdir(root)

    if args.run_anchor_fetch:
        run(
            [
                sys.executable,
                "pipeline/fetch_pride_anchors.py",
                "Test PubText/Test PubText",
            ],
            root,
        )

    if args.run_llm_extract:
        run(
            [
                sys.executable,
                "pipeline/extract_with_llm_v2.py",
                "Test PubText/Test PubText/",
            ],
            root,
        )

    run([sys.executable, "pipeline/build_submission_v2.py"], root)


if __name__ == "__main__":
    main()
