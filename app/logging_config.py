"""One place to configure logging so every module can just do

    logger = logging.getLogger("compliance_copilot.<area>")

and have it show up in the terminal running the FastAPI backend
(uvicorn), with a consistent, readable format.

Note: this is the BACKEND terminal -- wherever you run
`uvicorn app.main:app` (or `python -m uvicorn ...`). If you're running
Streamlit in a separate terminal, these lines won't show there; they
show in the FastAPI process's own console, since that's the process
that actually talks to the LLM and Chroma.
"""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()

    if root.handlers:
        return  # already configured (e.g. re-imported under a reloader)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root.setLevel(level)
    root.addHandler(handler)

    # Uvicorn/Chroma/httpx are chatty at INFO; keep them at WARNING so our
    # own application logs (question/citations/timings) aren't buried.
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
