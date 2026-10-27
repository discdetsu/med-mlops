"""Prefect flows. `make train` runs them in-process under `core` and via the server under `full`."""

import logging
import os

# Forward this package's stdlib loggers to the Prefect API, so their lines show up in the flow
# run logs in the UI. Must be set before Prefect configures logging on import. Prefect leaves
# their level unset (so WARNING); at INFO they also reach its console handler.
os.environ.setdefault("PREFECT_LOGGING_EXTRA_LOGGERS", "cxr_flows")
logging.getLogger("cxr_flows").setLevel(logging.INFO)
# kagglehub draws a progress bar per downloaded file (thousands of them). tqdm reads this
# when it is first imported, so it has to be set here, before anything imports tqdm.
os.environ.setdefault("TQDM_DISABLE", "1")
