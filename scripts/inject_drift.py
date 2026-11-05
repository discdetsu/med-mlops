"""Send a slice of the validation split that differs from the drift reference: AP views only by
default (37% of the reference). The next monitor pass should raise the drift scores in Grafana
and fire CxrDriftDetected.

    uv run python scripts/inject_drift.py [--n 150] [--slice patient_age>=70 ...]
    make inject-drift

Takes the same options as replay_traffic.py; any `--slice` replaces the AP default.
"""

import sys

from replay_traffic import main

if __name__ == "__main__":
    custom = any(a == "--slice" or a.startswith("--slice=") for a in sys.argv[1:])
    main(default_slice=() if custom else ("view_position=AP",))
