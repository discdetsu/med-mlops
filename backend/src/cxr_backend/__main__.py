"""`python -m cxr_backend <command>`: Django's manage.py, without a file outside the package."""

import os
import sys

from django.core.management import execute_from_command_line


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cxr_backend.settings")
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
