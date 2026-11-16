#!/usr/bin/env python3
"""Command-line entry point for LASSI. Run `python run_lassi.py --help`."""
import sys

from lassi.cli import main

if __name__ == "__main__":
    sys.exit(main())
