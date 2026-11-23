"""Copy to config/paths.py and edit. config/paths.py is not committed.

DATA_ROOT is the folder the Zenodo records are unpacked into; see docs/data.md
for the expected layout.
"""
import os
DATA_ROOT = os.environ.get("ORCHARD_DATA", "/path/to/orchard-data")
OUT_ROOT  = os.environ.get("ORCHARD_OUT", "./out")
