# figures

Scripts that build the paper's figures and the analyses behind its tables.

They were written against a working layout, not as a package, so they need two
directories on the path:

```bash
export ORCHARD_DATA=/path/to/orchard-data
export PYTHONPATH=$PWD/src/orchardmap:$PWD/figures/lib
python3 figures/analysis/s4_map_summary.py
```

| Folder | What is in it |
|---|---|
| `analysis/` | `s0`–`s5`, the numbered steps that produce the tables, and the `fig_*` scripts of the thesis chapter |
| `paper/` | the figures of the submitted paper |
| `lib/` | plotting style and shared helpers used by both |

Three modules the analysis imports — `common.py`, `entities.py` and
`priority_lib.py` — are pipeline code and live in `src/orchardmap/`, which is
why it is on the path above.
