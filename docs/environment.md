# Environment

## Python
`requirements.txt` covers the analysis, the detectors and the figures. Python
3.11 or newer.

## Not pip-installable

| Component | Used for | Note |
|---|---|---|
| ROS 2 | reading the recordings | the distribution the bags were recorded with |
| LIO-SAM | the maps of the paper | built from source |
| DLIO | the real-time estimator | built from source |
| KISS-ICP | registration-only baseline | ground removal on, deskew off |
| FAST-LIO | filter-based baseline | diverges on these recordings |
| RTAB-Map | pose-graph back end for KISS-ICP | optional |

Record the exact versions and commits used, with the launch and parameter
files, under `src/orchardmap/slam/` so the comparison in the paper can be
repeated.
