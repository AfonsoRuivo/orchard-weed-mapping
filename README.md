# Orchard weed and tree trunk mapping

Code for the paper *Automated Weed and Tree Trunk Mapping in Orchards with
Limited GNSS Availability* (Ruivo and Vale, submitted to *Robotics and
Autonomous Systems*) and for the MSc thesis it is based on, *Mobile Robot
Navigation for Vegetation Mapping in Agricultural Applications* (Instituto
Superior Técnico, 2026).

One pass of a ground platform carrying a LiDAR, an IMU, a colour camera and a
GNSS receiver is turned into a georeferenced map of the weeds and trees of an
orchard, in which every weed is related to the trees and ranked for
intervention. GNSS is used only to place the finished map in East–North–Up
coordinates, never to build it.

## Data

The code reads the two public dataset records, not the raw recordings:

| Record | DOI | Contents |
|---|---|---|
| Main | [10.5281/zenodo.23805668](https://doi.org/10.5281/zenodo.23805668) | session CB-11, calibration, trunk survey, labelled images |
| Second orchard | [10.5281/zenodo.23808549](https://doi.org/10.5281/zenodo.23808549) | session CA-10 |

Both are CC BY 4.0. The GNSS fixes in the released recordings were moved to a
synthetic origin, which keeps their timing and relative geometry but not the
location of the private orchards. Anything that needs a real geographic
reference, including the satellite backdrops of the paper's figures, cannot be
reproduced from the public data, by design.

See [docs/data.md](docs/data.md) for the layout the scripts expect.

## Layout

```
src/orchardmap/      the pipeline: localization, ground plane, height gate,
                     weed detection, projection and merging, trunk and row
                     detection, priority
tools/pick_trunks/   the tool used to pick trunk centres in the LiDAR map
tools/labels/        frame selection and the Make Sense export converter
runs/ca10/           the second-orchard run of the paper, end to end
runs/appendix_export/  per-frame export of detections, gate line and scan
figures/paper/       scripts that build the paper's figures
figures/analysis/    the analyses behind the tables
figures/lib/         shared plotting style and helpers
                     (see figures/README.md for the two PYTHONPATH entries)
config/              paths (copy paths.example.py to paths.py)
docs/                data layout, reproduction, environment
```

## Getting started

```bash
git clone https://github.com/OWNER/orchard-weed-mapping
cd orchard-weed-mapping
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config/paths.example.py config/paths.py   # then edit DATA_ROOT
export ORCHARD_DATA=/path/to/orchard-data
```

The SLAM estimators and the ROS 2 recording stack are not Python packages; see
[docs/environment.md](docs/environment.md).

## Reproducing the paper

[docs/reproduce.md](docs/reproduce.md) lists which script produces which figure
and table, and which ones are not yet in this repository.

## Citing

Cite the paper and, if you use the recordings, the dataset records above.
`CITATION.cff` carries the machine-readable form.

## Licence

MIT for the code (see `LICENSE`). The datasets are licensed separately on
Zenodo. The satellite imagery in the paper's figures is © Esri, Maxar,
Earthstar Geographics and is not redistributed here. The bounding boxes of
the labelled images were drawn in a third-party annotation program, named in
the paper, which is likewise not redistributed.
