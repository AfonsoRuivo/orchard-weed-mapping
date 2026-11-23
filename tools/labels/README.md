# labels

Helpers around the annotation, which was done in Make Sense
(<https://www.makesense.ai>), a third-party browser tool that is not
redistributed here.

| File | What it does |
|---|---|
| `pick_frames.py` | chooses the frames to label: the 30-frame test set evenly spaced over CB-11, and the 29 tuning frames, each between two test frames and at least ~14 s from any of them |
| `csv_to_yolo.py` | converts a Make Sense CSV export to YOLO `.txt`, one per image, empty where an image has no boxes |

Class indices are `0 weed`, `1 trunk`, `2 weed_cluster`.
