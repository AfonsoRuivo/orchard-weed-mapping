# Data layout

Unpack both Zenodo records under one folder and point `ORCHARD_DATA` at it.

```
$ORCHARD_DATA/
├── cb11_session/        rosbag2 .db3 + metadata.yaml (GNSS relocated),
│                        gnss_enu.csv (local ENU from the first fix)
├── calibration/         camera intrinsics, LiDAR–camera extrinsic,
│                        LiDAR–IMU rotation, calibration notes
├── trunk_survey/        tape measurements, surveyed trunk positions in the
│                        local frame, description of the survey method
├── labelled_images/     images (EXIF stripped), YOLO labels, train/val split,
│                        class list, test set
└── ca10_session/        the second-orchard record
```

## Derived products

Most figure and analysis scripts do not read the bag. They read the products
of a pipeline run: the trajectory, the dense point-cloud map, the detected
trunks and rows, the weed entities and the priority tables. Running the
pipeline from the bag writes these under `$ORCHARD_OUT`; the scripts then read
them from there.

## What cannot be reproduced from the public data

- **Absolute position.** The released GNSS fixes sit at a synthetic origin, so
  the East–North–Up frames in the scripts are relative. Figures drawn over
  satellite imagery need the true fixes and are therefore not reproducible
  from the public record.
- **Imagery.** The satellite backdrops are third-party (Esri, Maxar, Earthstar
  Geographics) and are not redistributed.
