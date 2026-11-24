# UVSD

This repository contains a small UVSD visual–sonar data sample and selected source files from the accompanying research. UVSD and the separately published UMOD dataset are distinct research resources. The demo directory follows the [UMOD demo layout](https://github.com/UM-Research-Ng/UMOD-dataset/tree/main/UMOD_demo).

## Dataset sample

```text
UVSD_demo/
├── Vis/
│   ├── imgs/
│   └── labels/
├── Sonar/
│   ├── imgs/
│   └── labels/
├── classes.txt
└── pairs.csv
```

The demo contains three optical–sonar image pairs from one acquisition recording, with human-reviewed object boxes for each modality. Match the images using `UVSD_demo/pairs.csv`. The corresponding label files use YOLO format: `class_id x_center y_center width height`, with coordinates normalized to the respective image dimensions. Class IDs are zero-based and follow the order in `UVSD_demo/classes.txt`.

The pairs come from the same frames of a screen recording. Sensor-level acquisition synchronization and geometric registration were unverified. The optical display overlay has been masked without cropping or resizing the optical images, and the boxes retain their original coordinates. Treat these examples as an illustrative subset from a single recording. They are not a benchmark or a train/validation/test split.

## Code

- `code/dafdet/` contains selected fusion-model components and supporting geometry code.
- `code/uvsd/` contains dataset, split-audit and evaluation helpers.
- `code/scripts/` contains selected preprocessing and dataset-export scripts.
- `code/requirements_preprocess.txt` lists preprocessing dependencies from the research workspace.

This is a source snapshot. Some files expect the original workspace layout, additional inputs or upstream dependencies. The demo and source snapshot do not constitute a complete reproduction package.
