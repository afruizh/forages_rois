# Forages ROIs: Automated Forage Grass Detection in Aerial Imagery

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![QGIS](https://img.shields.io/badge/QGIS-Compatible-green.svg)](https://qgis.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-red.svg)](https://onnxruntime.ai)

**An intelligent computer vision system for automatic of forage grasses in high-resolution UAV imagery using deep learning and GIS integration.**

**Documentation:** user guide for the desktop application and the command line at https://afruizh.github.io/forages_rois/ (source in [`docs`](docs)).

The documentation is built with MkDocs (`mkdocs.yml`). To view it locally, from this folder: `uvx --with mkdocs-material mkdocs serve`, then open http://127.0.0.1:8000.

---

## Features

### **Core Capabilities**
- **AI-Powered Detection**: Advanced YOLO-based object detection for forage identification
- **Standalone Desktop App**: Modern PySide6 GUI with intuitive interface
- **QGIS Integration**: Seamless plugin for professional GIS workflows
- **Batch Processing**: Multi-threaded processing for large datasets
- **Geospatial Support**: Full coordinate system and projection handling

### **Technical Features**
- **ONNX Runtime**: Optimized model inference for fast processing
- **Grid Generation**: Automated ROI grid creation with customizable overlap
- **Deployment Ready**: Packaged executable for easy distribution

---

## Requirements

### System Requirements
- **OS**: Windows 10/11 (primary), Linux/macOS (experimental)
- **Python**: 3.11.x (required for ONNX compatibility)
<!-- - **Memory**: -->
- **GPU**: NVIDIA GPU with CUDA support (optional, for acceleration)

### Option 3: QGIS Plugin Installation
1. **Copy** the `plugins/foragesrois` folder to your QGIS plugins directory
2. **Enable** the plugin in QGIS Plugin Manager
3. **Access** through the plugins menu or toolbar

---

## Architecture

```
forages_rois/
├── local_app/              # Standalone desktop application
│   ├── ForagesROIs.py         # Main application entry point
│   ├── custom_processor.py    # Core processing logic
│   ├── interface/             # Interface with processing
│   └── models/               # AI models (ONNX format)
├── plugins/               # QGIS plugin integration
│   ├── foragesrois    # QGIS plugin
```

Detail of `local_app` (v1.0):

```
local_app/
├── ForagesROIs.py            # entry point: GUI, or a task with --cli (--json summary)
├── custom_processor.py       # pipeline: tiling, inference, merging, numbering, plant traits
├── view.qml                  # main window (QML)
├── resources.qrc             # Qt resources (view.qml and res/), compiled to rc_resources.py
├── res/                      # images: icon.png, icon.ico, splash.png, tf.png (program logo)
├── interface/                # Qt side
│   ├── interface.py          #   bridge between the QML window and the processing
│   ├── processor.py          #   runs each task in a background thread
│   ├── raster_preview_item.py#   preview: raster, detection boxes and plant outlines
│   ├── detection_editor.py   #   erase and add boxes, with undo
│   └── batchprocessor.py     #   inference over the tiles
├── processing/               # processing steps without Qt
│   ├── rgb_bands.py          #   RGB view of multiband rasters, resampling (GDAL VRT)
│   ├── spectral_indices.py   #   RGBVI, ExG, NDVI, NDRE, GNDVI for plant traits
│   ├── coco_export.py        #   COCO dataset of RGB tiles from reviewed boxes
│   └── yolo_seg_util.py      #   decoding of the crop field model output
├── utils/                    # shared helpers
│   ├── cachemanager.py       #   cache folders
│   ├── app_settings.py       #   QSettings (models and cache folders, last folders)
│   └── result_summary.py     #   JSON summary of a result
├── models/                   # ONNX models (downloaded from Hugging Face)
└── deploy.ps1, pysidedeploy.spec, environment_app.yml   # build
```

---

## AI Models

The system uses state-of-the-art YOLO (You Only Look Once) models optimized for forage detection:

| Model | Input Size | Accuracy | Speed | Use Case |
|-------|------------|----------|-------|----------|
| `forages_rois_yolo_full_1024.onnx` | 1024×1024 | High | Medium | Production |
<!-- | `forages_rois_yolo_small.onnx` | 640×640 | Medium | Fast | Real-time | -->

<!-- ### Model Performance
- **Precision**: >90% on validation dataset
- **Recall**: >85% for forage detection
- **Speed**: ~100ms per image (GPU), ~500ms (CPU) -->

---

## Dataset

Our research is based on the comprehensive dataset:

**[Forage grasses in crop fields from ultra-high spatial resolution UAV-based imagery](https://dataverse.harvard.edu/citation?persistentId=doi:10.7910/DVN/DBGUFW)**

- **Resolution**: Ultra-high spatial resolution UAV imagery
- **Coverage**: crop fields and forage types
- **Annotations**: Precise polygon annotations for training
- **Format**: Compatible with COCO

---

<!-- ## Deployment

### Building Standalone Executable
```bash
cd local_app

# Generate resources (if needed)
pyside6-rcc resources.qrc -o rc_resources.py

# Create deployment specification
pyside6-deploy ForagesROIs.py --init

# Build executable
pyside6-deploy -c pysidedeploy.spec
```

### Configuration Options
```ini
# pysidedeploy.spec
title = ForagesROIs
icon = .\icon.ico
modules = Core,Gui,Network,Widgets,OpenGL
mode = release
arch = "x86_64"
extra_args = --quiet --noinclude-qt-translations
```

### Building with deploy.ps1 (forages_rois_app environment)
Standalone folder (exe plus DLLs and data) built with `pyside6-deploy` (Nuitka). Use the `forages_rois_app` environment (pip PySide6, which includes `pyside6-deploy`), not `forages_rois`.

```bash
cd local_app

# Once: create the build environment
conda env create -f environment_app.yml

# Each build (PowerShell)
conda activate forages_rois_app
.\deploy.ps1
```

`deploy.ps1` compiles `resources.qrc` with this environment's `pyside6-rcc` (pip Qt cannot read resources compiled by conda's rcc), runs `pyside6-deploy -c pysidedeploy.spec -f` and copies the files Nuitka does not bundle. Result: `local_app\dist\ForagesROIs.dist`; zip that folder to distribute it. If Windows blocks scripts, run `powershell -ExecutionPolicy Bypass -File .\deploy.ps1`.

### Configuration Options for deploy.ps1
```ini
# pysidedeploy.spec (main settings)
title = ForagesROIs
icon = icon.ico
exec_directory = dist          # pyside6-deploy copies every other folder in local_app into the build
python_path = <forages_rois_app>\python.exe
qml_files = view.qml
mode = standalone              # not onefile: the CUDA DLLs (over 1 GB) would be unpacked at every start
extra_args = --quiet --noinclude-qt-translations --windows-console-mode=attach
```

`--windows-console-mode=attach`: no terminal window on double click; `--cli` output still goes to the terminal it is run from.

### Extra files required next to ForagesROIs.exe
Copied by `deploy.ps1` from the environment (`%CONDA_PREFIX%\Library`):
- `models\`: `forages_rois_yolo_full_1024.onnx` and `crop_detection.onnx`.
- `proj_dir\share\proj\`: PROJ data from `Library\share\proj` (must match the PROJ version of the build). Without it: `Cannot find proj.db`.
- `gdal_data\`: GDAL data from `Library\share\gdal`.
- DLLs from `Library\bin` loaded at run time, which Nuitka does not detect:
  - CUDA: `cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`, `cufft64_11.dll`
  - cuDNN: all `cudnn*64_9.dll` (without them the exe crashes at the first tile)
  - NVRTC: `nvrtc64_120_0.dll`, `nvrtc64_120_0.alt.dll`, `nvrtc-builtins64_129.dll`
  - OpenBLAS: `openblas.dll` (numpy's BLAS; the environment uses OpenBLAS instead of MKL, which added about 540 MB and needed the OpenMP DLLs `libiomp5md.dll` and `libomp.dll`)

### Testing the build
In a terminal with no conda environment active, open the GUI and run one detection, for example `ForagesROIs.exe --cli --task tiling_detection --input sample.tif --output out\detections.shp`. Target PCs need an NVIDIA driver that supports CUDA 12.9.

Because of `--windows-console-mode=attach` the exe is a GUI program, so PowerShell does not wait for it: the prompt comes back at once while the run continues and prints to the same terminal. To wait until it finishes (in scripts, or to time a run), use one of:
- `Start-Process .\ForagesROIs.exe -ArgumentList "--cli --task tiling_detection --input sample.tif --output out" -Wait -NoNewWindow`
- `cmd /c ForagesROIs.exe --cli --task tiling_detection --input sample.tif --output out`

Output files are complete only when the process has ended. -->

### Command line use of ForagesROIs.exe
`ForagesROIs.exe --cli --task <task> --input <raster or shapefile> --output <folder or .shp>` runs a task without the GUI (`ForagesROIs.exe --help` lists all options). In PowerShell the prompt returns before the run ends; add `Start-Process ... -Wait -NoNewWindow` or run it through `cmd /c ForagesROIs.exe ...` to wait for it.

```
conda env create -f environment.yml
conda activate forages_rois
```

## Updates

### V1.0

- [X] Automatic tiling (tiling, processing, merging)
- [X] Cache System
- [X] Migration rasterio to gdal
- [X] Migration geopandas to gdal
- [ ] Adding crop detector model

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

**Note:** the trained models (`.onnx` files in `models`) are not covered by the Apache License. They were trained with Ultralytics YOLO and are distributed separately under the **AGPL-3.0** license at https://huggingface.co/anfruizhu/forages-rois-models.

## Acknowledgment

- **Bioversity International & CIAT** for research support
- **Harvard Dataverse** for dataset hosting
- **QGIS Community** for the amazing GIS platform
- **OpenCV & ONNX** communities for computer vision tools


## Authors

![Tropical Forage Program](./res/tf_small.png)

Tropical Forages Progam

Alliance Bioversity International & CIAT

<div align="center">

