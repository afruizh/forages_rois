# Forages ROIs: Automated Forage Grass Detection in Aerial Imagery

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![QGIS](https://img.shields.io/badge/QGIS-Compatible-green.svg)](https://qgis.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-red.svg)](https://onnxruntime.ai)

**An intelligent computer vision system for automatic of forage grasses in high-resolution UAV imagery using deep learning and GIS integration.**

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
``` -->

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

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

