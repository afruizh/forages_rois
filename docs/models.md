---
title: Models
---

# Models

ForagesROIs uses two ONNX models, loaded by file name from the models folder (by default `models` next to the executable; change it in **File > Preferences**, where each model can also be replaced by another file). They are included in the release. If they are missing, **File > Download Models** (or `ForagesROIs.exe --download_models`) downloads version v1.0.0 from Hugging Face and checks each file against its SHA256 checksum before using it.

| File | Task | Model | Input | Classes |
| :--- | :--- | :--- | :--- | :--- |
| `forages_rois_yolo_full_1024.onnx` | plant detection | YOLOv8n detect | 1024 px tiles | urochloa, megathyrsus |
| `crop_detection.onnx` | crop field segmentation | YOLO11m-seg | whole orthomosaic at 640 px | crop (a second class, target, is not used) |

The plant detector was trained on the dataset *Forage grasses in crop fields from ultra-high spatial resolution UAV-based imagery* (Harvard Dataverse, [doi:10.7910/DVN/DBGUFW](https://doi.org/10.7910/DVN/DBGUFW)): 8 orthomosaics of 1.1 to 1.7 cm per pixel, 374 tiles and 12,346 annotated plants. On the test tiles it reached precision 0.97, recall 0.97 and mAP@0.5 0.99, with about 97 % of the plants assigned to the right species. These values are optimistic, because the test tiles come from the same orthomosaics as the training tiles; expect lower values on new fields.

## License

The models were trained with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and are distributed under the **AGPL-3.0** license, separately from the code (Apache License 2.0). Download: [huggingface.co/anfruizhu/forages-rois-models](https://huggingface.co/anfruizhu/forages-rois-models), which also has the model card and the SHA256 checksums.

## Resolution

The plant model was trained on orthomosaics of 1.1 to 1.7 cm per pixel. For plant detection the application resamples finer rasters to 1.5 cm (option **Resample fine rasters to the model resolution**). The values for each model are in `MODEL_RESOLUTION` in `custom_processor.py`; record them there when a new model is added.

## Using other models

To try another model, choose its file in **File > Preferences > Models** (or `--plant_model` and `--crop_model` on the command line); the file can have any name. It must keep the input and output of the model it replaces: input `images` of 1024 px for plants and 640 px for crop fields, and for plants a YOLO detect output (YOLOv8 or YOLO11 without NMS, or YOLO26 with its final boxes), for crop fields a YOLO segment output with NMS. The classes must be in the same order (plants: 0 urochloa, 1 megathyrsus). The detection cache depends on the model file, so a new model always runs again, and the JSON summary of each run gives the `model` file and the `device` used.
