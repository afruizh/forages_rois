---
title: Models
---

# Models

ForagesROIs uses two ONNX models, loaded by file name from the models folder (by default `models` next to the executable; change it with **File > Set Models Folder...**). They are included in the release. If they are missing, **File > Download Models** (or `ForagesROIs.exe --download_models`) downloads version v1.0.0 from Hugging Face and checks each file against its SHA256 checksum before using it.

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

A new model must keep the file name and the output format of the one it replaces (YOLO detect exported to ONNX without NMS for plants, YOLO segment with NMS for crop fields). Put it in a separate folder and select that folder with **File > Set Models Folder...** to compare it with the current one.
