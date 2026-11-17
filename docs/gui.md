---
title: Desktop application (GUI)
---

# Desktop application (GUI)

## Main window

The left panel (**Settings**) holds the inputs, the task and the options. The right panel (**Preview**) shows the orthomosaic and the last result. The status bar at the bottom shows the progress.

| Field | Used by | Description |
| :--- | :--- | :--- |
| **Input Raster** | all tasks on a raster, `plant_traits` | The orthomosaic (GeoTIFF). You can also drag and drop it onto the preview. |
| **Input Shapefile** | `plot_numbering`, `postprocessing`, `plant_traits` | Shown only for these tasks: the detections shapefile to process. |
| **Crop Shapefile (Optional)** | `tiling_detection_on_crop` | The crop fields. Leave it blank to use `<raster>_crops.shp` from the output folder (made by `crop_detection`). **X** clears it. |
| **Output Folder** | all tasks | Where the results are written. Preview images go to `.cache_forages_rois` inside it. |
| **Task** | | The processing step to run (see below). |

Next to each path, **...** opens a file dialog and **^** opens the folder that contains it. The dialog opens in the folder of the file already chosen; otherwise in the last folder used for that kind of file (rasters, shapefiles, output folder), remembered between sessions; for shapefiles, in the output folder first.

## Rasters with more than 3 bands

For a raster with more than 3 bands (for example a multispectral orthomosaic with blue, green, red, red edge and NIR), a **Bands R, G, B** row appears under **Input Raster**. Each list shows the bands with their names. The application preselects the most likely red, green and blue bands, from the band colours stored in the file or from their names (such as "Red" or "650 nm"); check them in the preview and change them if needed. The preview updates at once.

The tasks then work on an 8 bit RGB view of the chosen bands, each stretched from its minimum to its maximum, the same conversion used for the training images. 16 bit and reflectance rasters with 3 bands are converted the same way. Normal 8 bit RGB orthomosaics, with or without an alpha band, are used as they are. The output files keep the name of the original raster, and their coordinates match it.

**Note:** the models were trained on RGB camera images. A colour image made from the narrow red, green and blue bands of a multispectral camera looks slightly different, so check the detections on your first flights.

Press **Run Inference** to start and **Cancel** to stop. A message tells you when the task ends, and **Open** in that message opens the output folder. If the output file already exists, nothing is run: delete it or choose another output folder.

## Tasks

| Task | Input | Output | What it does |
| :--- | :--- | :--- | :--- |
| `crop_detection` | raster | `<raster>_crops.shp` | Field polygons. |
| `tiling_detection_on_crop` | raster and crop fields | `<raster>_on_crop.shp` | Detects plants inside the fields and numbers each field separately. Recommended. |
| `tiling_detection` | raster | `<raster>.shp` | Detects plants on the whole orthomosaic and numbers them. |
| `tiling_detection_only` | raster | `<raster>.shp` | Detects plants without numbering. |
| `plot_numbering` | detections shapefile | the same shapefile | Numbers again an existing result with other options, without running the detection again. The new numbers are written into the input shapefile, like the edits; unsaved edits are saved first. |
| `postprocessing` | detections shapefile | `<input>_postprocess.shp` | Removes duplicated and badly shaped boxes, without numbering. |
| `plant_traits` | raster and numbered shapefile | `<input>_traits.shp` and `.csv` | Canopy outline, area and cover of each plant. |
| `export_coco` | raster and plant boxes shapefile | `<input>_coco/` | Exports the boxes as a COCO dataset of 1024 px RGB tiles, to annotate further or retrain (see [Outputs](outputs.md)). |

## Advanced options

Press **Advanced** to show them. Only the options of the selected task are shown.

**Detection** (detection tasks)

- **Confidence**: detection threshold from 0 to 1 (default 0.10). Lower values find more plants and more false detections.
- **Resample fine rasters to the model resolution**: on by default. The plant model was trained on orthomosaics of 1.1 to 1.7 cm per pixel; a finer raster (below 1.1 cm) is resampled to 1.5 cm for the detection, so plants have the size the model learned. The outputs keep the coordinates of the original raster. Coarser rasters are used as they are.
**Numbering**

- **Serpentine** (`plot_numbering`): reverse the direction every other row. On by default.
- **Start corner** (`plot_numbering`): corner where number 1 is placed: NW, NE, SW or SE.
- **Swap rows and columns** (`plot_numbering`): use when the trial was planted along the other axis.
- **Empty plots for missing plants** (detection tasks and `plot_numbering`): number planting positions, so that a missing plant gets an empty plot with its own number and the numbers after it do not shift. Off by default: only the detected plants are numbered.

The detection tasks always number in serpentine order from the north western corner. To change the order, run `plot_numbering` on their output.

**Plant traits** (`plant_traits`)

- **Index**: vegetation index used to separate plant from soil. **Auto** (default) uses NDVI when the raster has a near infrared band, otherwise RGBVI. ExG, NDRE and GNDVI can also be chosen (see [Outputs](outputs.md)). With a multispectral raster, the plant traits read its original bands, and the NDVI, NDRE and GNDVI of each plant are also written.
- **Threshold**: fixed index threshold. Leave it empty to choose it automatically for each field (Otsu's method).

## Preview

- **Use Native Vector Overlay (High Res)**: draws the results as vectors over the image, sharp at any zoom.
- **Show detections (boxes)**: shows or hides the boxes of the plants, with their plot numbers.
- **Show plant outlines**: after `plant_traits`, shows the canopy outlines in green under the boxes, so both can be compared. The boxes stay visible; uncheck either layer to see the other alone.
- **Color each detection**: one colour per plant, so that overlapping boxes of neighbouring plants can be told apart.

Empty plots are drawn grey and dashed. Zoom with the mouse wheel and pan by dragging.

## Erasing and adding detections (Edit group)

The **Edit** group works on the detections shown in the preview (it needs **Use Native Vector Overlay** and **Show detections**). It has two tools, **Erase boxes** and **Add boxes**; only one is on at a time (the button stays pressed; press it again to stop). In both, the left button still pans the view.

- **Erase boxes:** hold the **right** mouse button and trace over the boxes to remove. Every box the stroke touches disappears at once (the stroke is drawn in red).
- **Add boxes:** choose the **Species**, then drag with the **right** mouse button from one corner of the plant to the other. The box is drawn while you drag and added when you release (new boxes are orange until saved). **Esc** cancels the box being drawn; very small boxes (a click) are ignored.

Then:

- **Undo** (or `Ctrl+Z`) undoes the last stroke or added box; press it again for earlier ones.
- **Discard** forgets all erased and added boxes since the last save.
- **Save** writes the changes into the shown shapefile itself and puts it in **Input Shapefile**. After saving, they cannot be undone.

The numbers of the remaining plants are kept. An added box gets the chosen species (`class` and `class_name`), an empty `score` (so it can be told apart from detections), `grid_id` 0 (not numbered) and, with crop fields, the `crop_id` of the nearest plant. To number the plants again (for example after adding missed plants, or with empty plots for the erased positions), run `plot_numbering`: the shapefile is already its input.

## Inputs filled automatically

When a task ends, its output is put where the next task reads it: the crop fields of `crop_detection` go to **Crop Shapefile**, and the plants of the detection tasks, `plot_numbering` and `postprocessing` go to **Input Shapefile**, ready for `plot_numbering` or `plant_traits`. You can still choose another file with **...**.

## Menus

- **File > New** (`Ctrl+N`): starts again with an empty preview. It clears the raster, the detections and outlines shown, and **Input Raster**, **Input Shapefile** and **Crop Shapefile**; the **Output Folder**, the task and the options are kept, and no file is deleted. If there are erased or added boxes not saved yet, it asks first: **Save** writes them into the shapefile, **Discard** drops them, **Cancel** keeps everything as it was. It is disabled while a task runs.
- **File > Set Models Folder...**: choose the folder that contains the models (`.onnx`). The application checks it and remembers it. By default it uses the `models` folder next to the executable.
- **File > Download Models**: downloads the missing models from [Hugging Face](https://huggingface.co/anfruizhu/forages-rois-models) (about 100 MB) into the models folder, checks them and starts using them. The progress is shown in the status bar and **Cancel** stops it. If models are missing when the application starts, it offers to download them. If the models folder cannot be written, they go to a `models` folder in your application data.
- **File > Cache**: the application keeps temporary files (tiles, detections, previews and RGB views) so that a second run is faster.
  - With an output folder they go to `.cache_forages_rois` inside it. Before an output folder is chosen they go to the **default cache**, `%LOCALAPPDATA%\ForagesROIs\cache`. Nothing is written next to the input raster.
  - **Open Default Cache Folder** and **Open Output Folder Cache** open them in the file explorer.
  - **Set Default Cache Folder...** moves the default cache (a `.cache_forages_rois` folder is made inside the folder you choose), for example to another disk.
  - **Clear Default Cache** and **Clear Output Folder Cache** show their size and delete them after asking. The results in the output folder are kept.
- **Help > About**: version and credits.

## From QGIS

With the ForagesROIs launcher plugin, right click an orthomosaic layer and choose **Open in ForagesROIs**: the application opens with the raster already set. **Load ForagesROIs results** adds the outputs to the project with their style.

## Common messages

- **No plants were detected, so no output was created**: usually the orthomosaic is too coarse. The message shows its pixel size; use the full resolution orthomosaic (1 to 3 cm per pixel) or lower the confidence.
- **Error: Crop shapefile not found**: run `crop_detection` first, or choose a crop shapefile.
- **Output already exists, nothing was run**: delete the output or choose another output folder.
