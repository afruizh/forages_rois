---
title: Command line (CLI)
---

# Command line (CLI)

`ForagesROIs.exe --cli` runs one task without opening the window, which is useful for scripts and batch processing. The tasks and outputs are the same as in the [GUI](gui.md).

```
ForagesROIs.exe --cli --task <task> --input <file> --output <folder or .shp> [options]
```

`--output` can be a folder (the output name is made from the input name) or the full path of the output `.shp`.

## Options

| Option | Tasks | Description |
| :--- | :--- | :--- |
| `--cli` | | Run without the window. Without it, `--input` and `--output` only prefill the GUI. |
| `--task` | | `crop_detection`, `tiling_detection`, `tiling_detection_on_crop`, `tiling_detection_only`, `plot_numbering`, `postprocessing`, `plant_traits` or `export_coco`. |
| `--input` | all | Orthomosaic for the detection tasks; shapefile for `plot_numbering`, `postprocessing`, `plant_traits` and `export_coco`. |
| `--output` | all | Output folder or `.shp` file. |
| `--confidence` | detection tasks | Detection threshold from 0 to 1 (default 0.10). |
| `--empty_plots` | `tiling_detection`, `tiling_detection_on_crop`, `plot_numbering` | Number planting positions, so that a missing plant gets an empty plot with its own number. Off unless given: only the detected plants are numbered. |
| `--serpentine` | `plot_numbering` | Reverse the direction every other row. **Off unless given** (the GUI has it on by default). |
| `--start_corner` | `plot_numbering` | `NW` (default), `NE`, `SW` or `SE`. |
| `--swap_axes` | `plot_numbering` | Number along the other axis of the trial. |
| `--raster` | `plant_traits`, `export_coco` | The orthomosaic (the shapefile is `--input`). |
| `--crop_shapefile` | `tiling_detection_on_crop`, `export_coco` | Crop fields (default: `<raster>_crops.shp` in the output folder). |
| `--index` | `plant_traits` | `RGBVI`, `ExG`, `NDVI`, `NDRE` or `GNDVI`. Default: `NDVI` when the raster has a near infrared band, otherwise `RGBVI`. |
| `--nir_band` | `plant_traits` | Number of the near infrared band (default: found from the band names, such as NIR or 840 nm). |
| `--rededge_band` | `plant_traits` | Number of the red edge band (default: found from the band names, such as RedEdge or 730 nm). |
| `--threshold` | `plant_traits` | Fixed index threshold (default: automatic, Otsu's method per field). |
| `--bands` | tasks that read the raster | Bands used as red, green and blue, for example `3,2,1` for a multispectral orthomosaic ordered blue, green, red. Default: guessed from the band colours and names, otherwise `1,2,3`. |
| `--no_resample` | detection tasks | Do not resample rasters finer than the model resolution (by default, below 1.1 cm per pixel they are resampled to 1.5 cm). |
| `--clear_cache` | | `default` deletes the default cache; `output` deletes the cache of the `--output` folder (results are kept). Alone it only clears; with `--cli` it clears before the task. |
| `--json` | all | Print a JSON summary as the last line (status, output file, counts); see [Scripts and AI agents](agents.md). The exit code is 0 on success and 1 on error, with or without `--json`. |
| `--download_models` | | Download the missing models from Hugging Face into the models folder. Alone it only downloads; with `--cli` it downloads before running the task. |

`ForagesROIs.exe --help` lists all options. For `tiling_detection_on_crop`, the crop fields must be in the output folder as `<raster>_crops.shp`, so run `crop_detection` with the same output folder first.

## Examples

Full pipeline on one orthomosaic (in `cmd`):

```
ForagesROIs.exe --cli --task crop_detection --input D:\flights\trial.tif --output D:\results
ForagesROIs.exe --cli --task tiling_detection_on_crop --input D:\flights\trial.tif --output D:\results
ForagesROIs.exe --cli --task plant_traits --raster D:\flights\trial.tif --input D:\results\trial_on_crop.shp --output D:\results
ForagesROIs.exe --cli --task export_coco --raster D:\flights\trial.tif --input D:\results\trial_on_crop.shp --output D:\results
```

Download the models (for example after installing from source, or if the `models` folder was deleted):

```
ForagesROIs.exe --download_models
```

Number again from the south western corner, row by row, with empty plots for missing plants. With a folder as `--output`, the new numbers are written into the input shapefile itself:

```
ForagesROIs.exe --cli --task plot_numbering --input D:\results\trial_on_crop.shp --output D:\results --start_corner SW --empty_plots
```

To keep the input as it is, give a `.shp` path as `--output` and the numbered copy is written there:

```
ForagesROIs.exe --cli --task plot_numbering --input D:\results\trial_on_crop.shp --output D:\results\trial_SW.shp --start_corner SW --empty_plots
```

## PowerShell

`ForagesROIs.exe` is a windowed program, so PowerShell does not wait for it: the prompt comes back at once while the task keeps running and printing in the same terminal. The output files are complete only when the process ends. To wait (in scripts, or to time a run), use one of:

```
Start-Process .\ForagesROIs.exe -ArgumentList "--cli --task crop_detection --input D:\flights\trial.tif --output D:\results" -Wait -NoNewWindow
cmd /c ForagesROIs.exe --cli --task crop_detection --input D:\flights\trial.tif --output D:\results
```

In `cmd` the command waits normally.

## Batch processing

All orthomosaics of a folder, in `cmd`:

```
for %f in (D:\flights\*.tif) do (
  ForagesROIs.exe --cli --task crop_detection --input "%f" --output D:\results
  ForagesROIs.exe --cli --task tiling_detection_on_crop --input "%f" --output D:\results
)
```

(Use `%%f` instead of `%f` inside a `.bat` file.)

## Processing time

On a laptop with an Intel Core i7-1260P, 16 GB RAM and an NVIDIA GeForce MX550 (2 GB), for an orthomosaic of 6292 × 5116 pixels at 3 cm with two fields and 1,561 plants: `crop_detection` 8 s, `tiling_detection_on_crop` 17 s and `plant_traits` 8 s.
