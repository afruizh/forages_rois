---
title: Scripts and AI agents
---

# Scripts and AI agents

This page is for programs that run ForagesROIs on their own: scripts, workflow tools and AI agents (for example an assistant with a terminal). It says how to call each task, how to read the result without parsing the log, and what to check before using the outputs. Everything here also works for a person at the command line.

## In short

1. Call `ForagesROIs.exe --cli --task <task> ... --output <folder> --json`.
2. Wait for the process to end and read its **exit code**: `0` completed, `1` failed.
3. Parse the **last line** of the output as JSON; the lines before it are the log.
4. Check `status`, `skipped` and `counts` (see [Checks](#checks-before-using-the-results)) before the next step.

`ForagesROIs.exe --help` lists every option and, at the end, the tasks with their inputs and outputs. The full option reference is the [Command line](cli.md) page.

## Tasks

| Task | `--input` | Other inputs | Writes in `--output` | Typical next step |
| :--- | :--- | :--- | :--- | :--- |
| `crop_detection` | orthomosaic | | `<raster>_crops.shp` | `tiling_detection_on_crop` |
| `tiling_detection_on_crop` | orthomosaic | `<raster>_crops.shp` in the output folder, or `--crop_shapefile` | `<raster>_on_crop.shp` | `plant_traits` |
| `tiling_detection` | orthomosaic | | `<raster>.shp` | `plant_traits` |
| `tiling_detection_only` | orthomosaic | | `<raster>.shp` (not numbered) | `plot_numbering` |
| `plot_numbering` | detections shapefile | for an `_on_crop` result, `<raster>_crops.shp` next to it, or `--crop_shapefile` | the input shapefile, numbered again (or the `--output` `.shp`) | `plant_traits` |
| `postprocessing` | detections shapefile | | `<input>_postprocess.shp` | `plot_numbering` |
| `plant_traits` | numbered shapefile | `--raster` orthomosaic | `<input>_traits.shp` and `.csv` | read the CSV |
| `export_coco` | plant boxes shapefile | `--raster` orthomosaic, optional `--crop_shapefile` | `<input>_coco/` | annotation or training tools |

Options that change results: `--confidence` (detection threshold, default 0.10), `--empty_plots` (number planting positions), `--bands` (which bands are red, green and blue in a multispectral raster), `--no_resample`, `--index`, `--threshold`. See [Command line](cli.md).

## The JSON result

| Key | Meaning |
| :--- | :--- |
| `status` | `completed` or `error`. |
| `task` | The task that was run. |
| `message` | Human readable result or error (for example "No plants were detected ... Pixel size of the raster: about 14 cm"). Show it to the user when something failed. |
| `output_file` | The output shapefile (for `export_coco`, the `instances_default.json` of the dataset). |
| `skipped` | `true` when the output already existed and nothing was run (add `--overwrite` to run again). |
| `model`, `device` | Detection tasks: the model file used and where it ran (`GPU` or `CPU`). |
| `counts` | Read from the output (see below). |
| `csv_file` | For `plant_traits`: the table with one row per plot. |

`counts` by task:

- **Detection tasks, `plot_numbering`, `postprocessing`, `plant_traits`:** `features`, `plants`, `empty_plots`, `numbered_plants`, `plants_left_out` (isolated detections left without number), `fields` (with crop fields) and `classes` (plants per species).
- **`crop_detection`:** `features` and `fields`.
- **`export_coco`:** `images` (tiles), `annotations` (boxes) and `classes` (boxes per species).

Example (`tiling_detection_on_crop` with `--empty_plots`):

```json
{"status": "completed", "task": "tiling_detection_on_crop", "output_file": "D:\\results\\trial_on_crop.shp", "skipped": false,
 "counts": {"features": 2490, "plants": 1561, "empty_plots": 929, "numbered_plants": 1558, "plants_left_out": 3, "fields": 2,
            "classes": {"urochloa": 514, "megathyrsus": 1047}}, "message": "..."}
```

## A complete run

Use the same `--output` folder for every step, so each step finds the files of the previous one:

1. `--task crop_detection --input trial.tif` writes `trial_crops.shp`.
2. `--task tiling_detection_on_crop --input trial.tif` reads `trial_crops.shp` and writes `trial_on_crop.shp`.
3. `--task plant_traits --input trial_on_crop.shp --raster trial.tif` writes `trial_on_crop_traits.shp` and `trial_on_crop_traits.csv`.

### From Python

```python
import json
import subprocess

EXE = r"C:\Programs\ForagesROIs\ForagesROIs.exe"

def run(task, *args):
    """Run one task and return its JSON result; raise on failure."""
    proc = subprocess.run([EXE, "--cli", "--task", task, *args, "--json"],
                          capture_output=True, text=True)
    lines = proc.stdout.strip().splitlines()
    result = json.loads(lines[-1]) if lines else {"status": "error", "message": proc.stderr}
    if proc.returncode != 0 or result["status"] != "completed":
        raise RuntimeError(f"{task}: {result['message']}")
    return result

out = r"D:\results"
run("crop_detection", "--input", r"D:\flights\trial.tif", "--output", out)
plants = run("tiling_detection_on_crop", "--input", r"D:\flights\trial.tif", "--output", out)
print(plants["counts"])
traits = run("plant_traits", "--input", plants["output_file"], "--raster", r"D:\flights\trial.tif", "--output", out)
print(traits["csv_file"])
```

### From PowerShell

PowerShell does not wait for the program on its own. Run it through `cmd /c`, which waits and keeps the output:

```powershell
$out = cmd /c "C:\Programs\ForagesROIs\ForagesROIs.exe" --cli --task crop_detection --input D:\flights\trial.tif --output D:\results --json
$result = $out[-1] | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { Write-Error $result.message }
```

### Many orthomosaics

Loop over the files and give each its own output folder, so that results and cache files of different flights never mix. Because a task with an existing output is skipped, a batch that stopped halfway can simply be started again: finished steps report `skipped: true` and the rest run.

## Checks before using the results

- **Status:** `status` is `completed`. If `skipped` is `true`, the output is from an earlier run, possibly with other options; run again with `--overwrite` (or use a new `--output`) when the options changed.
- **Plant count:** `counts.plants` is plausible for the trial (number of plots per field times the fields). Very few plants usually means a raster that is too coarse (the message gives its pixel size) or a scene unlike the training data.
- **Numbering:** `counts.plants_left_out` is small compared with `counts.plants`. With `--empty_plots`, many empty plots in one field point to missed detections there.
- **Species:** `counts.classes` matches what was planted; a species that is not in the trial points to misclassified plants.
- **Traits:** in the CSV, plants with `canopy_m2` 0 had no plant pixels; many of them point to a wrong threshold or index.

## Common errors

| Message contains | Cause | What to do |
| :--- | :--- | :--- |
| `No plants were detected` | Raster too coarse, or very unlike the training images | Use the full resolution orthomosaic (1 to 3 cm per pixel), or lower `--confidence`. |
| `Crop shapefile not found` | `tiling_detection_on_crop` without crop fields | Run `crop_detection` with the same `--output`, or give `--crop_shapefile`. |
| `Output already exists` | The output file is there (`skipped` is `true`) | Add `--overwrite`, or use another `--output`. |
| `need the orthomosaic` or `need the plant ROIs shapefile` | `plant_traits` or `export_coco` without both inputs | Give the shapefile as `--input` and the orthomosaic as `--raster`. |
| `cannot be computed on this raster` | `--index NDVI`, `NDRE` or `GNDVI` on an RGB raster | Use `RGBVI` or `ExG`, or give `--nir_band`. |
| `Model not found` | The models were not downloaded | Run `ForagesROIs.exe --download_models` once. |

## Rules for agents

- Write only inside the `--output` folder. Never delete or overwrite the input orthomosaic or files outside the output folder; clear the cache with `--clear_cache` instead of deleting folders.
- Do not run two tasks on the same `--output` folder at the same time.
- A run on a large orthomosaic can take minutes; wait for the process to end instead of starting it again.
- Report `message` to the user when a task fails, and the main `counts` when it completes.
