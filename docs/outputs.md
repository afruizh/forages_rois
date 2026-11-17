---
title: Outputs
---

# Outputs

All outputs are ESRI shapefiles in the coordinate system of the orthomosaic. A QGIS style (`.qml`, same name) is written next to each one: plant borders in blue, empty plots grey and dashed, and the plot number as label.

## Crop fields (`<raster>_crops.shp`)

| Field | Description |
| :--- | :--- |
| `score` | Confidence of the detection (0 to 1). |
| `class` | 0 = crop field. |
| `class_name` | Name of the class: `crop`. |

## Plants and plots (`<raster>_on_crop.shp`, `<raster>.shp`)

One polygon (box) per plant, plus one per empty plot when empty plots are on.

| Field | Description |
| :--- | :--- |
| `score` | Confidence of the detection (0 to 1). Empty for empty plots. |
| `class` | Species: 0 = *Urochloa*, 1 = *Megathyrsus*. Empty for empty plots. |
| `class_name` | Species name: `urochloa` or `megathyrsus`. Empty for empty plots. |
| `crop_id` | Field the plot belongs to (with `tiling_detection_on_crop`; each field is numbered separately). |
| `grid_id` | Plot number. **0** means an isolated detection that was left without number for you to review. |
| `row` | Row of the plot in its field. |
| `col` | Planting position along the row. |
| `status` | `plant` (detected) or `empty` (planting position without a detected plant). Only when empty plots are on. |

Empty plots have the median size of the plants of the field. Between the plants of a row they are always added; before the first and after the last plant of a row they are added only inside the crop field (or, without crop fields, at most two), so that short rows do not get plots outside the trial.

## COCO dataset (`<input>_coco/`)

`export_coco` turns reviewed plant boxes (for example after erasing and adding boxes in the Edit group) into training tiles, in the layout that CVAT and Roboflow import as "COCO 1.0":

```
<input>_coco/
  annotations/instances_default.json
  images/default/tile_00000.jpg
  ...
```

- **Images:** 1024 × 1024 pixel RGB tiles in 8 bit JPEG, made from the same RGB view as the detection (chosen bands, and rasters finer than 1.1 cm resampled to 1.5 cm). Tiles at the edges are moved inward so that every tile is whole.
- **Tiles exported:** only tiles that contain at least one box; with crop fields (**Crop Shapefile**, or `<raster>_crops.shp` in the output folder), only tiles that touch a field.
- **Annotations:** one box per plant, `bbox` = [x, y, width, height] in tile pixels, clipped to the tile. A box is annotated in a tile only when at least half of it is inside. Empty plots are left out.
- **Categories:** 1 = `urochloa`, 2 = `megathyrsus` (the `class` field plus 1).
- **Traceability:** each image records its raster (`source_raster`) and its position in it (`x_offset`, `y_offset`, in pixels).

**Note:** export only areas you have reviewed. A plant without a box in an exported tile teaches the model that it is background. When the tiles are used for training, split them into training and validation sets by orthomosaic, not by tile, so that the metrics are not optimistic.

## Plant traits (`<input>_traits.shp` and `<input>_traits.csv`)

The polygon is the outline of the canopy instead of the box (green in QGIS and in the preview, where it can be shown together with the boxes). It is computed inside the box of the plant, so it never extends beyond it:

1. A vegetation index is computed for the pixels of the box, from the original bands of the orthomosaic.
2. The threshold that separates plant from soil is chosen for each field (Otsu's method), unless a fixed one is given.
3. A pixel shared by overlapping boxes goes to the plant with the nearest box centre.
4. Leaves split by shadows are joined, the largest group of plant pixels is kept and its holes are filled.

The input fields above are kept, plus:

| Field | Description |
| :--- | :--- |
| `in_grid` | 1 if the plant has a plot number, 0 if not. |
| `canopy_m2` | Canopy area (m²). 0 for empty plots or when no plant pixels were found. |
| `cell_m2` | Area of the plant's cell (m²): its enlarged box, shared with neighbours by nearest centre. |
| `cover_pct` | Canopy as a percentage of the cell. |
| `idx_mean` | Mean vegetation index of the canopy. |
| `idx_name` | Index used to separate plant from soil (RGBVI, ExG, NDVI, NDRE or GNDVI). |
| `idx_thr` | Threshold used for the field. |
| `ndvi_mean`, `ndre_mean`, `gndvi_mean` | Mean NDVI, NDRE and GNDVI of the canopy. Only for rasters with a near infrared band (NDRE also needs a red edge band). |

**Indices:**

- **RGBVI** = (G² − R·B) / (G² + R·B) and **ExG** = (2G − R − B) / (R + G + B) work with any RGB orthomosaic.
- **NDVI** = (NIR − R) / (NIR + R) separates plants from soil and dry straw better than the RGB indices; it is the default when the raster has a near infrared band.
- **NDRE** = (NIR − RE) / (NIR + RE) follows chlorophyll and saturates less than NDVI in dense canopy.
- **GNDVI** = (NIR − G) / (NIR + G).

The near infrared and red edge bands are found from the band names (such as NIR, RedEdge or 840 nm); with the CLI they can also be given with `--nir_band` and `--rededge_band`.

**Note:** the index means can be compared between flights only when the orthomosaic is calibrated reflectance. For the plant outline this does not matter, since the threshold is chosen for each field.

The CSV has the same columns plus `x` and `y` (centroid of the polygon, in the coordinates of the orthomosaic), one row per plot sorted by field and plot number, ready to join to the field book by `crop_id` and `grid_id`.
