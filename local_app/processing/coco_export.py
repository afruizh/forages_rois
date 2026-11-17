"""Export reviewed plant boxes as a COCO dataset of RGB tiles, to annotate further or to retrain.

    export_coco(raster, rois_shp, output_dir, crop_shp=None)

The raster is read through the same 8 bit RGB view as the detection (rgb_bands: chosen bands,
stretched from minimum to maximum, optionally resampled to the model resolution), so the tiles
look like the training tiles. The layout is the one CVAT and Roboflow import ("COCO 1.0"):

    output_dir/annotations/instances_default.json
    output_dir/images/default/tile_00000.jpg ...

Rules:
  * Tiles of tile_size x tile_size pixels on a regular grid; the last row and column are moved
    inward so every tile is whole (smaller rasters are padded with black).
  * With crop fields, only tiles that touch a field are exported (crop_detection class 0, or all
    polygons of a file without a class field). Tiles without any box are left out.
  * One annotation per plant box (empty plots, status "empty", are left out), clipped to the tile;
    a box is kept only when at least min_visible of its area is inside the tile.
  * Categories from the class field: id = class + 1, names from result_summary.CLASS_NAMES.
  * Each image records its raster and pixel offset (extra keys), so tiles can be traced back.

Export only areas that were reviewed: a plant left without a box in a tile teaches a model that
it is background.
"""
import os
import json
import datetime
import numpy as np
from osgeo import gdal, ogr

gdal.DontUseExceptions()

TILE_SIZE = 1024
MIN_VISIBLE = 0.5      # fraction of a box that must be inside a tile to be annotated in it
JPEG_QUALITY = 95


def _tile_starts(length, tile_size, overlap):
    """Start offsets of whole tiles along one axis (the last one moved inward)."""
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _to_pixels(inv_gt, x, y):
    return inv_gt[0] + inv_gt[1] * x + inv_gt[2] * y, inv_gt[3] + inv_gt[4] * x + inv_gt[5] * y


def _read_boxes(rois_shp, inv_gt):
    """[(x0, y0, x1, y1, class)] in raster pixels of the plant boxes (empty plots left out)."""
    ds = ogr.Open(rois_shp)
    if ds is None:
        raise FileNotFoundError(f"Could not open the shapefile: {rois_shp}")
    layer = ds.GetLayer()
    defn = layer.GetLayerDefn()
    has_status, has_class = defn.GetFieldIndex("status") >= 0, defn.GetFieldIndex("class") >= 0
    boxes = []
    for f in layer:
        geom = f.GetGeometryRef()
        if geom is None or (has_status and f.GetField("status") == "empty"):
            continue
        cls = f.GetField("class") if has_class else 0
        if cls is None:
            continue
        minx, maxx, miny, maxy = geom.GetEnvelope()
        xa, ya = _to_pixels(inv_gt, minx, maxy)
        xb, yb = _to_pixels(inv_gt, maxx, miny)
        boxes.append((min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb), int(cls)))
    return boxes


def _read_fields(crop_shp, inv_gt):
    """Crop fields as one shapely geometry in raster pixels, or None."""
    if not crop_shp or not os.path.exists(crop_shp):
        return None
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    ds = ogr.Open(crop_shp)
    if ds is None:
        return None
    layer = ds.GetLayer()
    has_class = layer.GetLayerDefn().GetFieldIndex("class") >= 0
    polys = []
    for f in layer:
        geom = f.GetGeometryRef()
        if geom is None or (has_class and f.GetField("class") not in (0, None)):
            continue  # crop_detection class 1 (target) is not a field
        parts = [geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())] if geom.GetGeometryType() in (ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D) else [geom]
        for part in parts:
            ring = part.GetGeometryRef(0)
            pts = [_to_pixels(inv_gt, *ring.GetPoint_2D(i)) for i in range(ring.GetPointCount())]
            if len(pts) >= 3:
                polys.append(Polygon(pts).buffer(0))
    return unary_union(polys) if polys else None


def _read_tile(ds, x, y, size):
    """size x size x 3 uint8 RGB of the tile at (x, y), padded with black outside the raster."""
    w, h = min(size, ds.RasterXSize - x), min(size, ds.RasterYSize - y)
    data = ds.ReadAsArray(x, y, w, h)
    data = np.stack([data] * 3) if data.ndim == 2 else data[:3]
    if data.shape[0] < 3:
        data = np.concatenate([data] + [data[-1:]] * (3 - data.shape[0]))
    tile = np.zeros((size, size, 3), np.uint8)
    tile[:h, :w] = np.clip(data.transpose(1, 2, 0), 0, 255).astype(np.uint8)
    return tile


def export_coco(raster_path, rois_shp, output_dir, crop_shp=None, tile_size=TILE_SIZE, overlap=0,
                image_format="jpg", min_visible=MIN_VISIBLE, progress_callback=None, interruption_check=None):
    """Write the COCO dataset to output_dir and return the path of its annotations file."""
    import cv2
    from processing.rgb_bands import rgb_raster
    from utils.result_summary import CLASS_NAMES
    if overlap >= tile_size:
        raise ValueError("The tile overlap must be smaller than the tile size.")
    raster_path = rgb_raster(raster_path)  # the same path for 8 bit RGB rasters and RGB views
    ds = gdal.Open(raster_path)
    if ds is None:
        raise FileNotFoundError(f"Could not open raster: {raster_path}")
    gt = ds.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)
    boxes = _read_boxes(rois_shp, inv_gt)
    if not boxes:
        raise ValueError(f"No plant boxes in {rois_shp}.")
    fields = _read_fields(crop_shp, inv_gt)
    arr = np.array([b[:4] for b in boxes])

    images_dir = os.path.join(output_dir, "images", "default")
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    from shapely.geometry import box as shapely_box
    xs = _tile_starts(ds.RasterXSize, tile_size, overlap)
    ys = _tile_starts(ds.RasterYSize, tile_size, overlap)
    images, annotations, used_classes = [], [], set()
    total, done = len(xs) * len(ys), 0
    for y in ys:
        for x in xs:
            done += 1
            if interruption_check and interruption_check():
                raise InterruptedError("Export cancelled.")
            if progress_callback and done % 10 == 0:
                progress_callback({"progress": 100 * done / total, "status": f"Exporting tile {done} of {total}"})
            if fields is not None and not fields.intersects(shapely_box(x, y, x + tile_size, y + tile_size)):
                continue
            # Boxes touching the tile, clipped to it, kept when enough of them is inside
            near = np.flatnonzero((arr[:, 0] < x + tile_size) & (arr[:, 2] > x) & (arr[:, 1] < y + tile_size) & (arr[:, 3] > y))
            tile_annotations = []
            for i in near:
                x0, y0, x1, y1, cls = boxes[i]
                cx0, cy0 = max(x0, x), max(y0, y)
                cx1, cy1 = min(x1, x + tile_size), min(y1, y + tile_size)
                area, clipped = (x1 - x0) * (y1 - y0), (cx1 - cx0) * (cy1 - cy0)
                if area <= 0 or clipped / area < min_visible:
                    continue
                w, h = cx1 - cx0, cy1 - cy0
                tile_annotations.append({"category_id": cls + 1, "bbox": [round(cx0 - x, 2), round(cy0 - y, 2), round(w, 2), round(h, 2)],
                                         "area": round(w * h, 2), "segmentation": [], "iscrowd": 0})
                used_classes.add(cls)
            if not tile_annotations:
                continue
            image_id = len(images) + 1
            file_name = f"tile_{image_id - 1:05d}.{image_format}"
            tile = cv2.cvtColor(_read_tile(ds, x, y, tile_size), cv2.COLOR_RGB2BGR)
            params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if image_format.lower() in ("jpg", "jpeg") else []
            cv2.imwrite(os.path.join(images_dir, file_name), tile, params)
            images.append({"id": image_id, "file_name": file_name, "width": tile_size, "height": tile_size,
                           "source_raster": os.path.basename(raster_path), "x_offset": x, "y_offset": y})
            for a in tile_annotations:
                annotations.append({"id": len(annotations) + 1, "image_id": image_id, **a})

    names = dict(CLASS_NAMES)
    categories = [{"id": c + 1, "name": names.get(c, str(c)), "supercategory": "plant"}
                  for c in sorted(set(names) | used_classes)]
    dataset = {
        "info": {"description": "ForagesROIs plant boxes", "date_created": datetime.date.today().isoformat(),
                 "source_raster": raster_path, "source_shapefile": rois_shp, "tile_size": tile_size,
                 "overlap": overlap, "pixel_size": abs(gt[1])},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    path = os.path.join(annotations_dir, "instances_default.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2)
    print(f"COCO export: {len(images)} tiles, {len(annotations)} boxes: {path}")
    return path
