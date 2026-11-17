"""Short, machine readable summary of a task result (CLI --json, and for scripts or AI agents).

summarize(results) takes the dictionary returned by Processor.run() and adds counts read from the
output shapefile, so a caller does not have to parse the log or open the shapefile.
"""
import os
from collections import Counter

CLASS_NAMES = {0: "urochloa", 1: "megathyrsus"}   # plant model classes (crop model: 0 = crop)


def _shapefile_counts(path, task):
    from osgeo import ogr
    ds = ogr.Open(path)
    if ds is None:
        return {}
    layer = ds.GetLayer()
    defn = layer.GetLayerDefn()
    names = {defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())}
    status, classes, fields, numbered, left_out = Counter(), Counter(), set(), 0, 0
    for f in layer:
        st = f.GetField("status") if "status" in names else "plant"
        status[st] += 1
        if "class" in names and f.GetField("class") is not None and st != "empty":
            classes[f.GetField("class")] += 1
        if "crop_id" in names and f.GetField("crop_id") is not None:
            fields.add(f.GetField("crop_id"))
        if "grid_id" in names and st != "empty":
            if f.GetField("grid_id"):
                numbered += 1
            else:
                left_out += 1
    counts = {"features": layer.GetFeatureCount()}
    if task == "crop_detection":
        counts["fields"] = counts["features"]
        return counts
    counts["plants"] = status.get("plant", 0)
    counts["empty_plots"] = status.get("empty", 0)
    if "grid_id" in names:
        counts["numbered_plants"] = numbered
        counts["plants_left_out"] = left_out
    if fields:
        counts["fields"] = len(fields)
    if classes:
        counts["classes"] = {CLASS_NAMES.get(c, str(c)): n for c, n in sorted(classes.items())}
    return counts


def _coco_counts(path):
    """Tiles, boxes and boxes per category of a COCO annotations file (export_coco)."""
    import json
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    names = {c["id"]: c["name"] for c in data.get("categories", [])}
    per_class = Counter(names.get(a["category_id"], str(a["category_id"])) for a in data.get("annotations", []))
    return {"images": len(data.get("images", [])), "annotations": len(data.get("annotations", [])),
            "classes": dict(sorted(per_class.items()))}


def summarize(results):
    """Dictionary with status, task, message, output file(s), whether nothing was run because the
    output existed, and counts from the output shapefile."""
    results = results or {}
    output = results.get("output_file")
    summary = {
        "status": results.get("status", "error"),
        "task": results.get("task"),
        "message": results.get("message", ""),
        "output_file": output,
        "skipped": bool(results.get("skipped", False)),
    }
    # Model file and device used (detection tasks), to tell which model made a result
    for key in ("model", "device"):
        if results.get(key):
            summary[key] = results[key]
    if output and output.lower().endswith(".json") and os.path.exists(output):
        summary["counts"] = _coco_counts(output)  # export_coco
    if output and output.lower().endswith(".shp") and os.path.exists(output):
        summary["counts"] = _shapefile_counts(output, results.get("task"))
        csv = os.path.splitext(output)[0] + ".csv"
        if results.get("task") == "plant_traits" and os.path.exists(csv):
            summary["csv_file"] = csv
    return summary
