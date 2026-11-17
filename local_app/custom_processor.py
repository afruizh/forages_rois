import os
import sys
import onnxruntime as ort
import cv2 as cv
import numpy as np
import pandas as pd
import shapely.geometry
from shapely.geometry import Polygon

from osgeo import gdal, ogr, osr
# The code checks return values (None) instead of catching exceptions: say so explicitly, which
# also silences GDAL's FutureWarning about exceptions becoming the default in GDAL 4.0
gdal.DontUseExceptions()
ogr.DontUseExceptions()
osr.DontUseExceptions()
import datetime

from interface.batchprocessor import BatchProcessor
import glob

# Default models folder: "models" next to the script or the executable (not the working directory).
# The models folder and each model file can be changed in File > Preferences (see interface/preferences.py).
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "models")
DEFAULT_MODEL_PATH = MODEL_PATH  # kept when MODEL_PATH is changed, for Restore Defaults
#MODEL_PATH = r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\models"

MODEL_FILENAME = "forages_rois_yolo_full_1024.onnx"
CROP_MODEL_FILENAME = "crop_detection.onnx"

# Plant model (Ultralytics detect, output (1, 4 + classes, anchors) with class probabilities).
# Minimum class probability to keep a plant; low values recover plants the model is unsure about
# at the cost of some weeds. The IoU for NMS between boxes of any class in a tile.
PLANT_CONFIDENCE = 0.10
PLANT_NMS_IOU = 0.2

# Ground resolution (metres per pixel) of the orthomosaics each model was trained on. For plant
# detection, a raster finer than "finest" is resampled to "target" (in the RGB view, see rgb_bands),
# so plants have the size in pixels the model learned; coarser rasters are used as they are.
# The plant model was trained on 8 orthomosaics of 1.1 to 1.7 cm per pixel. Record it for new models.
MODEL_RESOLUTION = {
    MODEL_FILENAME: {"finest": 0.011, "target": 0.015},
}
# Models the app needs, with what each one is used for
REQUIRED_MODELS = {
    MODEL_FILENAME: "plant detection (tiling_detection, tiling_detection_on_crop)",
    CROP_MODEL_FILENAME: "crop field detection (crop_detection)",
}

# Models chosen in File > Preferences (or --plant_model, --crop_model for one run): full path of the
# file used instead of the default one in MODEL_PATH, or None for the default
CUSTOM_MODELS = {MODEL_FILENAME: None, CROP_MODEL_FILENAME: None}
# GPU (CUDA) when available, otherwise CPU; False runs on the CPU only (Preferences, --cpu)
USE_GPU = True


def model_path(model_filename):
    """File used for a model: the one chosen in Preferences, otherwise the default in MODEL_PATH."""
    return CUSTOM_MODELS.get(model_filename) or os.path.join(MODEL_PATH, model_filename)


def execution_providers():
    """onnxruntime providers: CUDA first when USE_GPU and onnxruntime was built with it; onnxruntime
    falls back to the CPU by itself when CUDA cannot start (driver, missing DLLs)."""
    if USE_GPU and "CUDAExecutionProvider" in ort.get_available_providers():
        return [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def check_model(path, model_filename):
    """Check that a model file can be used for model_filename (plant or crop model). Returns
    (level, text): "ok", "warning" (usable, but check the text) or "error" (cannot be used)."""
    import ast
    if not path or not os.path.isfile(path):
        return "error", f"File not found: {path}"
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as e:
        return "error", f"onnxruntime {ort.__version__} cannot load it: {e}"
    plant = model_filename == MODEL_FILENAME
    size = 1024 if plant else 640
    inp, outs = sess.get_inputs()[0], sess.get_outputs()
    meta = sess.get_modelmeta().custom_metadata_map
    try:
        names = ast.literal_eval(meta.get("names", "{}"))
    except (ValueError, SyntaxError):
        names = {}
    # Only what makes a model compliant: computer vision task, input size and classes
    task = meta.get("task") or ("segment" if len(outs) >= 2 else "detect")
    if len(inp.shape) == 4 and isinstance(inp.shape[2], int):
        model_size = inp.shape[2]
    else:  # dynamic input: the size it was exported at, if recorded
        try:
            model_size = int(ast.literal_eval(meta.get("imgsz", "[0]"))[0]) or None
        except (ValueError, SyntaxError, TypeError, IndexError):
            model_size = None
    text = [f"Task: {task} (required: {'detect' if plant else 'segment'})",
            f"Input size: {f'{model_size} px' if model_size else 'any'} (required: {size} px)",
            "Classes: " + (", ".join(f"{k} {v}" for k, v in sorted(names.items())) if names else "not recorded")]
    if inp.name != "images" or len(inp.shape) != 4:
        return "error", "\n".join(text + ["The input must be an image (images, 3 channels)."])
    if not plant and len(outs) < 2:
        return "error", "\n".join(text + ["The crop field model must be a segmentation model."])
    warnings = []
    if plant and task != "detect":
        warnings.append("The plant model should be a detection model.")
    if model_size and model_size != size:
        warnings.append(f"The input size differs: the app sends {size} px.")
    expected = PLANT_CLASS_NAMES if plant else CROP_CLASS_NAMES
    if names and len(names) != len(expected):
        warnings.append("The classes differ: required " + ", ".join(f"{k} {v}" for k, v in expected.items()) + ".")
    elif names and any(str(names.get(k, "")).lower() != v for k, v in expected.items()):
        warnings.append("The class names differ: required " + ", ".join(f"{k} {v}" for k, v in expected.items())
                        + ", in this order.")
    return ("warning" if warnings else "ok"), "\n".join(text + warnings)


def missing_models(folder):
    """Return the required model files that are not in folder (models chosen in Preferences
    elsewhere do not count as missing)."""
    return [name for name in REQUIRED_MODELS
            if not CUSTOM_MODELS.get(name) and not os.path.isfile(os.path.join(folder, name))]

# The same models on Hugging Face (AGPL-3.0), fixed to the version of this release by its tag,
# with the SHA256 of each file so that a broken or different download is never used
MODELS_URL = "https://huggingface.co/anfruizhu/forages-rois-models/resolve/v1.0.0/"
MODEL_SHA256 = {
    MODEL_FILENAME: "2fa231d5d1530a17096a14a7375d9cc6f9fb5e9457fb0a2e7ce0aa8d8bafdda6",
    CROP_MODEL_FILENAME: "ad01824357615346c2b3946abdb6e3c7fe90f2656caa9c44bf691f7f3ee3fb70",
}

def download_models(folder, names=None, progress_callback=None, interruption_check=None):
    """Download models (by default the missing ones) from MODELS_URL into folder. Each file is
    written as .part, checked against MODEL_SHA256 and only then renamed. Returns the names
    downloaded; raises an exception with a readable message on failure."""
    import hashlib
    import urllib.request
    os.makedirs(folder, exist_ok=True)
    names = missing_models(folder) if names is None else names
    for i, name in enumerate(names):
        part = os.path.join(folder, name + ".part")
        sha = hashlib.sha256()
        try:
            with urllib.request.urlopen(MODELS_URL + name, timeout=60) as response, open(part, "wb") as f:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    if interruption_check and interruption_check():
                        raise InterruptedError("Download cancelled.")
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    sha.update(chunk)
                    done += len(chunk)
                    if progress_callback and total:
                        progress_callback({"progress": int(100 * (i + done / total) / len(names)),
                                           "status": f"Downloading {name}: {done >> 20} of {total >> 20} MB"})
        except InterruptedError:
            os.remove(part)
            raise
        except OSError as e:
            if os.path.exists(part):
                os.remove(part)
            raise OSError(f"Could not download {name} from\n{MODELS_URL + name}\n\n{e}\n\n"
                          "Check the internet connection, or download the models by hand and select "
                          "their folder in File > Preferences.") from e
        if sha.hexdigest() != MODEL_SHA256[name]:
            os.remove(part)
            raise ValueError(f"The downloaded {name} does not match the expected checksum and was deleted. Try again.")
        os.replace(part, os.path.join(folder, name))
        print(f"Downloaded {name} to {folder}")
    return names

# Preprocess image
SHAPEFILE_PARTS = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qml")


def replace_shapefile(src_shp, dst_shp):
    """Move the files of shapefile src_shp (.shp, .shx, .dbf, .prj, .cpg, .qml) over those of
    dst_shp. Parts that src_shp does not have (for example a .qml) are kept from dst_shp."""
    src_base, dst_base = os.path.splitext(src_shp)[0], os.path.splitext(dst_shp)[0]
    for ext in SHAPEFILE_PARTS:
        if os.path.exists(src_base + ext):
            os.replace(src_base + ext, dst_base + ext)


def model_stamp(model_filename=MODEL_FILENAME):
    """Name, size and modification time of a model file, for cache keys: replacing the model
    in the models folder then gives new detections instead of the cached ones."""
    path = model_path(model_filename)
    if not os.path.exists(path):
        return path
    st = os.stat(path)
    return f"{path}:{st.st_size}:{int(st.st_mtime)}"


def preprocess(np_img, imgsz=1024, pad_color=114, debug_path=None):

    img = np_img
    
    h0, w0 = img.shape[:2]
    r = imgsz / max(h0, w0)
    new_size = (int(w0 * r), int(h0 * r))
    resized = cv.resize(img, new_size, interpolation=cv.INTER_LINEAR)

    # Padding
    padded = np.full((imgsz, imgsz, 3), pad_color, dtype=np.uint8)
    
    pad_w = (imgsz - new_size[0]) // 2
    pad_h = (imgsz - new_size[1]) // 2
    padded[pad_h:pad_h+new_size[1], pad_w:pad_w+new_size[0]] = resized

    if debug_path:
        cv.imwrite(debug_path, cv.cvtColor(padded, cv.COLOR_RGB2BGR))

    img_input = padded.astype(np.float32) / 255.0
    img_input = img_input.transpose(2, 0, 1)  # HWC to CHW
    img_input = np.expand_dims(img_input, axis=0)  # Add batch dim
    return img_input, r, (h0, w0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(xywh):
    xy = xywh[:, :2]
    wh = xywh[:, 2:]
    top_left = xy - wh / 2
    bottom_right = xy + wh / 2
    return np.concatenate([top_left, bottom_right], axis=1)

def postprocess_yolo_output(outputs, conf_threshold=0.3, nms_threshold=0.5, input_size=1024, orig_shape=(1024, 1024)):
    """
    Convert raw YOLO ONNX output (1, 5+C, N) to bboxes and class IDs using sigmoid + NMS.

    Returns:
        bboxes (np.ndarray): Bounding boxes (N, 5) in xyxy format with scores.
        classes (np.ndarray): Class IDs (N,).
    """
    output = outputs[0]  # (1, 5+C, N)
    output = np.squeeze(output, axis=0)  # (5+C, N)
    output = output.transpose(1, 0)      # (N, 5+C)

    boxes_xywh = output[:, :4]
    objectness = sigmoid(output[:, 4])
    class_scores = sigmoid(output[:, 5:])  # shape (N, num_classes)

    # Final confidence = objectness * class_score per class
    scores = objectness[:, None] * class_scores  # shape (N, num_classes)
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # Filter by confidence
    mask = confidences > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    if len(boxes_xywh) == 0:
        return [[np.zeros((0, 5))], [np.zeros((0,), dtype=np.int32)]]

    # Convert to xyxy and scale
    boxes_xyxy = xywh2xyxy(boxes_xywh)

    # Undo letterbox scaling
    gain = input_size / max(orig_shape)
    pad_x = (input_size - orig_shape[1] * gain) / 2
    pad_y = (input_size - orig_shape[0] * gain) / 2
    boxes_xyxy[:, [0, 2]] -= pad_x
    boxes_xyxy[:, [1, 3]] -= pad_y
    boxes_xyxy /= gain
    boxes_xyxy = np.clip(boxes_xyxy, 0, [orig_shape[1], orig_shape[0], orig_shape[1], orig_shape[0]])

    # Prepare final lists
    final_boxes = []
    final_scores = []
    final_classes = []

    # NMS per class
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = confidences[cls_mask]

        boxes_nms = cls_boxes.copy()
        boxes_nms[:, 2] -= boxes_nms[:, 0]
        boxes_nms[:, 3] -= boxes_nms[:, 1]

        indices = cv.dnn.NMSBoxes(
            bboxes=boxes_nms.tolist(),
            scores=cls_scores.tolist(),
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold
        )

        if len(indices) > 0:
            indices = np.array(indices).flatten()
            final_boxes.append(cls_boxes[indices])
            final_scores.append(cls_scores[indices])
            final_classes.append(np.full(len(indices), cls, dtype=np.int32))

    if not final_boxes:
        return [[np.zeros((0, 5))], [np.zeros((0,), dtype=np.int32)]]

    final_boxes = np.concatenate(final_boxes, axis=0)
    final_scores = np.concatenate(final_scores, axis=0)
    final_classes = np.concatenate(final_classes, axis=0)

    final_boxes = np.concatenate([final_boxes, final_scores[:, None]], axis=1)  # (N, 5)

    return [[final_boxes], [final_classes]]

def postprocess_yolov8_output(outputs, conf_threshold=0.3, nms_threshold=0.5, input_size=1024, orig_shape=(1024, 1024), agnostic=False):
    """
    Convert raw YOLOv8/v11 ONNX output (1, 4+C, N) or (1, N, 4+C) to bboxes and class IDs using NMS.
    agnostic=True runs NMS across classes, so one object never gets a box per class.
    """
    output = outputs[0]
    output = np.squeeze(output, axis=0)

    # YOLOv8/11 exports can be (4+C, N) or (N, 4+C). We want (N, 4+C).
    if output.shape[0] < output.shape[1]:
        output = output.transpose(1, 0)

    # Check if this is an End2End ONNX export with NMS already applied: (N, 6) with at most a few
    # hundred rows (max detections). A raw 2 class model is also (N, 6) but with thousands of anchors.
    is_end2end = output.shape[1] == 6 and output.shape[0] <= 1000
    
    if is_end2end:
        boxes_xyxy = output[:, :4]
        confidences = output[:, 4]
        class_ids = output[:, 5].astype(np.int32)
        
        mask = confidences > conf_threshold
        final_boxes = boxes_xyxy[mask]
        final_classes = class_ids[mask]
        final_scores = confidences[mask]
        
        if len(final_boxes) == 0:
            return [[np.zeros((0, 5))], [np.zeros((0,), dtype=np.int32)]]
            
        # Undo letterbox scaling
        gain = input_size / max(orig_shape)
        pad_x = (input_size - orig_shape[1] * gain) / 2
        pad_y = (input_size - orig_shape[0] * gain) / 2
        final_boxes[:, [0, 2]] -= pad_x
        final_boxes[:, [1, 3]] -= pad_y
        final_boxes /= gain
        final_boxes = np.clip(final_boxes, 0, [orig_shape[1], orig_shape[0], orig_shape[1], orig_shape[0]])
        
        final_boxes = np.concatenate([final_boxes, final_scores[:, None]], axis=1)  # (N, 5)
        return [[final_boxes], [final_classes]]

    boxes_xywh = output[:, :4]
    scores = output[:, 4:]  # shape (N, num_classes)
    
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # Filter by confidence
    mask = confidences > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    if len(boxes_xywh) == 0:
        return [[np.zeros((0, 5))], [np.zeros((0,), dtype=np.int32)]]

    # Convert to xyxy and scale
    boxes_xyxy = xywh2xyxy(boxes_xywh)

    # Undo letterbox scaling
    gain = input_size / max(orig_shape)
    pad_x = (input_size - orig_shape[1] * gain) / 2
    pad_y = (input_size - orig_shape[0] * gain) / 2
    boxes_xyxy[:, [0, 2]] -= pad_x
    boxes_xyxy[:, [1, 3]] -= pad_y
    boxes_xyxy /= gain
    boxes_xyxy = np.clip(boxes_xyxy, 0, [orig_shape[1], orig_shape[0], orig_shape[1], orig_shape[0]])

    # Prepare final lists
    final_boxes_list = []
    final_scores_list = []
    final_classes_list = []

    # NMS per class, or over all boxes together when agnostic
    nms_groups = np.zeros_like(class_ids) if agnostic else class_ids
    for group in np.unique(nms_groups):
        cls_mask = nms_groups == group
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = confidences[cls_mask]

        boxes_nms = cls_boxes.copy()
        boxes_nms[:, 2] -= boxes_nms[:, 0]
        boxes_nms[:, 3] -= boxes_nms[:, 1]

        indices = cv.dnn.NMSBoxes(
            bboxes=boxes_nms.tolist(),
            scores=cls_scores.tolist(),
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold
        )

        if len(indices) > 0:
            indices = np.array(indices).flatten()
            final_boxes_list.append(cls_boxes[indices])
            final_scores_list.append(cls_scores[indices])
            final_classes_list.append(class_ids[cls_mask][indices].astype(np.int32))

    if not final_boxes_list:
        return [[np.zeros((0, 5))], [np.zeros((0,), dtype=np.int32)]]

    final_boxes = np.concatenate(final_boxes_list, axis=0)
    final_scores = np.concatenate(final_scores_list, axis=0)
    final_classes = np.concatenate(final_classes_list, axis=0)

    final_boxes = np.concatenate([final_boxes, final_scores[:, None]], axis=1)  # (N, 5)

    return [[final_boxes], [final_classes]]

def outputs_to_df(outputs):
    """
    Convert bounding boxes from the padded & resized image back to the original image coordinates.
    """
    bboxes = outputs[0][0]
    classes = outputs[1][0]
    
    has_polygons = len(outputs) >= 3
    polygons = outputs[2][0] if has_polygons else None

    results = []
    for i, (bbox, label) in enumerate(zip(bboxes, classes)):
        x1, y1, x2, y2, score = bbox

        row = {
            "xmin": x1,
            "ymin": y1,
            "xmax": x2,
            "ymax": y2,
            "score": score,
            "class": label
        }
        
        if has_polygons and polygons[i] is not None:
            row["polygon"] = polygons[i]
            
        results.append(row)
        
    df = pd.DataFrame(results)
    return df

# Class names written next to the class number (field class_name, 10 characters, the shapefile limit)
PLANT_CLASS_NAMES = {0: "urochloa", 1: "megathyrsus"}
CROP_CLASS_NAMES = {0: "crop", 1: "target"}

def add_class_names(df, names):
    """Add a class_name column from the class number; an unknown number keeps the number as text."""
    if df is not None and not df.empty and "class" in df:
        df["class_name"] = [names.get(int(c), str(int(c))) for c in df["class"]]
    return df

def pos2coords(pos, extent, img_width, img_height):
    left, bottom, right, top = extent
    extent_width = right - left
    extent_height = top - bottom
    x = (pos[0]) / img_width
    y = 1.0 - (pos[1]) / img_height
    coord_x = x * extent_width + left
    coord_y = y * extent_height + bottom
    return (coord_x, coord_y)

def create_shapefile_ogr(output_filename, geometries, projection, geom_type="polygon", allow_cols=None, data_df=None):
    from osgeo import ogr, osr
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_filename):
        driver.DeleteDataSource(output_filename)
    data_source = driver.CreateDataSource(output_filename)
    
    srs = None
    if projection:
        srs = osr.SpatialReference()
        try:
            srs.ImportFromWkt(projection)
        except Exception:
            srs = None
    ogr_geom_type = ogr.wkbPolygon
    layer = data_source.CreateLayer("features", srs, ogr_geom_type)
    
    layer.CreateField(ogr.FieldDefn("ID", ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn("Type", ogr.OFTString))
    
    if allow_cols and data_df is not None:
        for col in allow_cols:
            if col == "score":
                layer.CreateField(ogr.FieldDefn("score", ogr.OFTReal))
            elif col == "class":
                layer.CreateField(ogr.FieldDefn("class", ogr.OFTInteger))
            else:
                layer.CreateField(ogr.FieldDefn(col, ogr.OFTString))
    
    import numpy as np
    
    for idx, (geom_data, (_, row)) in enumerate(zip(geometries, data_df.iterrows() if data_df is not None else [])):
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("ID", idx+1)
        feature.SetField("Type", "forage_plant")
        
        if allow_cols and data_df is not None:
            for col in allow_cols:
                feature.SetField(col, row[col])
        
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for coord in geom_data:
            ring.AddPoint(coord[0], coord[1])
        ring.CloseRings()
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        feature.SetGeometry(polygon)
        
        layer.CreateFeature(feature)
        feature = None
        
    data_source = None

def save_shapefile_bb(df, extent, img_width, img_height, epsg_or_proj, allow_cols=[], output_filename=None):
    if df is None or df.empty:
        print("No results")
        return

    coord_polygons = []
    valid_df = []
    
    for index, detection in df.iterrows():
        if "polygon" in detection and detection["polygon"] is not None:
            polygon_pts = detection["polygon"]
            coord_polygon = []
            for pt in polygon_pts:
                coord = (pt[0], pt[1])
                new_coord = pos2coords(coord, extent, img_width, img_height)
                coord_polygon.append(new_coord)
            if len(coord_polygon) > 2:
                # Explicitly close the polygon to prevent it from looking 'clipped' or broken in GIS software
                if coord_polygon[0] != coord_polygon[-1]:
                    coord_polygon.append(coord_polygon[0])
                coord_polygons.append(coord_polygon)
                valid_df.append(detection)
        else:
            xmin = detection["xmin"]
            ymin = detection["ymin"]
            xmax = detection["xmax"]
            ymax = detection["ymax"]
            new_contour = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
            coord_polygon = []
            for point in new_contour:
                coord = (point[0], point[1])
                new_coord = pos2coords(coord, extent, img_width, img_height)
                coord_polygon.append(new_coord)
            
            if len(coord_polygon) > 2:
                if coord_polygon[0] != coord_polygon[-1]:
                    coord_polygon.append(coord_polygon[0])
                coord_polygons.append(coord_polygon)
                valid_df.append(detection)
            
    if output_filename is not None and coord_polygons:
        valid_df_obj = pd.DataFrame(valid_df)
        create_shapefile_ogr(output_filename, coord_polygons, epsg_or_proj, "polygon", allow_cols, valid_df_obj)

def check_raster(input_file):
    metadata = {}
    ds = gdal.Open(input_file)
    metadata["width"] = ds.RasterXSize
    metadata["height"] = ds.RasterYSize
    ds = None
    return metadata
    
## POSTPROCESSING FUNCTIONS
import numpy as np
from shapely.geometry import box
from shapely.geometry import Polygon


def compute_centroids(gdf):
    return np.stack([geom.centroid.coords[0] for geom in gdf.geometry])

# def compute_pca_axes(points):
#     pca = PCA(n_components=2)
#     pca.fit(points)
#     return pca.components_

# def project_to_grid_axes(points, axes):
#     return np.dot(points, axes.T)

# def compute_pca_axes(points):
#     """
#     Returns PCA axes as 2x2 matrix. Each row is a unit vector.
#     The first row is the direction of maximum variance (horizontal),
#     the second is orthogonal (vertical).
#     """
#     pca = PCA(n_components=2)
#     pca.fit(points)
#     axes = pca.components_
#     # Optionally, flip axes to ensure consistent orientation
#     # For example, force first axis to point right, second to point down
#     if axes[0, 0] < 0:
#         axes[0] *= -1
#     if axes[1, 1] < 0:
#         axes[1] *= -1
#     return axes

def remove_outlier_centroids(centroids, threshold=1.0):
    """
    Remove outlier centroids based on z-score threshold (no scipy).
    """
    mean = np.mean(centroids, axis=0)
    std = np.std(centroids, axis=0)
    z = np.abs((centroids - mean) / std)
    mask = (z < threshold).all(axis=1)
    return centroids[mask]

def estimate_grid_angle(centroids, bin_size=1.0):
    """
    Estimate the main grid orientation angle (in degrees) from centroids using pairwise angle histogramming.
    This method computes the angle between all pairs of centroids, builds a histogram, and selects the dominant angle.
    Args:
        centroids: np.ndarray of shape (N, 2)
        bin_size: bin size in degrees for the histogram
    Returns:
        Dominant grid angle in degrees (float)
    """
    import numpy as np
    from collections import Counter

    pts = np.array(centroids)
    N = len(pts)
    if N < 2:
        return 0.0

    # Compute all pairwise angles
    angles = []
    for i in range(N):
        for j in range(i+1, N):
            dx = pts[j, 0] - pts[i, 0]
            dy = pts[j, 1] - pts[i, 1]
            if dx == 0 and dy == 0:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            # Normalize to [-90, 90)
            angle = ((angle + 90) % 180) - 90
            angles.append(angle)

    if not angles:
        return 0.0

    # Histogram the angles
    bins = np.arange(-90, 90 + bin_size, bin_size)
    hist, bin_edges = np.histogram(angles, bins=bins)
    max_bin = np.argmax(hist)
    dominant_angle = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
    # Invert the angle to match the original coordinate system
    dominant_angle = (dominant_angle + 90) % 180
    print(f"Estimated grid angle (pairwise histogram): {dominant_angle:.2f}°")
    return dominant_angle

def compute_pca_axes(points):
    """
    Returns PCA axes as 2x2 matrix. Each row is a unit vector.
    The first row is the direction of maximum variance (usually horizontal),
    the second is orthogonal.
    """
    # PCA axes from the SVD of the centred points (same as sklearn PCA components_)
    _, _, axes = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)

    # Optionally, flip axes for consistent orientation
    # Make first axis point right (positive x direction)
    if axes[0, 0] < 0:
        axes[0] *= -1
    # Make second axis point up (positive y direction)
    if axes[1, 1] < 0:
        axes[1] *= -1

    angle = np.arctan2(axes[0, 0], axes[0, 1]) * 180 / np.pi
    print(f"Rotation angle from x-axis to first PCA axis: {angle:.2f}°")
    return axes, angle

def project_to_grid_axes(points, axes):
    """
    Projects points onto the PCA axes.
    """
    # Center points before projecting (optional, but usually desired)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    projected = np.dot(centered, axes.T)
    return projected

def project_to_grid_axes_angle(points, angle, center=None):
    """
    Projects points onto axes defined by a rotation angle (in degrees).
    The angle should be the rotation from the x-axis to the desired axis.
    """
    # Center points before projecting (optional, but usually desired)
    if center is None:
        center = np.mean(points, axis=0)
    centered = points - center

    # Build rotation matrix from angle (convert to radians)
    theta = np.deg2rad(angle)
    R = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    projected = np.dot(centered, R.T)
    projected = projected + center  # Translate back to original position
    return projected

def group_rows_cols(projected_points, row_tol=1.0):
    """
    Groups points into rows by repeatedly picking the topmost, leftmost point,
    then grouping all points within a small y-tolerance as a row.
    The rows are ordered from topmost (highest y) to bottom, and within each row from leftmost (lowest x) to right.
    """
    pts = projected_points.copy()
    used = np.zeros(len(pts), dtype=bool)
    rows = []
    while not np.all(used):
        unused_idx = np.where(~used)[0]
        unused_pts = pts[unused_idx]
        max_y = np.max(unused_pts[:, 1])  # Find the topmost y
        row_mask = np.abs(unused_pts[:, 1] - max_y) < row_tol
        row_indices = unused_idx[row_mask]
        row_pts = pts[row_indices]
        # Sort row left to right (increasing x)
        row_pts = row_pts[np.argsort(row_pts[:, 0])]
        rows.append(list(row_pts))
        used[row_indices] = True
    return rows

def assign_indices(rows, serpentine=False):
    index_map = {}
    idx = 1
    for r, row in enumerate(rows):
        if serpentine and not r % 2 == 0:
            row = list(reversed(row))
        for pt in row:
            index_map[tuple(pt)] = idx
            idx += 1
    return index_map


# --- Plot numbering ---
# assign_grid_ids() numbers plant centroids in sequential steps:
#   1. Convert centroids to metres (local UTM zone) when the layer CRS is geographic,
#      so every distance below is in metres whatever the input CRS.
#   2. Find the row direction from the angles between each plant and its nearest neighbours.
#   3. Measure the spacing between plants in a row and between rows from the same neighbours.
#   4. Rotate the centroids so rows run along the u axis and are stacked along the v axis.
#   5. Trace rows: cut the field into short strips along u, split each strip into row pieces
#      wherever the gap along v is larger than half the row spacing, and link the pieces
#      strip by strip. Then refine the angle from the slope of those rows and trace again.
#      Last, move each plant to the nearest row line (refine_rows), which separates rows that
#      the tracing mixed where neighbouring rows come close.
#   6. Column: position of each plant in its row, counted from the start corner side.
#   7. Number row by row from the start corner, reversing every other row for serpentine.
# With empty plots (the default), steps 6 and 7 work on planting positions instead of plants
# (planting_grid): the column comes from the distance along the row in position spacings, a
# missing plant leaves an empty plot with its own number, and a missing row keeps its row
# number, so the numbers match the field book whatever the mortality.
# Rows with very few plants (isolated detections) are left out of the numbering (ids 0),
# so they do not shift the serpentine order of the real rows.

def metres_transform(points, srs):
    """Step 1: functions that convert (N, 2) arrays from the layer CRS to metres and back;
    geographic coordinates go to the local UTM zone."""
    if srs is None or not srs.IsGeographic():
        same = lambda p: np.array(p, dtype=float)
        return same, same
    lon, lat = np.asarray(points, dtype=float).mean(axis=0)
    zone = int((lon + 180) // 6) + 1
    utm = osr.SpatialReference()
    utm.ImportFromEPSG((32600 if lat >= 0 else 32700) + zone)
    src = srs.Clone()
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    utm.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    forward = osr.CoordinateTransformation(src, utm)
    inverse = osr.CoordinateTransformation(utm, src)
    to_metres = lambda p: np.array(forward.TransformPoints(np.asarray(p, dtype=float).tolist()))[:, :2]
    from_metres = lambda p: np.array(inverse.TransformPoints(np.asarray(p, dtype=float).tolist()))[:, :2]
    return to_metres, from_metres

def centroids_to_metres(points, srs):
    """Step 1: return the points in metres; geographic coordinates go to the local UTM zone."""
    return metres_transform(points, srs)[0](points)

def nearest_neighbours(points, k):
    """Indices of the k nearest neighbours of each point, computed in chunks to limit memory."""
    n = len(points)
    k = min(k, n - 1)
    idx = np.empty((n, k), dtype=int)
    for start in range(0, n, 1024):
        stop = min(start + 1024, n)
        d = ((points[start:stop, None, :] - points[None, :, :]) ** 2).sum(-1)
        d[np.arange(stop - start), np.arange(start, stop)] = np.inf
        idx[start:stop] = np.argpartition(d, k - 1, axis=1)[:, :k]
    return idx

def estimate_row_direction(points, neighbours, closest=2):
    """Step 2: dominant angle (degrees, 0 to 180) of the links between each plant and its
    `closest` nearest neighbours. Only the closest links are used: they lie along the row,
    while farther neighbours add the across-row and diagonal directions."""
    vec = points[neighbours] - points[:, None, :]
    dist = np.hypot(vec[..., 0], vec[..., 1])
    nearest = np.argsort(dist, axis=1)[:, :closest]
    vec = np.take_along_axis(vec, nearest[..., None], axis=1).reshape(-1, 2)
    ang = np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) % 180
    hist, _ = np.histogram(ang, bins=180, range=(0, 180))
    hist = hist + np.roll(hist, 1) + np.roll(hist, -1)  # circular smoothing
    peak = np.argmax(hist) + 0.5
    # Refine the peak with the mean of the angles within 10 degrees of it (doubled angles, since 0 = 180)
    diff = (ang - peak + 90) % 180 - 90
    sel = np.radians(2 * diff[np.abs(diff) < 10])
    return peak + np.degrees(np.arctan2(np.sin(sel).mean(), np.cos(sel).mean())) / 2

def link_spacing(points, neighbours, direction, max_angle=20):
    """Step 3: median distance to the closest neighbour lying along a direction (unit vector).
    Returns None when too few plants have a neighbour in that direction."""
    vec = points[neighbours] - points[:, None, :]
    length = np.hypot(vec[..., 0], vec[..., 1])
    cos = np.abs(vec @ direction) / np.maximum(length, 1e-12)
    length = np.where(cos >= np.cos(np.radians(max_angle)), length, np.inf)
    closest = length.min(axis=1)
    closest = closest[np.isfinite(closest)]
    return float(np.median(closest)) if len(closest) >= 10 else None

def split_by_gaps(values, min_gap):
    """Steps 5 and 6: group 1D values, starting a new group wherever the gap exceeds min_gap.
    Groups are numbered from 0 in increasing order of value."""
    order = np.argsort(values)
    breaks = np.diff(values[order]) > min_gap
    groups = np.empty(len(values), dtype=int)
    groups[order] = np.concatenate([[0], np.cumsum(breaks)])
    return groups

def trace_rows(u, v, plant_spacing, row_spacing, strip_plants=6):
    """Step 5: assign each plant to a row.
    Splitting the whole field at once fails on long rows: small drift or curvature makes
    neighbouring rows touch somewhere along their length. Instead the field is cut into
    strips across the rows (about `strip_plants` plants long), rows are split by gaps inside
    each strip, and the pieces are linked strip by strip to the row with the closest v.
    Returns row labels numbered from 0 in increasing order of v."""
    max_dv = 0.5 * row_spacing
    strip = np.floor((u - u.min()) / (strip_plants * plant_spacing)).astype(int)
    labels = np.full(len(u), -1)
    active_v = []  # last known v of each row traced so far
    for s in np.unique(strip):
        idx = np.flatnonzero(strip == s)
        pieces = split_by_gaps(v[idx], max_dv)
        piece_v = np.array([v[idx[pieces == p]].mean() for p in range(pieces.max() + 1)])
        # Link pieces to rows one to one, closest pairs first; unlinked pieces start new rows
        pairs = sorted((abs(pv - rv), p, r) for p, pv in enumerate(piece_v)
                       for r, rv in enumerate(active_v) if abs(pv - rv) < max_dv)
        piece_row, used_rows = {}, set()
        for _, p, r in pairs:
            if p not in piece_row and r not in used_rows:
                piece_row[p] = r
                used_rows.add(r)
        for p, pv in enumerate(piece_v):
            if p not in piece_row:
                piece_row[p] = len(active_v)
                active_v.append(pv)
            active_v[piece_row[p]] = pv
            labels[idx[pieces == p]] = piece_row[p]
    # Renumber rows by mean v so that row 0 is the southmost
    mean_v = np.array([v[labels == r].mean() for r in range(len(active_v))])
    rank = np.empty(len(mean_v), dtype=int)
    rank[np.argsort(mean_v)] = np.arange(len(mean_v))
    return rank[labels]

def refine_rows(u, v, rows, iterations=5):
    """Step 5b: fit a line to each traced row (a gentle curve for long rows) and move each plant
    to the nearest row line. Where neighbouring rows come close, trace_rows can link a stretch of
    one row to its neighbour, so a row holds plants of two real rows; their line passes between
    both, and the plants move to the lines of the real rows. Returns labels from 0 in v order."""
    for _ in range(iterations):
        labels = np.unique(rows)
        fits = []
        for r in labels:
            m = rows == r
            n = m.sum()
            deg = 2 if n >= 15 else 1 if n >= 5 else 0
            fit = np.polyfit(u[m], v[m], deg)
            if deg > 0:
                # Refit without plants far from the line (plants of another row)
                step = np.median(np.abs(np.diff(np.sort([v[rows == k].mean() for k in labels])))) if len(labels) > 1 else np.inf
                near = np.abs(v[m] - np.polyval(fit, u[m])) < 0.3 * step
                if near.sum() > deg + 2:
                    fit = np.polyfit(u[m][near], v[m][near], deg)
            fits.append(fit)
        # Distance to each row line, only within the u range of that row (plus a little), so the
        # line of a short row does not take plants far beyond its ends
        dist = np.full((len(u), len(labels)), np.inf)
        for j, (r, fit) in enumerate(zip(labels, fits)):
            ur = u[rows == r]
            span = max(np.ptp(ur), 1e-9)
            inside = (u >= ur.min() - 0.25 * span - 1) & (u <= ur.max() + 0.25 * span + 1)
            dist[inside, j] = np.abs(v[inside] - np.polyval(fit, u[inside]))
        new_rows = labels[np.argmin(dist, axis=1)]
        if (new_rows == rows).all():
            break
        rows = new_rows
    mean_v = {r: v[rows == r].mean() for r in np.unique(rows)}
    order = sorted(mean_v, key=mean_v.get)
    rank = {r: i for i, r in enumerate(order)}
    return np.array([rank[r] for r in rows])

def position_spacing(gaps, guess):
    """Step 7: distance between planting positions along the rows. Gaps between consecutive
    plants are one spacing, or two or more where plants are missing, so the spacing is the unit
    that fits the gaps best (not their median, which grows with mortality)."""
    gaps = gaps[gaps > 0]
    if len(gaps) < 5:
        return guess
    s = np.percentile(gaps[gaps > 0.5 * np.median(gaps)], 25)
    for _ in range(5):
        k = np.round(gaps / s)
        fit = (k >= 1) & (np.abs(gaps / s - k) < 0.3)
        if not fit.any():
            break
        s = gaps[fit].sum() / k[fit].sum()
    return float(s)

def edge_line(v, u, spacing, late):
    """Step 7: line u = a + b v through the first plants of the rows (late = 1) or the last ones
    (late = -1). The field edge is often slanted, so a line fits better than one value. Rows that
    start late or end early (missing plants at the row ends) are dropped from the fit."""
    if len(v) < 3:
        return np.array([0.0, u.min() if late > 0 else u.max()])
    keep = np.ones(len(v), dtype=bool)
    for _ in range(5):
        coef = np.polyfit(v[keep], u[keep], 1)
        new_keep = late * (u - np.polyval(coef, v)) < 0.5 * spacing
        if new_keep.sum() < 3 or (new_keep == keep).all():
            break
        keep = new_keep
    return coef

def planting_grid(u, v, rows, valid_rows, plant_spacing, inside=None, max_edge_empty=2):
    """Step 7: planting positions of the rows, so that a missing plant leaves an empty plot
    instead of shifting the numbers after it.
    - Columns: the first plant of a row from its distance to the start line, then each plant from
      the gap to the previous one in spacings (local, so small errors do not add up along the row).
    - Empty plots between plants of a row are always added. Before the first and after the last
      plant (up to the start and end lines) they are added only where inside(u, v) is true (crop
      field polygon), or, without a polygon, when there are at most max_edge_empty of them (so the
      short rows of an L shaped field do not get plots outside the field).
    - A gap between rows of about twice the row spacing or more is a row without plants; its
      positions are those shared by the rows on each side (and inside the polygon, if given).
    Returns the row slot (0 based in v order, -1 for plants left out) and column of each plant,
    the empty plots as (slot, col, u, v), the number of slots and the position spacing."""
    members = [np.flatnonzero(rows == r) for r in valid_rows]
    members = [m[np.argsort(u[m])] for m in members]
    gaps = np.concatenate([np.diff(u[m]) for m in members])
    spacing = position_spacing(gaps, plant_spacing)
    row_v = np.array([v[m].mean() for m in members])
    start = edge_line(row_v, np.array([u[m[0]] for m in members]), spacing, 1)
    end = edge_line(row_v, np.array([u[m[-1]] for m in members]), spacing, -1)

    plant_col = np.zeros(len(u), dtype=int)
    for m in members:
        c = 1 + max(0, int(round((u[m[0]] - np.polyval(start, v[m[0]])) / spacing)))
        plant_col[m[0]] = c
        for a, b in zip(m[:-1], m[1:]):
            c += int(round((u[b] - u[a]) / spacing))  # 0: two detections in one position
            plant_col[b] = c

    def row_at(m, cols):
        """u and v of positions cols of a row: between its plants, or one spacing per position
        beyond its ends."""
        pc, first = np.unique(plant_col[m], return_index=True)
        pu, pv = u[m][first], v[m][first]
        cols = np.asarray(cols, dtype=float)
        uu = np.interp(cols, pc, pu) + (np.minimum(cols - pc[0], 0) + np.maximum(cols - pc[-1], 0)) * spacing
        return uu, np.interp(cols, pc, pv)

    # Row slots in v order, with the rows without plants
    slots = []
    row_gap = np.diff(row_v)
    row_step = np.median(row_gap) if len(row_gap) else 0
    for i in range(len(members)):
        if i > 0 and row_step > 0:
            n_missing = max(0, int(round(row_gap[i - 1] / row_step)) - 1)
            slots += [("missing", i - 1, i, k / (n_missing + 1)) for k in range(1, n_missing + 1)]
        slots.append(("row", i))

    plant_slot = np.full(len(u), -1)
    empty = []
    def add(slot, cols, uu, vv, check):
        for c, eu, ev in zip(cols, uu, vv):
            if not check or inside is None or inside(eu, ev):
                empty.append((slot, int(c), eu, ev))

    for slot, item in enumerate(slots):
        if item[0] == "row":
            m = members[item[1]]
            plant_slot[m] = slot
            pc = plant_col[m]
            interior = sorted(set(range(pc[0], pc[-1] + 1)) - set(pc))
            add(slot, interior, *row_at(m, interior), check=False)
            n_end = max(0, int(round((np.polyval(end, v[m[-1]]) - u[m[-1]]) / spacing)))
            for cols in (list(range(1, pc[0])), list(range(pc[-1] + 1, pc[-1] + 1 + n_end))):
                if cols and (inside is not None or len(cols) <= max_edge_empty):
                    add(slot, cols, *row_at(m, cols), check=True)
        else:
            _, a, b, frac = item
            ma, mb = members[a], members[b]
            cols = list(range(max(plant_col[ma][0], plant_col[mb][0]), min(plant_col[ma][-1], plant_col[mb][-1]) + 1))
            if cols:
                (ua, va), (ub, vb) = row_at(ma, cols), row_at(mb, cols)
                add(slot, cols, ua + frac * (ub - ua), va + frac * (vb - va), check=True)
    return plant_slot, plant_col, empty, len(slots), spacing

def assign_grid_ids(centroids, srs=None, serpentine=True, start_corner="NW", swap_axes=False,
                    k=8, min_row_fraction=0.1, empty_plots=False, inside=None):
    """Number plant centroids row by row (see the step list above).

    centroids: (N, 2) array in the layer CRS.
    start_corner: "NW", "NE", "SW" or "SE". The first letter picks the first row (northmost or
        southmost), the second letter the end of that row where numbering starts.
    swap_axes: use when the detected rows are actually the columns of the trial.
    empty_plots: number planting positions (step 7), so missing plants get empty plots and keep
        their number; otherwise number the detected plants only.
    inside: optional function (x, y) in the layer CRS -> bool, true inside the field (crop polygon).
    Returns grid_id, row and col arrays (1 based; 0 for plants left out of the grid) and the
    empty plots as a list of (x, y, grid_id, row, col) in the layer CRS.
    """
    n = len(centroids)
    grid_id = np.zeros(n, dtype=int)
    row = np.zeros(n, dtype=int)
    col = np.zeros(n, dtype=int)
    if n < 3:
        grid_id[np.argsort(centroids[:, 0])] = np.arange(1, n + 1)
        return grid_id, row, col, []

    # Step 1: metres
    to_metres, from_metres = metres_transform(centroids, srs)
    pts = to_metres(centroids)
    origin = pts.mean(axis=0)
    pts = pts - origin

    # Step 2: row direction d (pointing east) and its normal nrm (pointing north)
    neighbours = nearest_neighbours(pts, k)
    theta = np.radians(estimate_row_direction(pts, neighbours))
    d = np.array([np.cos(theta), np.sin(theta)])
    if swap_axes:
        d = np.array([-d[1], d[0]])

    def orient(d):
        d = d if d[0] >= 0 else -d
        nrm = np.array([-d[1], d[0]])
        return d, (nrm if nrm[1] >= 0 else -nrm)
    d, nrm = orient(d)

    # Step 3: spacing in a row and between rows. If no neighbour lies across the rows
    # (rows far apart), twice the in-row spacing is a safe threshold for splitting rows.
    plant_spacing = link_spacing(pts, neighbours, d) or link_spacing(pts, neighbours, d, max_angle=90)
    if plant_spacing is None:
        # Too few plants to estimate the spacing (fewer than 10 with a neighbour, for example a
        # small field): number them west to east, as for n < 3, without rows or empty plots
        print(f"Numbering: only {n} plants, too few for rows; numbered west to east")
        grid_id[np.argsort(centroids[:, 0])] = np.arange(1, n + 1)
        return grid_id, row, col, []
    row_spacing = link_spacing(pts, neighbours, nrm) or 2 * plant_spacing

    # Steps 4 and 5: rotate, split rows, refine the angle with the median row slope, split again
    for refine in (True, False):
        u, v = pts @ d, pts @ nrm
        rows = trace_rows(u, v, plant_spacing, row_spacing)
        if not refine:
            break
        slopes = []
        for r in np.unique(rows):
            m = rows == r
            if m.sum() >= 3 and np.ptp(u[m]) > 2 * plant_spacing:
                slopes.append(np.polyfit(u[m], v[m], 1)[0])
        if slopes:
            # Tilt d towards nrm by the residual slope of the rows
            a = np.arctan(np.median(slopes))
            d, nrm = orient(np.cos(a) * d + np.sin(a) * nrm)
    rows = refine_rows(u, v, rows)

    # Leave out rows with very few plants (isolated detections outside the trial)
    counts = np.bincount(rows)
    keep = counts >= max(2, min_row_fraction * np.median(counts))
    valid_rows = np.flatnonzero(keep)

    north_first = start_corner.upper().startswith("N")
    west_first = start_corner.upper().endswith("W")
    in_kept_rows = keep[rows]

    if empty_plots:
        # Steps 7 and 8: planting positions, then numbering of positions (plants and empty plots).
        # col is counted from the start corner side (W or E), whatever the serpentine direction.
        def to_layer(uv):
            uv = np.asarray(uv, dtype=float).reshape(-1, 2)
            return from_metres(origin + uv[:, :1] * d + uv[:, 1:] * nrm)
        inside_uv = (lambda eu, ev: inside(*to_layer([eu, ev])[0])) if inside else None
        slot_of, col_w, empty, n_slots, spacing = planting_grid(u, v, rows, valid_rows, plant_spacing, inside_uv)
        max_col = int(max([col_w.max()] + [e[1] for e in empty]))
        to_out = (lambda c: int(c)) if west_first else (lambda c: max_col + 1 - int(c))
        positions = [dict() for _ in range(n_slots)]  # out col -> plant indices ([] for an empty plot)
        for i in np.flatnonzero(slot_of >= 0):
            positions[slot_of[i]].setdefault(to_out(col_w[i]), []).append(i)
        empty_at = {}
        for e_index, (slot, c, _, _) in enumerate(empty):
            positions[slot][to_out(c)] = []
            empty_at[(slot, to_out(c))] = e_index
        empty_ids = [None] * len(empty)
        next_id = 1
        for r_index, slot in enumerate(range(n_slots - 1, -1, -1) if north_first else range(n_slots)):
            cols_sorted = sorted(positions[slot])
            if serpentine and r_index % 2 == 1:
                cols_sorted = cols_sorted[::-1]
            for c in cols_sorted:
                plants = positions[slot][c]
                if plants:
                    grid_id[plants], row[plants], col[plants] = next_id, r_index + 1, c
                else:
                    empty_ids[empty_at[(slot, c)]] = (next_id, r_index + 1, c)
                next_id += 1
        empty_xy = to_layer([(e[2], e[3]) for e in empty]) if empty else []
        empties = [(float(x), float(y), *ids) for (x, y), ids in zip(empty_xy, empty_ids)]
        shared = sum(1 for p in positions for plants in p.values() if len(plants) > 1)
        print(f"Numbering: row angle {np.degrees(np.arctan2(d[1], d[0])):.1f} deg, position spacing "
              f"{spacing:.2f} m, {n_slots} rows ({n_slots - len(valid_rows)} without plants), "
              f"{len(empties)} empty plots, {shared} positions with two detections, "
              f"{int((~in_kept_rows).sum())} plants left out")
        return grid_id, row, col, empties

    # Steps 6 and 7 without empty plots: numbering of the detected plants. v grows to the north and
    # u to the east. col is the position of the plant in its row, counted from the start corner
    # side (W or E), whatever the serpentine direction; a missing plant shifts the positions after
    # it in that row.
    row_order = valid_rows[::-1] if north_first else valid_rows

    next_id = 1
    for r_index, r in enumerate(row_order):
        members = np.flatnonzero(rows == r)
        members = members[np.argsort(u[members])]
        if not west_first:
            members = members[::-1]
        col[members] = np.arange(1, len(members) + 1)
        if serpentine and r_index % 2 == 1:
            members = members[::-1]
        grid_id[members] = np.arange(next_id, next_id + len(members))
        row[members] = r_index + 1
        next_id += len(members)

    print(f"Numbering: row angle {np.degrees(np.arctan2(d[1], d[0])):.1f} deg, plant spacing "
          f"{plant_spacing:.2f} m, row spacing {row_spacing:.2f} m, {len(valid_rows)} rows, "
          f"{int((~in_kept_rows).sum())} plants left out")
    return grid_id, row, col, []


# --- Filtering functions ---
def filter_by_aspect_ratio(wkt_list, fields_list, min_ratio=0.2, max_ratio=5.0):
    import shapely.wkt
    keep_wkt = []
    keep_fields = []
    for wkt, fields in zip(wkt_list, fields_list):
        poly = shapely.wkt.loads(wkt)
        minx, miny, maxx, maxy = poly.bounds
        w, h = maxx - minx, maxy - miny
        if h != 0 and w != 0:
            ratio = w / h
            if min_ratio <= ratio <= max_ratio:
                keep_wkt.append(wkt)
                keep_fields.append(fields)
    return keep_wkt, keep_fields

def polygon_nms_gdal(wkt_list, fields_list, iou_threshold=0.5, score_index=-1):
    # Keep the most confident polygon of each overlapping group (score field at score_index);
    # without a score field the largest polygon is kept
    import shapely.wkt
    polys = [shapely.wkt.loads(wkt) for wkt in wkt_list]
    if score_index >= 0:
        scores = [fields[score_index] or 0.0 for fields in fields_list]
    else:
        scores = [poly.area for poly in polys]
    indices = np.argsort(scores)[::-1]
    
    keep = []
    suppressed = set()
    
    for i in indices:
        if i in suppressed:
            continue
        keep.append(i)
        poly_i = polys[i]
        for j in indices:
            if j == i or j in suppressed:
                continue
            poly_j = polys[j]
            inter = poly_i.intersection(poly_j).area
            union = poly_i.union(poly_j).area
            iou = inter / union if union > 0 else 0
            if iou > iou_threshold:
                suppressed.add(j)
                
    keep_wkt = [wkt_list[i] for i in keep]
    keep_fields = [fields_list[i] for i in keep]
    return keep_wkt, keep_fields

def empty_plot_wkt(empty, polys):
    """Box for each empty plot (x, y, ...) with the median size of the plant boxes."""
    bounds = np.array([poly.bounds for poly in polys])
    w, h = np.median(bounds[:, 2] - bounds[:, 0]), np.median(bounds[:, 3] - bounds[:, 1])
    return [shapely.geometry.box(x - w / 2, y - h / 2, x + w / 2, y + h / 2).wkt for x, y, *_ in empty]

def rotate_polygon_to_pca_axes(polygon, centroid, axes):
    coords = np.array(polygon.exterior.coords) - centroid
    R = axes
    rotated_coords = coords @ R.T
    rotated_coords += centroid
    from shapely.geometry import Polygon
    return Polygon(rotated_coords)

# --- Main pipeline ---
def read_crop_polygons(crop_shp):
    """Crop field polygons in file order; crop_id k is polygon k - 1. From crop_detection only class 0
    (crop): the model's class 1, target, is not a field; files drawn by the user without a class field
    are used whole."""
    from osgeo import ogr
    import shapely.wkt
    crop_polys = []
    ds_crop = ogr.Open(crop_shp) if crop_shp and os.path.exists(crop_shp) else None
    if ds_crop:
        layer_crop = ds_crop.GetLayer()
        class_index = layer_crop.GetLayerDefn().GetFieldIndex("class")
        for feat in layer_crop:
            geom = feat.GetGeometryRef()
            if class_index >= 0 and feat.GetField(class_index) not in (0, None):
                continue
            if geom:
                crop_polys.append(shapely.wkt.loads(geom.ExportToWkt()))
        ds_crop = None
    return crop_polys

def label_polygons_from_shapefile(input_shp, output_shp, serpentine=False, row_tol=10,
                                   iou_thresh=0.3, min_ratio=0.2, max_ratio=5.0, align_to_grid=False, only_postprocess=False,
                                   start_corner="NW", swap_axes=False, empty_plots=False, crop_polys=None):
    # row_tol and align_to_grid are kept for compatibility; numbering now uses assign_grid_ids()
    # empty_plots: add an empty plot (status "empty") for each planting position without a plant
    # crop_polys: crop field polygons (read_crop_polygons) of a layer with crop_id; with them the empty
    # plots at the row ends are added only inside the field, as tiling_detection_on_crop does
    from osgeo import ogr, osr
    import shapely.wkt
    import shapely.geometry
    
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds_in = driver.Open(input_shp)
    if ds_in is None:
        print(f"Could not open {input_shp}")
        return
    layer_in = ds_in.GetLayer()
    srs = layer_in.GetSpatialRef()
    
    wkt_list = []
    fields_list = []
    layer_defn = layer_in.GetLayerDefn()
    field_count = layer_defn.GetFieldCount()
    status_index = layer_defn.GetFieldIndex("status")

    for feat in layer_in:
        geom = feat.GetGeometryRef()
        # Empty plots of an earlier numbering are not plants: the new numbering makes its own
        if geom and not (status_index >= 0 and feat.GetField(status_index) == "empty"):
            wkt_list.append(geom.ExportToWkt())
            fields_list.append([feat.GetField(i) for i in range(field_count)])

    ds_in = None

    wkt_list, fields_list = filter_by_aspect_ratio(wkt_list, fields_list, min_ratio, max_ratio)
    wkt_list, fields_list = polygon_nms_gdal(wkt_list, fields_list, iou_thresh, layer_defn.GetFieldIndex("score"))
    status_list = ["plant"] * len(wkt_list)

    if not only_postprocess and wkt_list:
        polys = [shapely.wkt.loads(wkt) for wkt in wkt_list]
        centroids = np.array([poly.centroid.coords[0] for poly in polys])

        # Layers from tiling_detection_on_crop have a crop_id per field: number each field
        # separately (ids restart at 1), as that task does, instead of one grid across fields
        crop_index = layer_defn.GetFieldIndex("crop_id")
        crop_ids = [fields[crop_index] if crop_index >= 0 else 0 for fields in fields_list]
        grid_ids = np.zeros(len(polys), dtype=int)
        rows = np.zeros(len(polys), dtype=int)
        cols = np.zeros(len(polys), dtype=int)
        n_plants = len(wkt_list)
        # Field polygons only when they match the crop_id values (crop_id k is polygon k - 1)
        known = [c for c in crop_ids if c]
        if not crop_polys or not known or max(known) > len(crop_polys):
            crop_polys = None
        for crop in np.unique(crop_ids[:n_plants]):
            members = np.flatnonzero(np.array(crop_ids[:n_plants]) == crop)
            inside = None
            if crop_polys and crop:
                inside = lambda x, y, poly=crop_polys[int(crop) - 1]: poly.contains(shapely.geometry.Point(x, y))
            grid_ids[members], rows[members], cols[members], empty = assign_grid_ids(
                centroids[members], srs, serpentine=serpentine, start_corner=start_corner, swap_axes=swap_axes,
                empty_plots=empty_plots, inside=inside)
            # Empty plots: box of the median plant size, no score or class
            for (x, y, g, r, c), wkt in zip(empty, empty_plot_wkt(empty, [polys[i] for i in members])):
                fields = [None] * field_count
                if crop_index >= 0:
                    fields[crop_index] = crop.item() if hasattr(crop, "item") else crop
                wkt_list.append(wkt)
                fields_list.append(fields)
                status_list.append("empty")
                crop_ids.append(crop)
                grid_ids, rows, cols = np.append(grid_ids, g), np.append(rows, r), np.append(cols, c)

        # Write features in numbering order (by field); plants left out of the grid (id 0) go last
        order = sorted(range(len(wkt_list)), key=lambda i: (crop_ids[i], grid_ids[i] == 0, grid_ids[i]))
        labels = [(int(grid_ids[i]), int(rows[i]), int(cols[i])) for i in order]
        wkt_list = [wkt_list[i] for i in order]
        fields_list = [fields_list[i] for i in order]
        status_list = [status_list[i] for i in order]

    if os.path.exists(output_shp):
        driver.DeleteDataSource(output_shp)
    ds_out = driver.CreateDataSource(output_shp)
    layer_out = ds_out.CreateLayer("labeled", srs, ogr.wkbPolygon)

    for i in range(field_count):
        layer_out.CreateField(layer_defn.GetFieldDefn(i))
    if not only_postprocess:
        # Reuse the fields when renumbering a layer that already has them
        for name in ("grid_id", "row", "col"):
            if layer_out.FindFieldIndex(name, True) < 0:
                layer_out.CreateField(ogr.FieldDefn(name, ogr.OFTInteger))
        if empty_plots and layer_out.FindFieldIndex("status", True) < 0:
            status_field = ogr.FieldDefn("status", ogr.OFTString)
            status_field.SetWidth(8)
            layer_out.CreateField(status_field)

    for idx, (wkt, fields) in enumerate(zip(wkt_list, fields_list)):
        geom = ogr.CreateGeometryFromWkt(wkt)
        feat = ogr.Feature(layer_out.GetLayerDefn())
        for i, val in enumerate(fields):
            if val is not None:
                feat.SetField(i, val)
        if not only_postprocess:
            feat.SetField("grid_id", labels[idx][0])
            feat.SetField("row", labels[idx][1])
            feat.SetField("col", labels[idx][2])
            if empty_plots:
                feat.SetField("status", status_list[idx])
        feat.SetGeometry(geom)
        layer_out.CreateFeature(feat)
        feat = None
        
    ds_out = None

def merge_shp_gdal(shp_paths, output_path, dissolve=False, explode=False, nms=False, nms_iou=0.5):
    from osgeo import ogr
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_path):
        driver.DeleteDataSource(output_path)
    if not shp_paths:
        return None
    ds_template = driver.Open(shp_paths[0])
    layer_template = ds_template.GetLayer()
    srs = layer_template.GetSpatialRef()
    geom_type = layer_template.GetGeomType()
    layer_defn = layer_template.GetLayerDefn()
    ds_template = None
    ds_out = driver.CreateDataSource(output_path)
    layer_out = ds_out.CreateLayer("merged", srs, geom_type)
    for i in range(layer_defn.GetFieldCount()):
        layer_out.CreateField(layer_defn.GetFieldDefn(i))
    for shp_path in shp_paths:
        ds_in = driver.Open(shp_path)
        if ds_in is None:
            continue
        layer_in = ds_in.GetLayer()
        for feat_in in layer_in:
            geom = feat_in.GetGeometryRef()
            if geom:
                feat_out = ogr.Feature(layer_out.GetLayerDefn())
                for i in range(layer_defn.GetFieldCount()):
                    feat_out.SetField(i, feat_in.GetField(i))
                feat_out.SetGeometry(geom.Clone())
                layer_out.CreateFeature(feat_out)
                feat_out = None
        ds_in = None
    ds_out.FlushCache()
    ds_out = None
    return output_path

# --- Tile overlap ---
# Tiles overlap by TILE_OVERLAP pixels so that every plant appears whole in at least one tile
# (a plant cut by a tile border would otherwise give two partial boxes or none). 128 px is about
# 3.8 m at 3 cm/px, twice the size of a plant box (about 1.85 m, 62 px).
# A plant in an overlap is detected by two or more tiles, and the model scores the same plant
# differently depending on where it sits in the tile (on the sample, 0.58 in one tile and 0.14
# in the next). keep_core_detections therefore (1) drops boxes cut by an inner tile border, since
# the whole plant is in the neighbouring tile, and (2) keeps the most confident copy of each plant
# across tiles (boxes of different tiles with IoU > 0.5 are the same plant).
# Polygon NMS still runs afterwards as a safety net.
TILE_OVERLAP = 128
TILE_DUPLICATE_IOU = 0.5
# Final polygon NMS before numbering. Boxes of neighbouring plants in dense rows overlap by up
# to about 0.2 IoU, so lower values remove real plants (0.15 removed 62 of 1525 on the sample).
ROI_NMS_IOU = 0.3

def raster_pixel_size_m(raster_path):
    """Approximate ground pixel size of a raster in metres (degrees converted at its centre), or None."""
    src = gdal.Open(raster_path)
    if src is None:
        return None
    gt = src.GetGeoTransform()
    size = (abs(gt[1]) + abs(gt[5])) / 2
    srs = osr.SpatialReference(wkt=src.GetProjection())
    if srs.IsGeographic():
        lat = gt[3] + gt[5] * src.RasterYSize / 2
        return size * 111320 * np.sqrt(np.cos(np.radians(lat)))
    return size * (srs.GetLinearUnits() or 1) if srs.IsProjected() else None

def tile_grid(size, tile_size, overlap):
    """Number of tiles and stride along one axis for a raster of `size` pixels."""
    stride = tile_size - overlap
    count = max(1, int(np.ceil((size - overlap) / stride)))
    return count, stride

def keep_core_detections(shp_files, images_dir, raster_path, tile_size, overlap, out_dir, border_px=2):
    """Write copies of the per tile shapefiles without boxes cut by an inner tile border and
    without duplicates across tiles (the most confident copy is kept). Returns the new files."""
    from osgeo import gdal, ogr
    src = gdal.Open(raster_path)
    gt = src.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)
    nx, _ = tile_grid(src.RasterXSize, tile_size, overlap)
    ny, _ = tile_grid(src.RasterYSize, tile_size, overlap)
    stride = tile_size - overlap
    src = None
    driver = ogr.GetDriverByName("ESRI Shapefile")
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: read every detection with its box in orthomosaic pixels; drop boxes cut by a border
    # shared with another tile (borders on the edge of the orthomosaic are kept)
    kept_files, candidates, cut = [], [], 0
    for shp in shp_files:
        name = os.path.basename(shp)
        tile_tif = os.path.join(images_dir, name.replace("_boxes.shp", ".tif"))
        tile_ds = gdal.Open(tile_tif)
        if tile_ds is None:  # unknown tile: keep everything
            kept_files.append(shp)
            continue
        tgt = tile_ds.GetGeoTransform()
        tile_ds = None
        col = int(round((tgt[0] - gt[0]) / gt[1] / stride))
        row = int(round((tgt[3] - gt[3]) / gt[5] / stride))
        tx0, ty0 = col * stride, row * stride

        ds_in = driver.Open(shp)
        layer_in = ds_in.GetLayer()
        score_index = layer_in.GetLayerDefn().GetFieldIndex("score")
        for feat in layer_in:
            geom = feat.GetGeometryRef()
            if geom is None:
                continue
            minx, maxx, miny, maxy = geom.GetEnvelope()
            xs = sorted([inv_gt[0] + inv_gt[1] * minx + inv_gt[2] * maxy, inv_gt[0] + inv_gt[1] * maxx + inv_gt[2] * miny])
            ys = sorted([inv_gt[3] + inv_gt[4] * minx + inv_gt[5] * maxy, inv_gt[3] + inv_gt[4] * maxx + inv_gt[5] * miny])
            if ((col > 0 and xs[0] - tx0 <= border_px) or (col < nx - 1 and tx0 + tile_size - xs[1] <= border_px) or
                    (row > 0 and ys[0] - ty0 <= border_px) or (row < ny - 1 and ty0 + tile_size - ys[1] <= border_px)):
                cut += 1
                continue
            score = feat.GetField(score_index) if score_index >= 0 else 0.0
            candidates.append((shp, name, feat.Clone(), (xs[0], ys[0], xs[1], ys[1]), score or 0.0))
        ds_in = None

    # Step 2: greedy NMS across tiles by score; boxes of the same tile were already deduplicated
    boxes = np.array([c[3] for c in candidates]).reshape(-1, 4)
    tiles = np.array([c[1] for c in candidates])
    order = np.argsort([-c[4] for c in candidates])
    removed = np.zeros(len(candidates), dtype=bool)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for i in order:
        if removed[i]:
            continue
        ix = np.clip(np.minimum(boxes[i, 2], boxes[:, 2]) - np.maximum(boxes[i, 0], boxes[:, 0]), 0, None)
        iy = np.clip(np.minimum(boxes[i, 3], boxes[:, 3]) - np.maximum(boxes[i, 1], boxes[:, 1]), 0, None)
        iou = ix * iy / np.maximum(areas[i] + areas - ix * iy, 1e-9)
        removed |= (iou > TILE_DUPLICATE_IOU) & (tiles != tiles[i])

    # Step 3: write the kept detections, one file per tile as before
    by_tile = {}
    for c, rm in zip(candidates, removed):
        by_tile.setdefault((c[0], c[1]), []).append(None if rm else c[2])
    for (shp, name), feats in by_tile.items():
        ds_in = driver.Open(shp)
        layer_in = ds_in.GetLayer()
        out_path = os.path.join(out_dir, name)
        if os.path.exists(out_path):
            driver.DeleteDataSource(out_path)
        ds_out = driver.CreateDataSource(out_path)
        layer_out = ds_out.CreateLayer("core", layer_in.GetSpatialRef(), layer_in.GetGeomType())
        defn = layer_in.GetLayerDefn()
        for i in range(defn.GetFieldCount()):
            layer_out.CreateField(defn.GetFieldDefn(i))
        for feat in feats:
            if feat is not None:
                layer_out.CreateFeature(feat)
        ds_in, ds_out = None, None
        kept_files.append(out_path)
    print(f"Tile overlap: kept {int((~removed).sum())} detections, dropped {cut} cut by a tile border"
          f" and {int(removed.sum())} duplicated in overlaps")
    return kept_files

def tile_raster_gdal(input_raster_path, tiles_dir, tile_size, prefix="", crop_shapefile=None, overlap=0):
    from osgeo import gdal
    src_ds = gdal.Open(input_raster_path)
    if src_ds is None:
        raise ValueError(f"Could not open raster: {input_raster_path}")
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize
    bands = src_ds.RasterCount
    geotransform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    data_type = src_ds.GetRasterBand(1).DataType
    # Tiles start every `stride` pixels, so neighbouring tiles share `overlap` pixels
    tiles_x, stride = tile_grid(width, tile_size, overlap)
    tiles_y, _ = tile_grid(height, tile_size, overlap)
    os.makedirs(tiles_dir, exist_ok=True)
    tile_files = []
    tile_count = 0
    nodata_value = -9999
    
    crop_geom = None
    if crop_shapefile and os.path.exists(crop_shapefile):
        ds_shp = ogr.Open(crop_shapefile)
        if ds_shp:
            layer = ds_shp.GetLayer()
            crop_geom = ogr.Geometry(ogr.wkbMultiPolygon)
            for feature in layer:
                geom = feature.GetGeometryRef()
                if geom:
                    crop_geom.AddGeometry(geom.Clone())
            ds_shp = None
    for row in range(tiles_y):
        for col in range(tiles_x):
            x_offset = col * stride
            y_offset = row * stride
            
            # Check crop shapefile intersection
            if crop_geom:
                minx = geotransform[0] + x_offset * geotransform[1]
                maxy = geotransform[3] + y_offset * geotransform[5]
                maxx = minx + tile_size * geotransform[1]
                miny = maxy + tile_size * geotransform[5]
                
                ring = ogr.Geometry(ogr.wkbLinearRing)
                ring.AddPoint(minx, miny)
                ring.AddPoint(maxx, miny)
                ring.AddPoint(maxx, maxy)
                ring.AddPoint(minx, maxy)
                ring.AddPoint(minx, miny)
                tile_poly = ogr.Geometry(ogr.wkbPolygon)
                tile_poly.AddGeometry(ring)
                
                if not crop_geom.Intersects(tile_poly):
                    continue
                    
            tile_filename = os.path.join(tiles_dir, f"{prefix}{tile_count:05d}.tif")
            if os.path.exists(tile_filename):
                tile_files.append(tile_filename)
                tile_count += 1
                continue
            x_size = min(tile_size, width - x_offset)
            y_size = min(tile_size, height - y_offset)
            tile_geotransform = list(geotransform)
            tile_geotransform[0] = geotransform[0] + x_offset * geotransform[1]
            tile_geotransform[3] = geotransform[3] + y_offset * geotransform[5]
            driver = gdal.GetDriverByName('GTiff')
            tile_ds = driver.Create(tile_filename, tile_size, tile_size, bands, data_type, options=['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=IF_SAFER'])
            tile_ds.SetGeoTransform(tile_geotransform)
            tile_ds.SetProjection(projection)
            for band_idx in range(1, bands + 1):
                src_band = src_ds.GetRasterBand(band_idx)
                tile_band = tile_ds.GetRasterBand(band_idx)
                data = src_band.ReadAsArray(x_offset, y_offset, x_size, y_size)
                if x_size < tile_size or y_size < tile_size:
                    if np.issubdtype(data.dtype, np.unsignedinteger) and nodata_value < 0:
                        nodata_value = 255
                    padded_data = np.full((tile_size, tile_size), nodata_value, dtype=data.dtype)
                    padded_data[:y_size, :x_size] = data
                    data = padded_data
                tile_band.WriteArray(data)
                tile_band.SetNoDataValue(nodata_value)
            tile_ds.FlushCache()
            tile_ds = None
            tile_files.append(tile_filename)
            tile_count += 1
    src_ds = None
    return tile_files


    

class ForagesROIsDetector():

    def __init__(self):

        self.ort_sess = None

        pass

    def initialize(self, model_filename=MODEL_FILENAME):

        #Load model
        model_filepath = model_path(model_filename)
        if self.ort_sess is None or getattr(self, "current_model_file", None) != model_filepath:

            if not os.path.exists(model_filepath):
                raise FileNotFoundError(f"Model not found:\n{model_filepath}\n\n"
                                        "Choose the models in File > Preferences, "
                                        "or download them with File > Download Models "
                                        "(command line: add --download_models).")

            # GPU (CUDA) when available, otherwise CPU (see execution_providers)
            self.ort_sess = ort.InferenceSession(model_filepath, providers=execution_providers())
            self.current_model_file = model_filepath
            self.model_file = model_filepath
            self.device = "GPU" if "CUDAExecutionProvider" in self.ort_sess.get_providers() else "CPU"
            print(f"Model {model_filepath} on the {self.device}")
            try:
                from utils.app_settings import settings
                settings().setValue("last_device", self.device)  # shown in Preferences
            except Exception:
                pass

            # self.ort_sess = ort.InferenceSession(model_filepath
            #                     ,providers=ort.get_available_providers()
            #                     )

        return
    
    def inference(self, filepath, output_folder=None, model_filename=MODEL_FILENAME, output_filename=None,
                  conf_threshold=PLANT_CONFIDENCE):

        self.initialize(model_filename)

        # Get basename without extension
        basename = os.path.splitext(os.path.basename(filepath))[0]
        if output_folder is None:
            # Set output folder as filepath dir
            output_folder = os.path.dirname(filepath)

        os.makedirs(output_folder, exist_ok=True)
        
        np_image = cv.imread(filepath)
        np_image = cv.cvtColor(np_image, cv.COLOR_BGR2RGB)

        is_raster = False

        # Check if file is a raster tif image
        extent = None
        epsg = None
        if filepath.lower().endswith('.tif') or filepath.lower().endswith('.tiff'):
            is_raster = True
            ds = gdal.Open(filepath)
            if ds is None:
                raise ValueError(f"Could not open raster file: {filepath}")
            gt = ds.GetGeoTransform()
            width = ds.RasterXSize
            height = ds.RasterYSize
            
            left = gt[0]
            top = gt[3]
            right = left + width * gt[1] + height * gt[2]
            bottom = top + width * gt[4] + height * gt[5]
            extent = (min(left, right), min(bottom, top), max(left, right), max(bottom, top))
            
            epsg = ds.GetProjection()
            if not epsg:
                raise ValueError(f"No CRS found in raster file: {filepath}")
            ds = None
        else:
            # extent and epsg must be provided or set elsewhere for non-tif images
            raise ValueError("EPSG code must be provided for non-tif images.")
        
        print("Projection / EPSG:", epsg)

        img_prec, scale, (h0, w0) = preprocess(np_image)
        outputs = self.ort_sess.run(None, {'images':img_prec})
        outputs = postprocess_yolov8_output(outputs, conf_threshold=conf_threshold, nms_threshold=PLANT_NMS_IOU,
                                            orig_shape=(1024,1024), agnostic=True)
        boxes_df = outputs_to_df(outputs)

        if is_raster:

            shp_bbox = output_filename if output_filename else os.path.join(output_folder, basename + "_boxes.shp")

            print("Saving shapefile to", shp_bbox)

            save_shapefile_bb(add_class_names(boxes_df, PLANT_CLASS_NAMES),
                                extent,
                                np_image.shape[1],
                                np_image.shape[0],
                                epsg,
                                allow_cols=["score","class","class_name"]
                                , output_filename=shp_bbox
                                )
        else:

            #Draw boxes on the image


            # csv fileanme
            csv_filename = os.path.join(output_folder, basename + "_boxes.csv")

            print("Saving csv file to", csv_filename)
            # Save dataframe boxes_df
            boxes_df.to_csv(csv_filename, index=False)

    @staticmethod
    def tile_progress(progress_callback):
        """Convert BatchProcessor progress ("percent") to the GUI format ("progress", "status").
        Tiles take 5 to 90 % of the bar; the rest is tile extraction, merging and numbering."""
        if not progress_callback:
            return None
        def callback(info):
            done, total = info.get("processed_count", 0), info.get("total_files", 0)
            progress_callback({"progress": 5 + 0.85 * info.get("percent", 0),
                               "status": f"Detecting plants: tile {done} of {total}"})
        return callback

    def batch_processing(self, folder, output_folder, format="tif"
                        , progress_callback=None
                        , interruption_check=None
                        , conf_threshold=PLANT_CONFIDENCE
                        ):

        processor = BatchProcessor()

        def processFunction(filepath, output_files):

            if os.path.exists(output_files[0]):
                print(f"File already exists {output_files[0]}")
            else:
                print(output_files[0])
                output_dir = os.path.dirname(output_files[0])

                self.inference(filepath, output_dir, conf_threshold=conf_threshold)
                # results = self.inference_file(filepath)

                # for index, result in enumerate(results):
                #     cv.imwrite(output_files[index], result)
                #     print(f"File saved {output_files[index]}")                

        processor.batch_process(input_dir=folder
                                , output_dir=output_folder
                                , processing_fc=processFunction
                                , pattern = '**/*.' + format
                                , output_suffixes = ["boxes"]
                                , output_format="shp"
                                , format=format
                                , progress_callback=progress_callback
                                , interruption_check=interruption_check
                                )
                                
    def crop_inference(self, input_filepath, output_filepath, progress_callback=None, interruption_check=None):
        if progress_callback: progress_callback({"progress": 10, "status": "Running crop detection model..."})
        
        self.initialize(CROP_MODEL_FILENAME)
        
        basename = os.path.splitext(os.path.basename(input_filepath))[0]
        output_folder = os.path.dirname(output_filepath)
        os.makedirs(output_folder, exist_ok=True)
        
        from osgeo import gdal
        ds = gdal.Open(input_filepath)
        if ds is None:
            raise ValueError(f"Could not open raster file: {input_filepath}")
            
        gt = ds.GetGeoTransform()
        width = ds.RasterXSize
        height = ds.RasterYSize
        
        left = gt[0]
        top = gt[3]
        right = left + width * gt[1] + height * gt[2]
        bottom = top + width * gt[4] + height * gt[5]
        extent = (min(left, right), min(bottom, top), max(left, right), max(bottom, top))
        
        epsg = ds.GetProjection()
        if not epsg:
            raise ValueError(f"No CRS found in raster file: {input_filepath}")
            
        imgsz = 640
        r = imgsz / max(width, height)
        new_w = int(width * r)
        new_h = int(height * r)
        
        img_array = ds.ReadAsArray(0, 0, width, height, buf_xsize=new_w, buf_ysize=new_h)
        ds = None
        
        import numpy as np
        import cv2 as cv
        
        if len(img_array.shape) >= 3:
            np_image = np.stack([img_array[0], img_array[1], img_array[2]], axis=-1)
        else:
            np_image = np.stack([img_array, img_array, img_array], axis=-1)
            
        if np_image.dtype != np.uint8:
            np_image = cv.normalize(np_image, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            
        img_prec, scale, (h0, w0) = preprocess(np_image, imgsz=imgsz, pad_color=114)
        
        outputs = self.ort_sess.run(None, {'images':img_prec})
        
        conf_threshold = 0.25
        nms_threshold = 0.7
        
        if len(outputs) >= 2:
            from processing import yolo_seg_util
            outputs = yolo_seg_util.postprocess_yolov8_seg_output(outputs, conf_threshold=conf_threshold, nms_threshold=nms_threshold, input_size=imgsz, orig_shape=(new_h, new_w))
        else:
            outputs = postprocess_yolov8_output(outputs, conf_threshold=conf_threshold, nms_threshold=nms_threshold, input_size=imgsz, orig_shape=(new_h, new_w))
            
        boxes_df = outputs_to_df(outputs)
        
        shp_bbox = output_filepath if output_filepath else os.path.join(output_folder, basename + "_crops.shp")
        
        save_shapefile_bb(add_class_names(boxes_df, CROP_CLASS_NAMES),
                            extent,
                            new_w,
                            new_h,
                            epsg,
                            allow_cols=["score","class","class_name"],
                            output_filename=shp_bbox
                            )
                            
        if progress_callback: progress_callback({"progress": 100, "status": "Crop detection complete."})

    def tile_inference(self, input_filepath, output_filepath, only=False, clean_cache=False, progress_callback=None, interruption_check=None,
                       conf_threshold=PLANT_CONFIDENCE, empty_plots=False):

        # Get basename without extension
        basename = os.path.splitext(os.path.basename(output_filepath))[0]

        # Set output folder as filepath dir
        output_folder = os.path.dirname(output_filepath)
        
        from utils.cachemanager import CacheManager
        cache_manager = CacheManager(project_path=output_folder)
        
        tile_size = 1024
        
        cache_key = {
            "input_raster_path": input_filepath,
            "tile_size": tile_size,
            "tile_overlap": TILE_OVERLAP
        }
        key = cache_manager.compute_key(cache_key)
        # Detections depend also on the threshold, tiles do not
        det_key = cache_manager.compute_key({**cache_key, "confidence": conf_threshold, "parser": "yolov8", "model": model_stamp()})

        images_dir = cache_manager.get_cache_folder_path("tiles", key)
        shp_dir = cache_manager.get_cache_folder_path("shp", det_key)

        os.makedirs(output_folder, exist_ok=True)

        # Extract tiles and save using GDAL
        if progress_callback: progress_callback({"progress": 2, "status": "Extracting tiles..."})
        if not cache_manager.exists("tiles", key):
            print(f"Extracting tiles from {input_filepath} to cache...")
            tile_raster_gdal(input_filepath, images_dir, tile_size, prefix="tile_", overlap=TILE_OVERLAP)
        else:
            print(f"Using cached tiles from {images_dir}")

        # Process each tile
        self.batch_processing(images_dir, shp_dir, progress_callback=self.tile_progress(progress_callback), interruption_check=interruption_check,
                              conf_threshold=conf_threshold)

        # Keep each detection only in the tile that owns its centre, then merge
        shp_files = glob.glob(os.path.join(shp_dir, "*.shp"))
        shp_files = keep_core_detections(shp_files, images_dir, input_filepath, tile_size, TILE_OVERLAP,
                                         cache_manager.get_cache_folder_path("core", det_key))
        print(f"Merging {len(shp_files)} files")
        if progress_callback: progress_callback({"progress": 92, "status": "Merging tiles and numbering plants..."})

        if shp_files:
            merged_dir = cache_manager.get_cache_folder_path("merged", det_key)
            merged_shp = os.path.join(merged_dir, "merged.shp")
            merge_shp_gdal(shp_files, merged_shp)

            # Post process the merged shapefile
            safe_path = os.path.normpath(output_filepath)
            if not only:
                label_polygons_from_shapefile(merged_shp, safe_path, serpentine=True, row_tol=1.0, min_ratio=1/1.8, max_ratio=1.8, iou_thresh=ROI_NMS_IOU, align_to_grid=False,
                                              empty_plots=empty_plots)
            else:
                import shutil
                shutil.copyfile(merged_shp, safe_path)
                if os.path.exists(merged_shp.replace('.shp','.shx')):
                    shutil.copyfile(merged_shp.replace('.shp','.shx'), safe_path.replace('.shp','.shx'))
                if os.path.exists(merged_shp.replace('.shp','.dbf')):
                    shutil.copyfile(merged_shp.replace('.shp','.dbf'), safe_path.replace('.shp','.dbf'))
                if os.path.exists(merged_shp.replace('.shp','.prj')):
                    shutil.copyfile(merged_shp.replace('.shp','.prj'), safe_path.replace('.shp','.prj'))
            
            if clean_cache:
                print("Cleaning cache for this execution...")
                cache_manager.clean_cache_folder_path("tiles", key)
                cache_manager.clean_cache_folder_path("shp", det_key)
                cache_manager.clean_cache_folder_path("merged", det_key)
                cache_manager.clean_cache_folder_path("core", det_key)
        else:
            print("No shapefiles found to merge in", shp_dir)

    def plot_numbering(self, input_filepath, output_filepath, serpentine=True, align_to_grid=False, only_postprocess=False,
                       start_corner="NW", swap_axes=False, empty_plots=False, crop_shapefile=None):
        # crop_shapefile: the crop fields of a tiling_detection_on_crop result, so that renumbering adds
        # the same empty plots at the row ends (inside each field) as the first run
        safe_input_filepath = os.path.normpath(input_filepath)
        safe_input_output_filepath = os.path.normpath(output_filepath)
        # Output may be the input itself (numbering written into the shown shapefile, like the
        # edits): write a temporary shapefile next to it, then replace the input files with it
        in_place = os.path.normcase(os.path.abspath(safe_input_filepath)) == os.path.normcase(os.path.abspath(safe_input_output_filepath))
        if in_place:
            safe_input_output_filepath = os.path.splitext(safe_input_filepath)[0] + "_numbering_tmp.shp"

        # Post process the merged shapefile
        label_polygons_from_shapefile(safe_input_filepath, safe_input_output_filepath, serpentine=serpentine, row_tol=1.0, min_ratio=1/1.8, max_ratio=1.8, iou_thresh=ROI_NMS_IOU, align_to_grid=align_to_grid, only_postprocess=only_postprocess,
                                      start_corner=start_corner, swap_axes=swap_axes, empty_plots=empty_plots,
                                      crop_polys=read_crop_polygons(crop_shapefile))

        if in_place:
            replace_shapefile(safe_input_output_filepath, safe_input_filepath)

    def plant_traits(self, raster_filepath, rois_filepath, output_filepath, index=None, threshold=None, progress_callback=None,
                     bands=None):
        # Plant outlines and traits from the ROIs (see compute_plant_traits)
        return compute_plant_traits(os.path.normpath(raster_filepath), os.path.normpath(rois_filepath),
                                    os.path.normpath(output_filepath), index=index, threshold=threshold,
                                    progress_callback=progress_callback, bands=bands)

    def tile_inference_on_crop(self, input_filepath, output_filepath, crop_shapefile, only=False, clean_cache=False, progress_callback=None, interruption_check=None,
                               conf_threshold=PLANT_CONFIDENCE, empty_plots=False):
        basename = os.path.splitext(os.path.basename(output_filepath))[0]
        output_folder = os.path.dirname(output_filepath)
        
        from utils.cachemanager import CacheManager
        cache_manager = CacheManager(project_path=output_folder)
        
        tile_size = 1024
        cache_key = {"input_raster_path": input_filepath, "tile_size": tile_size, "tile_overlap": TILE_OVERLAP, "crop_mask": crop_shapefile}
        key = cache_manager.compute_key(cache_key)
        # Detections depend also on the threshold, tiles do not
        det_key = cache_manager.compute_key({**cache_key, "confidence": conf_threshold, "parser": "yolov8", "model": model_stamp()})

        images_dir = cache_manager.get_cache_folder_path("tiles", key)
        shp_dir = cache_manager.get_cache_folder_path("shp", det_key)
        os.makedirs(output_folder, exist_ok=True)

        if progress_callback: progress_callback({"progress": 2, "status": "Extracting tiles over crops..."})
        if not cache_manager.exists("tiles", key):
            print(f"Extracting tiles from {input_filepath} over crops to cache...")
            tile_raster_gdal(input_filepath, images_dir, tile_size, prefix="tile_", crop_shapefile=crop_shapefile, overlap=TILE_OVERLAP)
        else:
            print(f"Using cached crop tiles from {images_dir}")

        self.batch_processing(images_dir, shp_dir, progress_callback=self.tile_progress(progress_callback), interruption_check=interruption_check,
                              conf_threshold=conf_threshold)

        # Keep each detection only in the tile that owns its centre, then merge
        shp_files = glob.glob(os.path.join(shp_dir, "*.shp"))
        shp_files = keep_core_detections(shp_files, images_dir, input_filepath, tile_size, TILE_OVERLAP,
                                         cache_manager.get_cache_folder_path("core", det_key))
        print(f"Merging {len(shp_files)} files")
        if progress_callback: progress_callback({"progress": 92, "status": "Merging tiles and numbering plants..."})

        if shp_files:
            merged_dir = cache_manager.get_cache_folder_path("merged", det_key)
            merged_shp = os.path.join(merged_dir, "merged.shp")
            merge_shp_gdal(shp_files, merged_shp)

            safe_path = os.path.normpath(output_filepath)
            label_polygons_from_shapefile_by_crop(merged_shp, safe_path, crop_shapefile, serpentine=True, row_tol=1.0, min_ratio=1/1.8, max_ratio=1.8, iou_thresh=ROI_NMS_IOU,
                                                  empty_plots=empty_plots)

            if clean_cache:
                cache_manager.clean_cache_folder_path("tiles", key)
                cache_manager.clean_cache_folder_path("shp", det_key)
                cache_manager.clean_cache_folder_path("merged", det_key)
                cache_manager.clean_cache_folder_path("core", det_key)
        else:
            print("No shapefiles found to merge in", shp_dir)

def label_polygons_from_shapefile_by_crop(input_shp, output_shp, crop_shp, serpentine=True, row_tol=1.0, min_ratio=1/1.8, max_ratio=1.8, iou_thresh=ROI_NMS_IOU,
                                          start_corner="NW", swap_axes=False, empty_plots=False):
    # row_tol is kept for compatibility; numbering now uses assign_grid_ids()
    # empty_plots: add an empty plot (status "empty") for each planting position without a plant;
    # at the row ends only inside the crop polygon
    from osgeo import ogr
    import shapely.wkt
    import shapely.geometry
    import os
    
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds_in = driver.Open(input_shp)
    if not ds_in: return
    layer_in = ds_in.GetLayer()
    srs = layer_in.GetSpatialRef()
    
    crop_polys = read_crop_polygons(crop_shp)

    # Load plants
    plants = []
    layer_defn = layer_in.GetLayerDefn()
    field_count = layer_defn.GetFieldCount()
    status_index = layer_defn.GetFieldIndex("status")
    for feat in layer_in:
        geom = feat.GetGeometryRef()
        if geom and not (status_index >= 0 and feat.GetField(status_index) == "empty"):
            wkt = geom.ExportToWkt()
            fields = [feat.GetField(i) for i in range(field_count)]
            plants.append((wkt, fields))
    ds_in = None

    # Process per crop
    final_labeled_plants = []
    
    for crop_id, crop_poly in enumerate(crop_polys, start=1):
        crop_wkt_list = []
        crop_fields_list = []
        for wkt, fields in plants:
            plant_poly = shapely.wkt.loads(wkt)
            if crop_poly.contains(plant_poly.centroid) or crop_poly.intersects(plant_poly):
                crop_wkt_list.append(wkt)
                crop_fields_list.append(fields)
                
        if not crop_wkt_list:
            continue
            
        crop_wkt_list, crop_fields_list = filter_by_aspect_ratio(crop_wkt_list, crop_fields_list, min_ratio, max_ratio)
        crop_wkt_list, crop_fields_list = polygon_nms_gdal(crop_wkt_list, crop_fields_list, iou_thresh, layer_defn.GetFieldIndex("score"))
        
        if not crop_wkt_list:
            continue
            
        polys = [shapely.wkt.loads(wkt) for wkt in crop_wkt_list]
        centroids = np.array([poly.centroid.coords[0] for poly in polys])
        # Each crop field is numbered on its own, starting at 1
        inside = lambda x, y, poly=crop_poly: poly.contains(shapely.geometry.Point(x, y))
        grid_ids, rows, cols, empty = assign_grid_ids(centroids, srs, serpentine=serpentine,
                                                      start_corner=start_corner, swap_axes=swap_axes,
                                                      empty_plots=empty_plots, inside=inside)

        crop_labeled = [(g, r, c, wkt, fields, "plant") for g, r, c, wkt, fields
                        in zip(grid_ids, rows, cols, crop_wkt_list, crop_fields_list)]
        # Empty plots: box of the median plant size, no score or class
        crop_labeled += [(g, r, c, wkt, [None] * field_count, "empty")
                         for (x, y, g, r, c), wkt in zip(empty, empty_plot_wkt(empty, polys))]
        crop_labeled.sort(key=lambda x: (x[0] == 0, x[0]))  # numbering order, left out plants last
        for grid_id, r, c, wkt, fields, status in crop_labeled:
            final_labeled_plants.append((crop_id, int(grid_id), int(r), int(c), wkt, fields, status))

    if os.path.exists(output_shp): driver.DeleteDataSource(output_shp)
    ds_out = driver.CreateDataSource(output_shp)
    layer_out = ds_out.CreateLayer("labeled", srs, ogr.wkbPolygon)

    for name in ("crop_id", "grid_id", "row", "col"):
        layer_out.CreateField(ogr.FieldDefn(name, ogr.OFTInteger))
    if empty_plots:
        status_field = ogr.FieldDefn("status", ogr.OFTString)
        status_field.SetWidth(8)
        layer_out.CreateField(status_field)
    for i in range(field_count):
        if layer_out.FindFieldIndex(layer_defn.GetFieldDefn(i).GetName(), True) < 0:
            layer_out.CreateField(layer_defn.GetFieldDefn(i))

    for crop_id, grid_id, r, c, wkt, fields, status in final_labeled_plants:
        geom = ogr.CreateGeometryFromWkt(wkt)
        feat = ogr.Feature(layer_out.GetLayerDefn())
        for i, val in enumerate(fields):
            if val is not None:
                feat.SetField(layer_defn.GetFieldDefn(i).GetName(), val)
        # Set the numbering last so it replaces any old values from the input layer
        feat.SetField("crop_id", crop_id)
        feat.SetField("grid_id", grid_id)
        feat.SetField("row", r)
        feat.SetField("col", c)
        if empty_plots:
            feat.SetField("status", status)
        feat.SetGeometry(geom)
        layer_out.CreateFeature(feat)
        feat = None
    ds_out = None


# --- Plant traits ---
# compute_plant_traits() turns plant ROIs (boxes) into plant outlines and traits, in sequential steps:
#   1. Read the raster window under each box (margin 0 by default, so the outline never leaves the
#      detected box). The original raster is read, not the RGB view, so that a multispectral
#      orthomosaic also gives its red edge and near infrared bands (spectral_indices.find_bands).
#   2. Compute a vegetation index in each window: NDVI when there is a near infrared band, otherwise
#      RGBVI; ExG, NDRE and GNDVI can be chosen. The canopy means of NDVI, NDRE and GNDVI are also
#      written when the bands exist.
#   3. Choose the threshold that separates plant from soil and shadow: Otsu on the index values of
#      all windows of the field (crop_id), unless the user gives a fixed value.
#   4. Voronoi split: a pixel covered by several windows belongs to the plant with the nearest box
#      centre, so every pixel is counted for one plant only. Each cell stays inside its own window.
#   5. Join leaves split by shadows (small closing), keep the largest connected group of plant
#      pixels in the cell (drops small weeds and noise) and fill its holes.
#   6. Outline that group as a polygon and compute the traits: canopy area (m2), cell area (m2),
#      cover (% of the cell) and mean index of the canopy.
# The result is a new layer of plant outlines with the ROI ids (crop_id, grid_id, row, col) and a
# CSV with the same table. Plants with no plant pixels keep their box as geometry and canopy 0; so do the
# empty plots of the numbering (status "empty"), which are also left out of steps 3 and 4.

# Step 2 (vegetation indices: RGBVI, ExG, NDVI, NDRE, GNDVI) is in spectral_indices.py

def otsu_threshold(values, bins=256):
    """Step 3: Otsu threshold of 1D values (maximises the variance between the two classes)."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    hist, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centres[np.argmax(between)])

def pixel_area_m2(gt, srs, x, y):
    """Area of one pixel in m2 at map position (x, y), also for geographic rasters."""
    pts = np.array([[x, y], [x + gt[1], y], [x, y + gt[5]]])
    m = centroids_to_metres(pts, srs)
    return float(np.linalg.norm(m[1] - m[0]) * np.linalg.norm(m[2] - m[0]))

def compute_plant_traits(raster_path, rois_shp, output_shp, index=None, threshold=None, progress_callback=None, margin=0.0,
                         bands=None):
    """Write plant outlines with traits to output_shp and the same table to a .csv next to it.
    index: index for the plant mask (None: NDVI with a near infrared band, else RGBVI).
    margin: fraction of the box size added on each side (0: the outline stays inside the box).
    bands: band numbers by name (spectral_indices.find_bands; None: found from the band names)."""
    import csv
    from osgeo import gdal, ogr, osr
    from processing.spectral_indices import (INDICES, find_bands, available_indices, default_index, read_bands,
                                  compute_index, NIR_INDICES)
    raster = gdal.Open(raster_path)
    if raster is None:
        raise FileNotFoundError(f"Could not open raster: {raster_path}")
    if raster.RasterCount < 3:
        raise ValueError("Plant traits need a raster with at least 3 bands (R, G, B).")
    bands = bands or find_bands(raster_path)
    index = index or default_index(bands)
    index = next((n for n in INDICES if n.upper() == index.upper()), index)  # "ndvi" -> "NDVI"
    if index not in available_indices(bands):
        raise ValueError(f"Index {index} cannot be computed on this raster; available: {', '.join(available_indices(bands))}.")
    extra = [n for n in NIR_INDICES if n in available_indices(bands)]  # canopy means written as columns
    print(f"Plant traits: bands {bands}, mask index {index}" + (f", also {', '.join(extra)}" if extra else ""))
    gt = raster.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)
    raster_srs = osr.SpatialReference(wkt=raster.GetProjection())
    W, H = raster.RasterXSize, raster.RasterYSize

    ds_in = ogr.Open(rois_shp)
    if ds_in is None:
        raise FileNotFoundError(f"Could not open ROIs shapefile: {rois_shp}")
    layer = ds_in.GetLayer()
    srs = layer.GetSpatialRef()
    defn = layer.GetLayerDefn()
    keep_fields = [n for n in ("crop_id", "grid_id", "row", "col", "status", "score", "class", "class_name")
                   if defn.GetFieldIndex(n) >= 0]
    rois = []
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        minx, maxx, miny, maxy = geom.GetEnvelope()
        # Box extent in raster pixels (x0, y0 inclusive; x1, y1 exclusive), clipped to the raster
        px = sorted([inv_gt[0] + inv_gt[1] * minx + inv_gt[2] * maxy, inv_gt[0] + inv_gt[1] * maxx + inv_gt[2] * miny])
        py = sorted([inv_gt[3] + inv_gt[4] * minx + inv_gt[5] * maxy, inv_gt[3] + inv_gt[4] * maxx + inv_gt[5] * miny])
        # Add a margin, since part of the canopy often falls outside the detected box
        mx_px, my_px = margin * (px[1] - px[0]), margin * (py[1] - py[0])
        x0, x1 = max(0, int(np.floor(px[0] - mx_px))), min(W, int(np.ceil(px[1] + mx_px)))
        y0, y1 = max(0, int(np.floor(py[0] - my_px))), min(H, int(np.ceil(py[1] + my_px)))
        if x1 <= x0 or y1 <= y0:
            continue
        fields = {n: feat.GetField(n) for n in keep_fields}
        rois.append({"fields": fields, "empty": fields.get("status") == "empty",
                     "box": (x0, y0, x1, y1), "centre": ((px[0] + px[1]) / 2, (py[0] + py[1]) / 2),
                     "wkt": geom.ExportToWkt()})
    ds_in = None
    n = len(rois)
    if n == 0:
        raise ValueError("No ROIs found in the shapefile.")

    # Steps 1 and 2: read each window once and compute the index and valid pixels
    for i, roi in enumerate(rois):
        x0, y0, x1, y1 = roi["box"]
        arrays, valid = read_bands(raster, bands, x0, y0, x1 - x0, y1 - y0)  # black background is not data
        roi["index"], roi["valid"] = compute_index(index, arrays), valid
        roi["extra"] = {name: compute_index(name, arrays) for name in extra}
        if progress_callback and i % 100 == 0:
            progress_callback({"progress": 5 + 45 * i / n, "status": f"Plant traits: reading plant {i} of {n}"})

    # Step 3: threshold per field
    crop_ids = np.array([roi["fields"].get("crop_id") or 0 for roi in rois])
    thresholds = {}
    for crop in np.unique(crop_ids):
        if threshold is not None:
            thresholds[crop] = float(threshold)
        else:
            values = [rois[i]["index"][rois[i]["valid"]] for i in np.flatnonzero(crop_ids == crop) if not rois[i]["empty"]]
            thresholds[crop] = otsu_threshold(np.concatenate(values)) if values else 0.0
        print(f"Plant traits: field {crop}, {index} threshold {thresholds[crop]:.3f}"
              f" ({'fixed' if threshold is not None else 'Otsu'})")

    # Pixel area in m2 (the field is small, so one value at its centre is enough)
    cx_all = np.mean([r["centre"][0] for r in rois])
    cy_all = np.mean([r["centre"][1] for r in rois])
    mx = gt[0] + gt[1] * cx_all + gt[2] * cy_all
    my = gt[3] + gt[4] * cx_all + gt[5] * cy_all
    px_m2 = pixel_area_m2(gt, srs or raster_srs, mx, my)

    boxes = np.array([r["box"] for r in rois])
    centres = np.array([r["centre"] for r in rois])
    is_plant = np.array([not r["empty"] for r in rois])

    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_shp):
        driver.DeleteDataSource(output_shp)
    ds_out = driver.CreateDataSource(output_shp)
    layer_out = ds_out.CreateLayer("traits", srs, ogr.wkbPolygon)
    integer_fields = ("crop_id", "grid_id", "row", "col", "class", "in_grid")
    extra_fields = [f"{name.lower()}_mean" for name in extra]  # ndvi_mean, ndre_mean, gndvi_mean
    out_fields = keep_fields + ["in_grid", "canopy_m2", "cell_m2", "cover_pct", "idx_mean", "idx_name", "idx_thr"] + extra_fields
    for name in out_fields:
        if name in integer_fields:
            field = ogr.FieldDefn(name, ogr.OFTInteger)
        elif name in ("idx_name", "status", "class_name"):
            field = ogr.FieldDefn(name, ogr.OFTString)
        else:
            field = ogr.FieldDefn(name, ogr.OFTReal)
            field.SetWidth(12)
            field.SetPrecision(4)
        layer_out.CreateField(field)

    rows_csv = []
    for i, roi in enumerate(rois):
        x0, y0, x1, y1 = roi["box"]
        # Step 4: Voronoi cell, the pixels of this box closer to its centre than to the centre of
        # any other box that also covers them
        yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
        d_own = (xx - centres[i, 0]) ** 2 + (yy - centres[i, 1]) ** 2
        # Empty plots have no plant: canopy 0, and they take no pixels from the plants around them
        cell = roi["valid"].copy() if not roi["empty"] else np.zeros_like(roi["valid"])
        others = np.flatnonzero(is_plant & (boxes[:, 0] < x1) & (boxes[:, 2] > x0) & (boxes[:, 1] < y1) & (boxes[:, 3] > y0))
        for j in others:
            if j == i:
                continue
            bx0, by0, bx1, by1 = boxes[j]
            covered = (xx >= bx0) & (xx < bx1) & (yy >= by0) & (yy < by1)
            d_j = (xx - centres[j, 0]) ** 2 + (yy - centres[j, 1]) ** 2
            cell &= ~(covered & (d_j < d_own))

        # Step 5: plant pixels in the cell; a small closing joins leaves split by shadows, then the
        # largest connected group is kept and its holes are filled (soil seen between leaves)
        thr = thresholds[crop_ids[i]]
        plant = (cell & (roi["index"] > thr)).astype(np.uint8)
        plant = cv.morphologyEx(plant, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8)) & cell
        count, labels, stats, _ = cv.connectedComponentsWithStats(plant, connectivity=8)
        canopy = np.zeros_like(plant)
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
            contours, _ = cv.findContours((labels == largest).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(canopy, contours, -1, 1, thickness=cv.FILLED)
            canopy &= cell.astype(np.uint8)

        # Step 6: outline and traits
        canopy_px, cell_px = int(canopy.sum()), int(cell.sum())
        geom = None
        if canopy_px > 0:
            contours, _ = cv.findContours(canopy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv.contourArea).reshape(-1, 2)
            if len(contour) >= 3:
                ring = ogr.Geometry(ogr.wkbLinearRing)
                for cx, cy in np.vstack([contour, contour[:1]]):
                    gx, gy = x0 + cx + 0.5, y0 + cy + 0.5
                    ring.AddPoint_2D(gt[0] + gt[1] * gx + gt[2] * gy, gt[3] + gt[4] * gx + gt[5] * gy)
                geom = ogr.Geometry(ogr.wkbPolygon)
                geom.AddGeometry(ring)
        if geom is None:
            geom = ogr.CreateGeometryFromWkt(roi["wkt"])  # no plant pixels: keep the box

        values = dict(roi["fields"])
        values.update({
            "in_grid": 1 if (values.get("grid_id") or 0) > 0 else 0,
            "canopy_m2": round(canopy_px * px_m2, 4),
            "cell_m2": round(cell_px * px_m2, 4),
            "cover_pct": round(100.0 * canopy_px / cell_px, 2) if cell_px else 0.0,
            "idx_mean": round(float(roi["index"][canopy > 0].mean()), 4) if canopy_px else 0.0,
            "idx_name": index,
            "idx_thr": round(thr, 4),
        })
        for name, field_name in zip(extra, extra_fields):
            values[field_name] = round(float(roi["extra"][name][canopy > 0].mean()), 4) if canopy_px else 0.0
        feat = ogr.Feature(layer_out.GetLayerDefn())
        for name in out_fields:
            if values.get(name) is not None:
                feat.SetField(name, values[name])
        feat.SetGeometry(geom)
        layer_out.CreateFeature(feat)
        centroid = geom.Centroid()
        values["x"], values["y"] = round(centroid.GetX(), 8), round(centroid.GetY(), 8)
        rows_csv.append(values)
        if progress_callback and i % 100 == 0:
            progress_callback({"progress": 50 + 45 * i / n, "status": f"Plant traits: plant {i} of {n}"})
    ds_out = None
    raster = None

    # CSV with the same table, sorted by field and plot number, ready to join to the field book
    csv_path = os.path.splitext(output_shp)[0] + ".csv"
    rows_csv.sort(key=lambda v: (v.get("crop_id") or 0, (v.get("grid_id") or 0) == 0, v.get("grid_id") or 0))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields + ["x", "y"])
        writer.writeheader()
        writer.writerows(rows_csv)
    print(f"Plant traits: {int(is_plant.sum())} plants, {n - int(is_plant.sum())} empty plots, pixel area {px_m2 * 1e4:.2f} cm2, CSV: {csv_path}")
    return output_shp, csv_path
