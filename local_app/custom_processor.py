import os
import onnxruntime as ort
import cv2 as cv
import numpy as np
import pandas as pd
import shapely.geometry
from shapely.geometry import Polygon

from osgeo import gdal, ogr, osr
import datetime

from interface.batchprocessor import BatchProcessor
import glob

MODEL_PATH = "./models"
#MODEL_PATH = r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\models"

MODEL_FILENAME = "forages_rois_yolo_full_1024.onnx"

# Preprocess image
def preprocess(np_img, imgsz=1024):

    img = np_img
    
    h0, w0 = img.shape[:2]
    r = imgsz / max(h0, w0)
    new_size = (int(w0 * r), int(h0 * r))
    resized = cv.resize(img, new_size, interpolation=cv.INTER_LINEAR)

    # Padding
    padded = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized

    #cv.imwrite("padded_img.jpg", cv.cvtColor(padded, cv.COLOR_RGB2BGR))

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

def outputs_to_df(outputs):
    """
    Convert bounding boxes from the padded & resized image back to the original image coordinates.
    
    Parameters:
        bboxes (list): List of bounding boxes as [x1, y1, x2, y2, score].
        classes (list): List of corresponding class labels.
        original_size (tuple): (height, width) of the original image.
        target_size (tuple): (target_height, target_width) used for resizing/padding.
    
    Returns:
        DataFrame: A pandas DataFrame with columns [xmin, ymin, xmax, ymax, score, class].
    """

    bboxes = outputs[0][0]
    classes = outputs[1][0]

    results = []
    for bbox, label in zip(bboxes, classes):
        x1, y1, x2, y2, score = bbox

        
        results.append({
            "xmin": x1,
            "ymin": y1,
            "xmax": x2,
            "ymax": y2,
            "score": score,
            "class": label
        })
        
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(results)
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
from sklearn.decomposition import PCA


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
    pca = PCA(n_components=2)
    pca.fit(points)
    axes = pca.components_

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

def polygon_nms_gdal(wkt_list, fields_list, iou_threshold=0.5):
    import shapely.wkt
    polys = [shapely.wkt.loads(wkt) for wkt in wkt_list]
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

def rotate_polygon_to_pca_axes(polygon, centroid, axes):
    coords = np.array(polygon.exterior.coords) - centroid
    R = axes
    rotated_coords = coords @ R.T
    rotated_coords += centroid
    from shapely.geometry import Polygon
    return Polygon(rotated_coords)

# --- Main pipeline ---
def label_polygons_from_shapefile(input_shp, output_shp, serpentine=False, row_tol=10,
                                   iou_thresh=0.3, min_ratio=0.2, max_ratio=5.0, align_to_grid=False, only_postprocess=False):
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
    
    for feat in layer_in:
        geom = feat.GetGeometryRef()
        if geom:
            wkt_list.append(geom.ExportToWkt())
            fields_list.append([feat.GetField(i) for i in range(field_count)])
            
    ds_in = None
    
    wkt_list, fields_list = filter_by_aspect_ratio(wkt_list, fields_list, min_ratio, max_ratio)
    wkt_list, fields_list = polygon_nms_gdal(wkt_list, fields_list, iou_thresh)
    
    if not only_postprocess and wkt_list:
        polys = [shapely.wkt.loads(wkt) for wkt in wkt_list]
        centroids = np.array([poly.centroid.coords[0] for poly in polys])
        
        clean_centroids = remove_outlier_centroids(centroids, threshold=4.0)
        angle = estimate_grid_angle(clean_centroids)
        
        projected = project_to_grid_axes_angle(centroids, angle)
        rows = group_rows_cols(projected, row_tol=1.0)
        idx_map = assign_indices(rows, serpentine)
        
        labels = []
        for c in centroids:
            proj_c = project_to_grid_axes_angle(np.array([c]), angle, center=centroids.mean(axis=0))[0]
            best_match = min(idx_map.keys(), key=lambda k: np.linalg.norm(np.array(k) - proj_c))
            labels.append(idx_map[best_match])
            
        combined = list(zip(labels, wkt_list, fields_list))
        combined.sort(key=lambda x: x[0])
        labels, wkt_list, fields_list = zip(*combined) if combined else ([], [], [])
        
    if os.path.exists(output_shp):
        driver.DeleteDataSource(output_shp)
    ds_out = driver.CreateDataSource(output_shp)
    layer_out = ds_out.CreateLayer("labeled", srs, ogr.wkbPolygon)
    
    for i in range(field_count):
        layer_out.CreateField(layer_defn.GetFieldDefn(i))
    if not only_postprocess:
        layer_out.CreateField(ogr.FieldDefn("grid_id", ogr.OFTInteger))
        
    for idx, (wkt, fields) in enumerate(zip(wkt_list, fields_list)):
        geom = ogr.CreateGeometryFromWkt(wkt)
        feat = ogr.Feature(layer_out.GetLayerDefn())
        for i, val in enumerate(fields):
            feat.SetField(i, val)
        if not only_postprocess:
            feat.SetField("grid_id", labels[idx])
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

def tile_raster_gdal(input_raster_path, tiles_dir, tile_size, prefix=""):
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
    tiles_x = int(np.ceil(width / tile_size))
    tiles_y = int(np.ceil(height / tile_size))
    os.makedirs(tiles_dir, exist_ok=True)
    tile_files = []
    tile_count = 0
    nodata_value = -9999
    for row in range(tiles_y):
        for col in range(tiles_x):
            tile_filename = os.path.join(tiles_dir, f"{prefix}{tile_count:05d}.tif")
            if os.path.exists(tile_filename):
                tile_files.append(tile_filename)
                tile_count += 1
                continue
            x_offset = col * tile_size
            y_offset = row * tile_size
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

    def initialize(self):

        #Load model
        if self.ort_sess is None:            

            model_filepath = os.path.join(MODEL_PATH, MODEL_FILENAME)

            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    # Optional: additional options can be provided, e.g.
                    #"gpu_mem_limit":  * 1024 * 1024 * 1024,
                    #"gpu_mem_limit":  6 * 1024,
                    # "cudnn_conv_algo_search": "EXHAUSTIVE",
                    # "do_copy_in_default_stream": True,
                })
            ]

            self.ort_sess = ort.InferenceSession(model_filepath, providers=providers)

            # self.ort_sess = ort.InferenceSession(model_filepath
            #                     ,providers=ort.get_available_providers()
            #                     )

        return
    
    def inference(self, filepath, output_folder=None):

        self.initialize()

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
            with rio.open(filepath) as src:
                bounds = src.bounds
                extent = bounds  # (left, bottom, right, top)
                crs = src.crs
                if crs:
                    if crs.is_epsg_code:
                        epsg = crs.to_string().replace("EPSG:EPSG:", "EPSG:")
                    elif crs.to_epsg() is not None:
                        epsg = f"EPSG:{crs.to_epsg()}"
                        epsg = crs.to_string().replace("EPSG:EPSG:", "EPSG:")
                    else:
                        raise ValueError(f"Could not determine EPSG code for file: {filepath}")
                else:
                    raise ValueError(f"No CRS found in raster file: {filepath}")
        else:
            # extent and epsg must be provided or set elsewhere for non-tif images
            raise ValueError("EPSG code must be provided for non-tif images.")
        
        epsg = crs.to_string().replace("EPSG:EPSG:", "EPSG:")
        epsg = crs.to_string().replace("EPSG:", "")
        print(epsg)

        img_prec, scale, (h0, w0) = preprocess(np_image)
        outputs = self.ort_sess.run(None, {'images':img_prec})
        outputs = postprocess_yolo_output(outputs, conf_threshold=0.26, nms_threshold=0.2, orig_shape=(1024,1024))
        boxes_df = outputs_to_df(outputs)

        if is_raster:

            shp_bbox = os.path.join(output_folder, basename + "_boxes.shp")

            print("Saving shapefile to", shp_bbox)

            save_shapefile_bb(boxes_df,
                                extent,
                                np_image.shape[1],
                                np_image.shape[0],
                                epsg,
                                allow_cols=["score","class"]
                                , output_filename=shp_bbox
                                )
        else:

            #Draw boxes on the image


            # csv fileanme
            csv_filename = os.path.join(output_folder, basename + "_boxes.csv")

            print("Saving csv file to", csv_filename)
            # Save dataframe boxes_df
            boxes_df.to_csv(csv_filename, index=False)

    def batch_processing(self, folder, output_folder, format="tif"
                        , progress_callback=None
                        , interruption_check=None
                        ):

        processor = BatchProcessor()

        def processFunction(filepath, output_files):

            if os.path.exists(output_files[0]):
                print(f"File already exists {output_files[0]}")
            else:
                print(output_files[0])
                output_dir = os.path.dirname(output_files[0])

                self.inference(filepath, output_dir)
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
    def tile_inference(self, input_filepath, output_filepath, only=False, clean_cache=False):

        # Get basename without extension
        basename = os.path.splitext(os.path.basename(output_filepath))[0]

        # Set output folder as filepath dir
        output_folder = os.path.dirname(output_filepath)
        
        try:
            from cachemanager import CacheManager
        except ImportError:
            from .cachemanager import CacheManager
        cache_manager = CacheManager(project_path=output_folder)
        
        tile_size = 1024
        
        cache_key = {
            "input_raster_path": input_filepath,
            "tile_size": tile_size
        }
        key = cache_manager.compute_key(cache_key)
        
        images_dir = cache_manager.get_cache_folder_path("tiles", key)
        shp_dir = cache_manager.get_cache_folder_path("shp", key)
        
        os.makedirs(output_folder, exist_ok=True)

        # Extract tiles and save using GDAL
        if not cache_manager.exists("tiles", key):
            print(f"Extracting tiles from {input_filepath} to cache...")
            tile_raster_gdal(input_filepath, images_dir, tile_size, prefix="tile_")
        else:
            print(f"Using cached tiles from {images_dir}")

        # Process each tile
        self.batch_processing(images_dir, shp_dir)

        # Merge all shapefiles in shp_dir and save
        shp_files = glob.glob(os.path.join(shp_dir, "*.shp"))
        print(f"Merging {len(shp_files)} files")

        if shp_files:
            merged_dir = cache_manager.get_cache_folder_path("merged", key)
            merged_shp = os.path.join(merged_dir, "merged.shp")
            merge_shp_gdal(shp_files, merged_shp)

            # Post process the merged shapefile
            safe_path = os.path.normpath(output_filepath)
            if not only:
                label_polygons_from_shapefile(merged_shp, safe_path, serpentine=True, row_tol=1.0, min_ratio=1/1.8, max_ratio=1.8, iou_thresh=0.15, align_to_grid=False)
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
                cache_manager.clean_cache_folder_path("shp", key)
                cache_manager.clean_cache_folder_path("merged", key)
        else:
            print("No shapefiles found to merge in", shp_dir)

    def plot_numbering(self, input_filepath, output_filepath, serpentine=True, align_to_grid=False, only_postprocess=False):
        safe_input_filepath = os.path.normpath(input_filepath)
        safe_input_output_filepath = os.path.normpath(output_filepath)        
        
        # Post process the merged shapefile
        label_polygons_from_shapefile(safe_input_filepath, safe_input_output_filepath, serpentine=serpentine, row_tol=1.0, min_ratio=1/1.8, max_ratio=1.8, iou_thresh=0.15, align_to_grid=align_to_grid, only_postprocess=only_postprocess)



