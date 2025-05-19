import os
import onnxruntime as ort
import cv2 as cv
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry
from shapely.geometry import Polygon

import rasterio as rio
from rasterio.crs import CRS
import datetime

from rasterio.mask import mask
from rasterio.enums import Resampling

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
    # extent is a rasterio BoundingBox: left, bottom, right, top
    left, bottom, right, top = extent.left, extent.bottom, extent.right, extent.top
    extent_width = right - left
    extent_height = top - bottom
    x = (pos[0]) / img_width
    y = 1.0 - (pos[1]) / img_height
    coord_x = x * extent_width + left
    coord_y = y * extent_height + bottom
    return (coord_x, coord_y)


def save_shapefile_bb(df, extent, img_width, img_height, epsg, allow_cols=[], output_filename=None):
    print(type(df))
    if type(df) == "NoneType":
        print("No results")
        return

    df_tree_polygons_test = pd.DataFrame()
    tree_bb = []
    count = 0

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
        #print("len", len(coord_polygon))
        if len(coord_polygon) > 2:
            polygon_object = shapely.geometry.Polygon(coord_polygon)
            #mydic = {'Class': 'Tree', 'ID': count, 'label': 'Tree'}
            mydic = {'Type': 'forage_plant'}
            for col in allow_cols:
                mydic[col] = detection[col]
            df_item_test = pd.DataFrame(mydic, index=[count])
            df_tree_polygons_test = pd.concat((df_tree_polygons_test, df_item_test))
            tree_bb.append(polygon_object)
            count = count + 1

    # Create geodataframe
    gdf_trees = gpd.GeoDataFrame(df_tree_polygons_test, geometry=tree_bb)
    if epsg is not None:
        gdf_trees = gdf_trees.set_crs(epsg=epsg)
    print(gdf_trees)

    # Area calculation (optional, only if extent is valid)
    if extent is not None:
        lon = extent.left
        lat = extent.bottom
        new_crs = f"+proj=cea +lat_0={lat} +lon_0={lon} +units=m"
        gdf_trees["area_m2"] = gdf_trees.to_crs(new_crs).area
        #gdf_trees["a_diam_m"] = np.sqrt(gdf_trees["area_m2"] * 4.0 / np.pi)

    if output_filename is not None:
        gdf_trees_bb = gdf_trees.copy()
        print("HERE")
        print(output_filename)        
        if gdf_trees_bb.empty:
        # Ensure at least the geometry column exists with correct type
            gdf_trees_bb = gpd.GeoDataFrame(columns=gdf_trees_bb.columns, geometry='geometry', crs=gdf_trees_bb.crs)
        safe_path = os.path.normpath(output_filename)
        gdf_trees_bb.to_file(safe_path, index=False)

    # ...existing code for saving outputs...
    # if "bounding_boxes" in self.parameters["vector_outputs"]:
    #     gdf_trees_bb = gdf_trees.copy()
    #     new_output_filename = output_filename.replace("_vector.shp", "_vector_bb.shp")
    #     gdf_trees_bb.to_file(new_output_filename, index=False)
    #     if not new_output_filename in self.output_files:
    #         self.output_files.append(new_output_filename)

    # if "centroids" in self.parameters["vector_outputs"]:
    #     gdf_trees_c = gdf_trees.copy()
    #     gdf_trees_c['geometry'] = gdf_trees_c['geometry'].centroid
    #     new_output_filename = output_filename.replace("_vector.shp", "_vector_centroids.shp")
    #     gdf_trees_c.to_file(new_output_filename, index=False)
    #     if not new_output_filename in self.output_files:
    #         self.output_files.append(new_output_filename)


def check_raster(input_file):

    metadata = {}

    with rio.open(input_file) as src:
        metadata["width"] = src.width  # Image width (pixels)
        metadata["height"] = src.height  # Image height (pixels)

        return metadata
    
## POSTPROCESSING FUNCTIONS
import numpy as np
import geopandas as gpd
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
def filter_by_aspect_ratio(gdf, min_ratio=0.2, max_ratio=5.0):
    """Remove geometries whose bounding boxes are too elongated."""
    def is_valid_bbox(poly):
        minx, miny, maxx, maxy = poly.bounds
        w, h = maxx - minx, maxy - miny
        if h == 0 or w == 0:
            return False
        ratio = w / h
        return min_ratio <= ratio <= max_ratio

    return gdf[gdf.geometry.apply(is_valid_bbox)].copy()

def compute_iou(boxA, boxB):
    inter = boxA.intersection(boxB).area
    union = boxA.union(boxB).area
    return inter / union if union > 0 else 0

# def nms_polygons(gdf, iou_thresh=0.7):
#     """Apply NMS based on bounding box IoU."""
#     gdf = gdf.copy()
#     boxes = [box(*geom.bounds) for geom in gdf.geometry]
#     scores = np.array([geom.area for geom in gdf.geometry])  # or other score
#     indices = scores.argsort()[::-1]

#     keep = []
#     suppressed = set()

#     count = 0
#     for i in indices:
#         count += 1
#         print(f"Processing {count}/{len(indices)}")
#         if i in suppressed:
#             continue
#         keep.append(i)
#         for j in indices:
#             if j == i or j in suppressed:
#                 continue
#             iou = compute_iou(boxes[i], boxes[j])
#             if iou > iou_thresh:
#                 suppressed.add(j)
#     return gdf.iloc[keep].copy()

def nms_polygons(gdf, iou_thresh=0.7):
    """Apply NMS based on bounding box IoU, optimized with spatial index."""
    gdf = gdf.copy()
    boxes = [box(*geom.bounds) for geom in gdf.geometry]
    scores = np.array([geom.area for geom in gdf.geometry])  # or other score
    indices = scores.argsort()[::-1]

    keep = []
    suppressed = set()
    sindex = gdf.sindex  # spatial index for fast bbox queries

    for i in indices:
        if i in suppressed:
            continue
        keep.append(i)
        # Only check for overlap with candidates whose bboxes intersect
        candidates = list(sindex.intersection(boxes[i].bounds))
        for j in candidates:
            if j == i or j in suppressed:
                continue
            iou = compute_iou(boxes[i], boxes[j])
            if iou > iou_thresh:
                suppressed.add(j)
    return gdf.iloc[keep].copy()

def rotate_polygon_to_pca_axes(polygon, centroid, axes):
    """
    Rotates a shapely polygon so its local axes align with the PCA axes.
    Args:
        polygon: shapely.geometry.Polygon
        centroid: (x, y) tuple, the centroid of the polygon
        axes: 2x2 numpy array, PCA axes (each row is a unit vector)
    Returns:
        shapely.geometry.Polygon, rotated polygon centered at the centroid
    """
    # Translate polygon to origin
    coords = np.array(polygon.exterior.coords) - centroid
    # Build rotation matrix from PCA axes (axes[0] is new x, axes[1] is new y)
    R = axes
    # Rotate
    rotated_coords = coords @ R.T
    # Translate back to centroid
    rotated_coords += centroid
    return Polygon(rotated_coords)

# Example usage after PCA and axes computation:
# centroids = compute_centroids(gdf)
# axes = compute_pca_axes(centroids)
# for i, row in gdf.iterrows():
#     poly = row.geometry
#     centroid = poly.centroid.coords[0]
#     gdf.at[i, "geometry_rotated"] = rotate_polygon_to_pca_axes(poly, centroid, axes)


# --- Main pipeline ---
def label_polygons_from_shapefile(gdf, output_path=None, serpentine=False, row_tol=10,
                                   iou_thresh=0.3, min_ratio=0.2, max_ratio=5.0, align_to_grid=False, only_postprocess=False):
    # Save original CRS
    orig_crs = gdf.crs
    reproj_for_pca = False
    utm_crs = None

    # If CRS is geographic (degrees), reproject to UTM for PCA/grouping
    if gdf.crs.is_geographic:
        # Compute centroid longitude to pick UTM zone
        centroid = gdf.unary_union.centroid
        lon = centroid.x
        lat = centroid.y
        utm_zone = int((lon + 180) // 6) + 1
        is_northern = lat >= 0
        utm_crs = f"EPSG:{32600 + utm_zone if is_northern else 32700 + utm_zone}"
        gdf = gdf.to_crs(utm_crs)
        reproj_for_pca = True


    print(f"Filtering by aspect ratio {min_ratio} < aspect ratio < {max_ratio}...")
    gdf = filter_by_aspect_ratio(gdf, min_ratio, max_ratio)
    print(f"Applytin non-max suppression with threshold {iou_thresh}...")
    gdf = nms_polygons(gdf, iou_thresh)

    if not only_postprocess:

        print("Computing centroids...")
        centroids = compute_centroids(gdf)

        # # Save projected points as a shapefile for visualization/debugging
        # projected_points_geom = [shapely.geometry.Point(pt[0], pt[1]) for pt in centroids]
        # projected_gdf = gpd.GeoDataFrame(geometry=projected_points_geom, crs=gdf.crs)
        # projected_gdf.to_file("./local/centroid_points.shp", index=False)

        print("Removing outliers...")
        clean_centroids = remove_outlier_centroids(centroids, threshold=4.0) # 2 standard deviations
        print(f"Computing PCA axes...")
        #axes, angle = compute_pca_axes(clean_centroids)
        angle = estimate_grid_angle(clean_centroids)


        if align_to_grid:
            print("Aligning polygons to grid...")
            # rotate polygons to PCA axes and update geometry in place
            for i, row in gdf.iterrows():
                poly = row.geometry
                centroid = poly.centroid.coords[0]
                #gdf.at[i, "geometry"] = rotate_polygon_to_pca_axes(poly, centroid, axes)

        print("Projecting centroids to grid axes...")
        #projected = project_to_grid_axes(centroids, axes)
        projected = project_to_grid_axes_angle(centroids, angle)
        
        # # Save projected points as a shapefile for visualization/debugging
        # projected_points_geom = [shapely.geometry.Point(pt[0], pt[1]) for pt in projected]
        # projected_gdf = gpd.GeoDataFrame(geometry=projected_points_geom, crs=gdf.crs)
        # projected_gdf.to_file("./local/projected_points.shp", index=False)

        print("Grouping projected points into rows...")
        rows = group_rows_cols(projected, row_tol=1.0)
        print(f"Assigning indices to rows...")
        idx_map = assign_indices(rows, serpentine)

        print("Assigning numbering to polygons...")
        labels = []
        projected_points_geom = []
        for i, geom in enumerate(gdf.geometry):
            #print(f"processing {i}")
            c = np.array(geom.centroid.coords[0])
            #proj_c = np.dot(c, axes.T)
            proj_c = project_to_grid_axes_angle(c, angle, center=centroids.mean(axis=0))
            projected_points_geom.append(shapely.geometry.Point(proj_c[0], proj_c[1]))

            best_match = min(idx_map.keys(), key=lambda k: np.linalg.norm(np.array(k) - proj_c))
            labels.append(idx_map[best_match])

        # # Save projected points as a shapefile for visualization/debugging
        # projected_gdf = gpd.GeoDataFrame(geometry=projected_points_geom, crs=gdf.crs)
        # projected_gdf.to_file("./local/centroid_ordering.shp", index=False)

        gdf = gdf.copy()
        gdf["grid_id"] = labels
        #reorder the dataframe by grid_id
        gdf = gdf.sort_values(by=["grid_id"])

        # Save projected points as a shapefile for visualization/debugging
        projected_points_geom = [shapely.geometry.Point(pt[0], pt[1]) for pt in projected]
        projected_gdf = gpd.GeoDataFrame(geometry=projected_points_geom, crs=gdf.crs)
        projected_gdf["grid_id"] = labels
        projected_gdf = projected_gdf.sort_values(by=["grid_id"])

        if reproj_for_pca and orig_crs is not None:
            projected_gdf = projected_gdf.to_crs(orig_crs)

        projected_gdf.to_file("./local/projected_points.shp", index=False)        

    # Reproject back to original CRS if we changed it
    if reproj_for_pca and orig_crs is not None:
        gdf = gdf.to_crs(orig_crs)

    if output_path:        
        gdf.to_file(output_path)
        

    return gdf

class TILER():

    def __init__(self, path_raster, path_vector, category="tree", supercategory="tree"
            ,allow_clipped_annotations = True, allow_no_annotations=True, class_column = [], invalid_class=[]
            , preffix = 'tile_', crs = "6933", license = None, information = None, contributor = None, license_url = None
            , output_format = ".tif"
        ):

        self.path_raster = path_raster # geotiff data
        self.path_vector = path_vector # a geopandas dataframe
        
        self.path_output = None
        self.path_annotations = None
        self.path_images = None

        self.raster = None
        self.vector = None

        self.gdf = None # geopandas dataframe with bounding box of raster file

        self.coco_images = None

        self.temp_file = None

        #self.crs = "4326"
        #self.crs = "6933" #units in meters
        self.crs = crs

        # Load files
        self.load_files()

        self.supercategory = supercategory
        self.category = category
        self.allow_clipped_annotations = allow_clipped_annotations
        self.allow_no_annotations = allow_no_annotations

        self.class_column = class_column
        self.invalid_class = invalid_class

        self.preffix = preffix

        self.license = license
        self.information = information
        self.contributor = contributor
        self.license_url = license_url

        self.output_format = output_format

    def load_files(self):

        self.raster = rio.open(self.path_raster)

        #Create bounding box
        image_geo = self.raster
        bb = [(image_geo.bounds[0], image_geo.bounds[3])
                ,(image_geo.bounds[2], image_geo.bounds[3])
                ,(image_geo.bounds[2], image_geo.bounds[1])
                ,(image_geo.bounds[0], image_geo.bounds[1])]

        self.gdf = gpd.GeoDataFrame(geometry=[shapely.geometry.Polygon(bb)])
        #self.gdf = self.set_crs(epsg="3857")
        self.gdf = self.gdf.set_crs(self.raster.crs)
        self.gdf = self.gdf.to_crs(epsg=self.crs)

    def create_grid(self, rows0, w_overlap=0, h_overlap=0):

        cols0 = rows0

        xmin, ymin, xmax, ymax = self.gdf.total_bounds

        Dh = abs(ymax - ymin)
        # oh = Dh*h_overlap
        # tile_h = (Dh - oh)/rows0 + oh

        tile_h = Dh/(rows0-rows0*h_overlap+h_overlap)
        oh = tile_h*h_overlap

        Dw = abs(xmax - xmin)
        # ow = Dw*w_overlap
        # tile_w = (Dw - ow)/cols0 + ow

        tile_w = Dw/(cols0-cols0*w_overlap+w_overlap)
        ow = tile_w*w_overlap


        # # Use only the value for the max 
        # if Dw > Dh and Dh/Dw > 0.7:
        #     tile_h = tile_w
        #     oh = ow
        # elif Dh > Dw and Dw/Dh > 0.7:
        #     tile_w = tile_h
        #     ow = oh

        # # Use only the value for the max 
        if Dw > Dh:
            tile_h = tile_w
            oh = ow
        elif Dh > Dw:
            tile_w = tile_h
            ow = oh

        #tile_h = abs(ymax - ymin)/rows0
        #tile_w = abs(xmax - xmin)/cols0

        #cols = list(np.arange(xmin, xmax, tile_w))
        #rows = list(np.arange(ymax, ymin, - tile_h))

        #print("TILE SIZE")
        #print(tile_h)
        #print(tile_w)

        cols = list(np.arange(xmin, xmax-(ow), tile_w - ow))
        rows = list(np.arange(ymax, ymin-(-(oh)), - (tile_h-oh)))

        #print(cols)
        #print(rows)

        df_grid = pd.DataFrame()

        polygons = []
        count = 0
         
        for y in rows:
            for x in cols:

                poly = Polygon([(x,y), (x+tile_w, y), (x+tile_w, y- tile_h), (x, y- tile_h)])

                polygons.append(poly)

                df_item = pd.DataFrame({'id': count}, index=[count])
                df_grid = pd.concat((df_grid, df_item))

                count = count + 1

        #self.grid = gpd.GeoDataFrame(df_item, {'geometry':polygons})
        self.grid = gpd.GeoDataFrame(df_grid, geometry=polygons)

        # Fix index error with "module 'pandas' has no attribute 'Int64Index'"
        self.grid.reset_index(drop=True, inplace=True)
        self.grid.set_index("id", inplace = True)

        #self.grid["row_id"] = self.grid.index + 1
        #self.grid.reset_index(drop=True, inplace=True)
        #self.grid.set_index("row_id", inplace = True)

        self.grid = self.grid.set_crs(epsg=self.crs, allow_override=True)
        #grid.to_file("grid.shp")

    def extract_tiles(self, scale = 1.0):

            #size = 256

            #splitImageIntoCells(self.raster, self.path_images, size)

            coco_images = []

            for i, grid_element in enumerate(self.grid.geometry):
                basename = self.preffix + str(i) + self.output_format
                filename = os.path.join(self.path_images, basename)

                if os.path.exists(filename):
                    print(f"File already exists {filename}")
                    continue


                tile, w, h = self.clip_raster(i, filename)
                

                coco_images.append({
                    "id": i+1
                    , "file_name": basename
                    , "width": w
                    , "height": h
                })   

            self.coco_images = coco_images 

            return

    def clip_raster(self, id, filename, scale = 1.0):

        #vector = self.grid.to_crs(self.raster.crs)
        vector = self.grid[id:id+1].to_crs(self.raster.crs)

        raster = self.raster

        if (scale != 1.0):
            # Resample if necessary
            # resample data to target shape
            dataset = raster
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * scale),
                    int(dataset.width * scale)
                ),
                resampling=Resampling.bilinear
            )

            # scale image transform
            transform = dataset.transform * dataset.transform.scale(
                (dataset.width / data.shape[-1]),
                (dataset.height / data.shape[-2])
            )

            #

        #tile, tile_transform = mask(self.raster, [vector.geometry[id]], crop=True)
        #tile, tile_transform = mask(raster, vector.geometry, crop=True, filled = True)
        tile, tile_transform = mask(raster, vector.geometry, crop=True)

        width = tile.shape[2]
        height = tile.shape[1]

        if "tif" in self.output_format:

            tile_meta = self.raster.meta.copy()

            tile_meta.update({
                "driver":"Gtiff",
                "height":height, # height starts with shape[1]
                "width":width, # width starts with shape[2]
                "transform":tile_transform
            })
            #print("TILE", tile.shape[1], tile.shape[2])
            with rio.open(filename, 'w', **tile_meta) as dst:
                dst.write(tile)

        else:
            
            tile = np.transpose(tile, (1,2,0))
            # print(type(tile))
            # print(tile.shape)
            # print(filename)
            tile = cv.cvtColor(tile, cv.COLOR_RGB2BGR)
            cv.imwrite(filename, tile)

        return tile, width, height
    

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

    def tile_inference(self, input_filepath, output_filepath, only=False):

        # Get basename without extension
        basename = os.path.splitext(os.path.basename(output_filepath))[0]

        # Set output folder as filepath dir
        output_folder = os.path.dirname(output_filepath)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        #output_folder = os.path.join(output_folder, f"{basename}_{timestamp}")
        output_folder = os.path.join(output_folder, f"{basename}_foragesrois_temp")
        images_dir = os.path.join(output_folder, "tiles")
        shp_dir = os.path.join(output_folder, "shp")
        

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(shp_dir, exist_ok=True)

        # tiling
        converter = TILER(input_filepath
                , ""
                , category = "category"
                , supercategory = "supercategory"
                , allow_clipped_annotations = False
                , allow_no_annotations = False
                , class_column = ["label"]
                , invalid_class=["target", "empty"]
                , preffix = ''
                , crs = "4326"
                )
        
        converter.path_images = images_dir
        
        metadata = check_raster(input_filepath)


        w = metadata["width"]
        h = metadata["height"]
        max_px = 1024
        overlap = 0.25

        rows = 1

        if (w > max_px or h > max_px):

            max_val = max(w,h)
            #print(max_val)        
            #rows = np.ceil(max_val/final_max_px)
            #rows = (max_val - np.ceil(overlap*max_val))/(max_px - np.ceil(overlap*max_val))
            rows = np.ceil((max_val-overlap*max_px)/(max_px*(1-overlap)))

        print("rows", rows)
        print("overlap", overlap)

        # Create a vector grid for each tile
        converter.create_grid(rows, overlap, overlap)

        # Extract tiles and save
        converter.extract_tiles()


        # Process each tile
        self.batch_processing(images_dir,shp_dir)

        # Merge all shapefiles in shp_dir and save
        # Find all shapefiles in shp_dir
        shp_files = glob.glob(os.path.join(shp_dir, "*.shp"))
        print(f"Merging {len(shp_files)} files")

        if shp_files:
            gdfs = [] #= [gpd.read_file(os.path.normpath(shp)) for shp in shp_files]
            for shp in shp_files:
                gdf = gpd.read_file(os.path.normpath(shp))
                if not gdf.empty:
                    gdfs.append(gdf)

            print(f"Merging {len(gdfs)} files with detections")


            merged_gdf = pd.concat(gdfs, ignore_index=True)
            merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry="geometry")


            # Post process the merged shapefile
            if not only:
                gdf_labeled = label_polygons_from_shapefile(merged_gdf, serpentine=True, row_tol=1.0, min_ratio=1/1.8, max_ratio=1.8, iou_thresh=0.15, align_to_grid=False)
            else:
                gdf_labeled = merged_gdf


            #merged_gdf.to_file(output_filepath, index=False)
            safe_path = os.path.normpath(output_filepath)
            gdf_labeled.to_file(safe_path, index=False)
        else:
            print("No shapefiles found to merge in", shp_dir)

    def plot_numbering(self, input_filepath, output_filepath, serpentine=True, align_to_grid=False,only_postprocess=False):

        safe_input_filepath = os.path.normpath(input_filepath)
        safe_input_output_filepath = os.path.normpath(output_filepath)        

        merged_gdf = gpd.read_file(safe_input_filepath)
        
        # Post process the merged shapefile
        gdf_labeled = label_polygons_from_shapefile(merged_gdf, serpentine=serpentine, row_tol=1.0, min_ratio=1/1.8, max_ratio=1.8, iou_thresh=0.15, align_to_grid=align_to_grid, only_postprocess=only_postprocess)

        gdf_labeled.to_file(safe_input_output_filepath, index=False)



