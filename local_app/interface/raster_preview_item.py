import os
import numpy as np
import cv2
from osgeo import gdal, ogr

from PySide6.QtQuick import QQuickPaintedItem
from PySide6.QtGui import QImage, QColor, QPainter, QPolygonF, QPen, QFont, QPainterPath, QFontMetricsF
from PySide6.QtCore import Property, QRectF, QPointF, Signal, Slot, Qt

# Longest side of the reduced copy of the raster kept in memory and drawn while zoomed out
BASE_SIZE = 2048
# Smallest box (screen pixels on each side) the add tool accepts, so a stray click adds nothing
MIN_BOX_PX = 4


def to_rgb8(img_data):
    """GDAL array (bands, rows, cols) or (rows, cols) -> contiguous rows x cols x 3 uint8 RGB."""
    if len(img_data.shape) == 3:
        img_data = img_data.transpose(1, 2, 0)
        if img_data.shape[2] >= 3:
            img_data = img_data[:, :, :3]
    elif len(img_data.shape) == 2:
        img_data = np.stack([img_data, img_data, img_data], axis=-1)
    if img_data.dtype != np.uint8:
        return cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return np.ascontiguousarray(img_data)


class RasterPreviewItem(QQuickPaintedItem):

    rasterWidthChanged = Signal()
    rasterHeightChanged = Signal()
    sourcePathChanged = Signal()  # view.qml binds to sourcePath (placeholder text)
    editChanged = Signal()        # erasedCount / addedCount / canUndo changed (erase and add tools, see DetectionEditor)
    shapefilePathChanged = Signal()  # detections layer changed (view.qml enables the Edit group)
    outlinePathChanged = Signal()    # plant outlines layer changed

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._sourcePath = ""
        self._shapefilePath = ""
        self._shapefileStamp = None   # modification time of the loaded shapefile (see _file_stamp)
        self._outlineStamp = None
        self._showVectorOverlay = False
        self._colorByDetection = False   # one colour per polygon instead of light blue

        self._contentX = 0.0
        self._contentY = 0.0
        self._zoomScale = 1.0
        
        self._dataset = None
        self._raster_w = 0
        self._raster_h = 0
        self._inv_gt = None
        self._base = None             # reduced copy of the whole raster (QImage), see load_raster
        self._base_scale = 0.0        # its size relative to the raster

        self._polygons = []
        self._empty = []              # True for the empty plots of the numbering (status "empty")
        self._labels = []             # (polygon index, centroid in raster pixels, text) for numbered layers
        self._label_min_zoom = 1.0    # zoom from which polygons are big enough to read labels
        self._editor = None           # DetectionEditor of the shown shapefile (erase tool)
        self._trace = []              # screen points of the stroke being traced
        self._box_drag = None         # [start, end] screen points of the box being dragged (add tool)
        self._gt = None               # geotransform of the raster (raster pixels to map coordinates)
        # Second layer: plant outlines (plant_traits output), drawn under the boxes
        self._outlinePath = ""
        self._outlines = []           # QPolygonF in raster pixels
        self._showOutlines = True

        self.setRenderTarget(QQuickPaintedItem.FramebufferObject)

    # ---- Erase tool: trace with the mouse over boxes to erase them (logic in DetectionEditor) ----

    def _to_raster(self, x, y):
        # Same origin as paint (src_x, src_y), so a stroke touches exactly the boxes drawn under it
        src_x = max(0, min(int(self._contentX / self._zoomScale), self._raster_w - 1))
        src_y = max(0, min(int(self._contentY / self._zoomScale), self._raster_h - 1))
        return src_x + x / self._zoomScale, src_y + y / self._zoomScale

    @Slot(float, float)
    def beginErase(self, x, y):
        if self._editor is None:
            return
        self._editor.begin_stroke()
        self._trace = [QPointF(x, y)]
        self.eraseTo(x, y)

    @Slot(float, float)
    def eraseTo(self, x, y):
        """Erase the boxes touched by the segment from the last point to (x, y), in screen pixels."""
        if self._editor is None or not self._trace:
            return
        last = self._trace[-1]
        x0, y0 = self._to_raster(last.x(), last.y())
        x1, y1 = self._to_raster(x, y)
        # a few screen pixels of tolerance, so a stroke near a box border still touches it
        if self._editor.erase_segment(x0, y0, x1, y1, tolerance=3.0 / self._zoomScale):
            self.editChanged.emit()
        self._trace.append(QPointF(x, y))
        self.update()

    @Slot()
    def endErase(self):
        if self._editor is not None:
            self._editor.end_stroke()
            self.editChanged.emit()
        self._trace = []
        self.update()

    @Slot()
    def undoErase(self):
        if self._editor is not None and self._editor.undo():
            self.editChanged.emit()
            self.update()

    @Slot()
    def discardEdits(self):
        """Forget the erased boxes that were not saved (reload the shapefile)."""
        self._trace = []
        self.load_shapefile()
        self.update()

    @Slot()
    def clearAll(self):
        """Empty the preview (File > New): no raster, no detections, no outlines, no edits.
        Files on disk are not touched; unsaved edits are dropped, so ask before calling this."""
        self._trace, self._box_drag = [], None
        self.outlinePath = ""
        self.shapefilePath = ""   # also drops the editor (load_shapefile)
        self.sourcePath = ""      # also drops the raster and its reduced copy (load_raster)
        self.update()

    # ---- Add tool: drag with the mouse to draw a new box (logic in DetectionEditor) ----

    @Slot(float, float)
    def beginBox(self, x, y):
        if self._editor is not None:
            self._box_drag = [QPointF(x, y), QPointF(x, y)]

    @Slot(float, float)
    def dragBoxTo(self, x, y):
        if self._box_drag:
            self._box_drag[1] = QPointF(x, y)
            self.update()

    @Slot(int)
    def endBox(self, class_id):
        """Add the dragged box with class class_id; boxes under MIN_BOX_PX screen pixels are ignored."""
        drag, self._box_drag = self._box_drag, None
        if self._editor is not None and drag:
            a, b = drag
            if abs(b.x() - a.x()) >= MIN_BOX_PX and abs(b.y() - a.y()) >= MIN_BOX_PX:
                from utils.result_summary import CLASS_NAMES
                x0, y0 = self._to_raster(a.x(), a.y())
                x1, y1 = self._to_raster(b.x(), b.y())
                self._editor.add_box(x0, y0, x1, y1, {"class": class_id, "class_name": CLASS_NAMES.get(class_id, str(class_id))})
                self.editChanged.emit()
        self.update()

    @Slot()
    def cancelBox(self):
        self._box_drag = None
        self.update()

    @Property(int, notify=editChanged)
    def erasedCount(self):
        return len(self._editor.erased) + len(self._editor.erased_added) if self._editor else 0

    @Property(int, notify=editChanged)
    def addedCount(self):
        return self._editor.added_count if self._editor else 0

    @Property(bool, notify=editChanged)
    def hasEdits(self):
        return bool(self._editor and self._editor.has_edits)

    @Property(bool, notify=editChanged)
    def canUndo(self):
        return bool(self._editor and self._editor.can_undo)

    def _to_map(self, x, y):
        """Raster pixel (x, y) to map coordinates, for the added boxes."""
        gt = self._gt
        return gt[0] + gt[1] * x + gt[2] * y, gt[3] + gt[4] * x + gt[5] * y

    @Slot(result=str)
    def saveEdits(self):
        """Write the edits (erased and added boxes) into the shown shapefile itself and reload it.
        Returns its path, or "" when there is nothing to save. Undo works only until this point."""
        if self._editor is None or not self._editor.has_edits:
            return ""
        path = self._editor.save(self._shapefilePath, self._shapefilePath, to_map=self._to_map)
        self.load_shapefile()
        self.update()
        return path

    @Property(str, notify=sourcePathChanged)
    def sourcePath(self):
        return self._sourcePath

    @sourcePath.setter
    def sourcePath(self, val):
        if self._sourcePath != val:
            self._sourcePath = val
            self.load_raster()
            self.update()
            self.sourcePathChanged.emit()

    @Property(str, notify=shapefilePathChanged)
    def shapefilePath(self):
        return self._shapefilePath

    @staticmethod
    def _file_stamp(path):
        """Modification time of a shapefile, so a layer is read again when a task rewrites the file
        with the same name (otherwise the preview keeps the old features)."""
        try:
            return os.path.getmtime(path) if path else None
        except OSError:
            return None

    @shapefilePath.setter
    def shapefilePath(self, val):
        stamp = self._file_stamp(val)
        if self._shapefilePath != val or self._shapefileStamp != stamp:
            changed = self._shapefilePath != val
            self._shapefilePath, self._shapefileStamp = val, stamp
            self.load_shapefile()
            self.update()
            if changed:
                self.shapefilePathChanged.emit()

    @Property(str, notify=outlinePathChanged)
    def outlinePath(self):
        return self._outlinePath

    @outlinePath.setter
    def outlinePath(self, val):
        stamp = self._file_stamp(val)
        if self._outlinePath != val or self._outlineStamp != stamp:
            changed = self._outlinePath != val
            self._outlinePath, self._outlineStamp = val, stamp
            self._outlines = [poly for poly, _, _ in self._read_polygons(val)]
            self.update()
            if changed:
                self.outlinePathChanged.emit()

    @Property(bool)
    def showOutlines(self):
        return self._showOutlines

    @showOutlines.setter
    def showOutlines(self, val):
        if self._showOutlines != val:
            self._showOutlines = val
            self.update()

    def _read_polygons(self, path):
        """[(QPolygonF in raster pixels, feature, geometry)] of the outer rings of a shapefile."""
        if not path or not os.path.exists(path) or not self._inv_gt:
            return []
        ds = ogr.Open(path)
        if not ds:
            return []
        out = []
        for feature in ds.GetLayer():
            geom = feature.GetGeometryRef()
            if geom is None:
                continue
            ring = geom.GetGeometryRef(0) if geom.GetGeometryCount() > 0 else geom
            if ring is None or ring.GetPointCount() == 0:
                continue
            points = [QPointF(self._inv_gt[0] + self._inv_gt[1] * x + self._inv_gt[2] * y,
                              self._inv_gt[3] + self._inv_gt[4] * x + self._inv_gt[5] * y)
                      for x, y, *_ in ring.GetPoints()]
            out.append((QPolygonF(points), feature, geom.Clone()))
        return out

    @Property(bool)
    def showVectorOverlay(self):
        return self._showVectorOverlay

    @showVectorOverlay.setter
    def showVectorOverlay(self, val):
        if self._showVectorOverlay != val:
            self._showVectorOverlay = val
            self.update()

    @Property(bool)
    def colorByDetection(self):
        return self._colorByDetection

    @colorByDetection.setter
    def colorByDetection(self, val):
        if self._colorByDetection != val:
            self._colorByDetection = val
            self.update()

    @Property(float)
    def contentX(self):
        return self._contentX

    @contentX.setter
    def contentX(self, val):
        if self._contentX != val:
            self._contentX = val
            self.update()

    @Property(float)
    def contentY(self):
        return self._contentY

    @contentY.setter
    def contentY(self, val):
        if self._contentY != val:
            self._contentY = val
            self.update()

    @Property(float)
    def zoomScale(self):
        return self._zoomScale

    @zoomScale.setter
    def zoomScale(self, val):
        if self._zoomScale != val:
            self._zoomScale = val
            self.update()

    @Property(int, notify=rasterWidthChanged)
    def rasterWidth(self):
        return self._raster_w

    @Property(int, notify=rasterHeightChanged)
    def rasterHeight(self):
        return self._raster_h

    def load_raster(self):
        self._dataset = None
        self._raster_w = 0
        self._raster_h = 0
        self._inv_gt = None
        self._gt = None
        self._base = None
        self._base_scale = 0.0

        if not self._sourcePath or not os.path.exists(self._sourcePath):
            self.rasterWidthChanged.emit()
            self.rasterHeightChanged.emit()
            return
            
        self._dataset = gdal.Open(self._sourcePath)
        if self._dataset:
            self._raster_w = self._dataset.RasterXSize
            self._raster_h = self._dataset.RasterYSize
            gt = self._dataset.GetGeoTransform()
            if gt:
                self._gt = gt
                self._inv_gt = gdal.InvGeoTransform(gt)
            # Read the whole raster once at reduced size: while zoomed out, paint draws from this
            # copy instead of reading (and decoding) the whole file again on every repaint
            scale = min(1.0, BASE_SIZE / max(self._raster_w, self._raster_h))
            bw, bh = max(1, int(self._raster_w * scale)), max(1, int(self._raster_h * scale))
            data = self._dataset.ReadAsArray(buf_xsize=bw, buf_ysize=bh)
            if data is not None:
                data = to_rgb8(data)
                self._base = QImage(data.data, bw, bh, data.strides[0], QImage.Format_RGB888).copy()
                self._base_scale = bw / self._raster_w

        self.rasterWidthChanged.emit()
        self.rasterHeightChanged.emit()

    def load_shapefile(self):
        self._polygons = []
        self._empty = []              # True for the empty plots of the numbering (status "empty")
        self._labels = []
        fids = []                     # feature id of each polygon, for the erase tool
        self._editor = None
        self.editChanged.emit()
        if not self._shapefilePath or not os.path.exists(self._shapefilePath):
            return
            
        if not self._inv_gt:
            return
            
        shp_ds = ogr.Open(self._shapefilePath)
        if not shp_ds:
            return
            
        layer = shp_ds.GetLayer()
        if not layer:
            return

        # Numbered layers get a grid_id label per polygon
        has_grid = layer.GetLayerDefn().GetFieldIndex("grid_id") >= 0
        has_status = layer.GetLayerDefn().GetFieldIndex("status") >= 0
        widths = []

        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is not None:
                # Use only the outer ring
                ring = geom.GetGeometryRef(0) if geom.GetGeometryCount() > 0 else geom
                if ring:
                    points = []
                    for i in range(ring.GetPointCount()):
                        x, y, _ = ring.GetPoint(i)
                        # Transform geographic coordinates to raster pixel coordinates
                        px = self._inv_gt[0] + self._inv_gt[1]*x + self._inv_gt[2]*y
                        py = self._inv_gt[3] + self._inv_gt[4]*x + self._inv_gt[5]*y
                        points.append(QPointF(px, py))
                    if points:
                        poly = QPolygonF(points)
                        self._polygons.append(poly)
                        self._empty.append(has_status and feature.GetField("status") == "empty")
                        fids.append(feature.GetFID())
                        widths.append(poly.boundingRect().width())
                        grid_id = feature.GetField("grid_id") if has_grid else None
                        if grid_id:  # 0 or empty means not numbered
                            c = geom.Centroid()
                            cx = self._inv_gt[0] + self._inv_gt[1]*c.GetX() + self._inv_gt[2]*c.GetY()
                            cy = self._inv_gt[3] + self._inv_gt[4]*c.GetX() + self._inv_gt[5]*c.GetY()
                            self._labels.append((len(self._polygons) - 1, QPointF(cx, cy), str(grid_id)))

        # Show labels only when a typical polygon is at least about 30 screen pixels wide
        if widths:
            self._label_min_zoom = 30.0 / max(float(np.median(widths)), 1e-6)

        # Erase tool works on these polygons (raster pixels), see DetectionEditor
        from .detection_editor import DetectionEditor
        self._editor = DetectionEditor(fids, [[(p.x(), p.y()) for p in poly] for poly in self._polygons])
        self.editChanged.emit()

    def paint(self, painter: QPainter):
        if not self._dataset or self._raster_w == 0 or self._raster_h == 0:
            return

        w = int(self.width())
        h = int(self.height())
        if w <= 0 or h <= 0:
            return

        # Map the viewable area back to the original raster using the zoom scale
        # The viewable area is defined by contentX, contentY, width, height on the scaled image
        src_x = int(self._contentX / self._zoomScale)
        src_y = int(self._contentY / self._zoomScale)
        src_w = int(w / self._zoomScale)
        src_h = int(h / self._zoomScale)

        # Clip bounds to the raster size
        src_x = max(0, min(src_x, self._raster_w - 1))
        src_y = max(0, min(src_y, self._raster_h - 1))
        src_w = max(1, min(src_w, self._raster_w - src_x))
        src_h = max(1, min(src_h, self._raster_h - src_y))

        # Calculate actual draw sizes based on the clipped source rect to avoid stretching
        draw_w = int(src_w * self._zoomScale)
        draw_h = int(src_h * self._zoomScale)
        
        if self._base is not None and self._zoomScale <= self._base_scale:
            # Zoomed out: draw the visible part of the reduced copy (no file read)
            s = self._base_scale
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawImage(QRectF(0, 0, draw_w, draw_h), self._base, QRectF(src_x * s, src_y * s, src_w * s, src_h * s))
        else:
            # Zoomed in: read exactly the necessary pixels from GDAL, scaled to the destination size
            try:
                img_data = self._dataset.ReadAsArray(src_x, src_y, src_w, src_h, buf_xsize=draw_w, buf_ysize=draw_h)
            except Exception as e:
                print("GDAL Read Error:", e)
                return

            if img_data is None:
                return

            # rows x cols x 3 RGB, as QImage.Format_RGB888 expects
            img_data = to_rgb8(img_data)
            qimg = QImage(img_data.data, draw_w, draw_h, img_data.strides[0], QImage.Format_RGB888)

            # Draw the cropped raster image
            painter.drawImage(0, 0, qimg)

        # Plant outlines (plant_traits): green border with a light fill, under the boxes
        if self._showOutlines and self._outlines:
            painter.save()
            painter.scale(self._zoomScale, self._zoomScale)
            painter.translate(-src_x, -src_y)
            painter.setRenderHint(QPainter.Antialiasing)
            outline_pen = QPen(QColor(60, 220, 90))
            outline_pen.setWidthF(1.8 / self._zoomScale)
            painter.setPen(outline_pen)
            painter.setBrush(QColor(60, 220, 90, 50))
            for poly in self._outlines:
                painter.drawPolygon(poly)
            painter.restore()

        # Draw vector overlay if requested
        if self._showVectorOverlay and self._polygons:
            painter.save()
            # Set up the painter to match the raster coordinate system
            # The top-left of the QImage corresponds to (src_x, src_y) in raster pixel coords
            # First scale by zoom
            painter.scale(self._zoomScale, self._zoomScale)
            # Then translate back by the source offset
            painter.translate(-src_x, -src_y)
            
            # Light blue borders only (no fill), same style as QGIS_STYLE in processor.py
            pen = QPen(QColor(90, 190, 255))
            pen.setWidthF(2.5 / self._zoomScale) # Keep stroke width visually constant regardless of zoom
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            
            # Use anti-aliasing for vectors
            painter.setRenderHint(QPainter.Antialiasing)
            
            erased = self._editor.erased if self._editor else set()
            for i, poly in enumerate(self._polygons):
                if i in erased:
                    continue  # erased with the erase tool (until saved or undone)
                if self._empty[i]:
                    # Empty plots of the numbering (no plant detected): grey, dashed
                    pen.setColor(QColor(200, 200, 200))
                    pen.setStyle(Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawPolygon(poly)
                    pen.setStyle(Qt.SolidLine)
                    pen.setColor(QColor(90, 190, 255))
                    painter.setPen(pen)
                    continue
                if self._colorByDetection:
                    # Golden ratio steps of hue give neighbouring polygons clearly different colours
                    pen.setColor(QColor.fromHsvF((i * 0.618034) % 1.0, 0.9, 1.0))
                    painter.setPen(pen)
                # Optionally, we could clip to the visible rectangle to avoid drawing offscreen polygons
                painter.drawPolygon(poly)

            # Boxes added with the add tool (until saved or undone): orange, so they stand out
            if self._editor is not None:
                pen.setColor(QColor(255, 170, 0))
                painter.setPen(pen)
                for _, points in self._editor.added_boxes():
                    painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))

            painter.restore()

        # Draw numbering labels in screen coordinates so the text keeps a constant size
        if self._showVectorOverlay and self._labels and self._zoomScale >= self._label_min_zoom:
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)
            font = QFont()
            font.setPixelSize(14)
            font.setBold(True)
            metrics = QFontMetricsF(font)
            halo = QPen(QColor(40, 40, 40, 220), 3.0)  # dark halo keeps white text readable
            halo.setJoinStyle(Qt.RoundJoin)
            erased = self._editor.erased if self._editor else set()
            for index, pt, text in self._labels:
                if index in erased:
                    continue
                sx = (pt.x() - src_x) * self._zoomScale
                sy = (pt.y() - src_y) * self._zoomScale
                if 0 <= sx <= w and 0 <= sy <= h:
                    # White text centred on the polygon, drawn as a path to get the halo
                    path = QPainterPath()
                    path.addText(sx - metrics.horizontalAdvance(text) / 2,
                                 sy + (metrics.ascent() - metrics.descent()) / 2, font, text)
                    painter.strokePath(path, halo)
                    painter.fillPath(path, QColor(255, 255, 255))
            painter.restore()

        # Stroke being traced with the erase tool (screen coordinates)
        if len(self._trace) > 1:
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)
            trace_pen = QPen(QColor(255, 60, 60, 220), 3.0)
            trace_pen.setCapStyle(Qt.RoundCap)
            trace_pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(trace_pen)
            painter.drawPolyline(QPolygonF(self._trace))
            painter.restore()

        # Box being dragged with the add tool (screen coordinates)
        if self._box_drag:
            painter.save()
            drag_pen = QPen(QColor(255, 170, 0), 2.0, Qt.DashLine)
            painter.setPen(drag_pen)
            painter.setBrush(QColor(255, 170, 0, 40))
            painter.drawRect(QRectF(self._box_drag[0], self._box_drag[1]).normalized())
            painter.restore()
