import os
import numpy as np
import cv2
from osgeo import gdal, ogr

from PySide6.QtQuick import QQuickPaintedItem
from PySide6.QtGui import QImage, QColor, QPainter, QPolygonF, QPen
from PySide6.QtCore import Property, QRectF, QPointF, Signal, Slot, Qt

class RasterPreviewItem(QQuickPaintedItem):
    
    rasterWidthChanged = Signal()
    rasterHeightChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._sourcePath = ""
        self._shapefilePath = ""
        self._showVectorOverlay = False
        
        self._contentX = 0.0
        self._contentY = 0.0
        self._zoomScale = 1.0
        
        self._dataset = None
        self._raster_w = 0
        self._raster_h = 0
        self._inv_gt = None
        
        self._polygons = []
        
        self.setRenderTarget(QQuickPaintedItem.FramebufferObject)

    @Property(str)
    def sourcePath(self):
        return self._sourcePath

    @sourcePath.setter
    def sourcePath(self, val):
        if self._sourcePath != val:
            self._sourcePath = val
            self.load_raster()
            self.update()

    @Property(str)
    def shapefilePath(self):
        return self._shapefilePath

    @shapefilePath.setter
    def shapefilePath(self, val):
        if self._shapefilePath != val:
            self._shapefilePath = val
            self.load_shapefile()
            self.update()

    @Property(bool)
    def showVectorOverlay(self):
        return self._showVectorOverlay

    @showVectorOverlay.setter
    def showVectorOverlay(self, val):
        if self._showVectorOverlay != val:
            self._showVectorOverlay = val
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
                self._inv_gt = gdal.InvGeoTransform(gt)
            
        self.rasterWidthChanged.emit()
        self.rasterHeightChanged.emit()

    def load_shapefile(self):
        self._polygons = []
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
                        self._polygons.append(QPolygonF(points))
                        
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
        
        # Read exactly the necessary pixels from GDAL, automatically scaled to the destination size
        try:
            img_data = self._dataset.ReadAsArray(src_x, src_y, src_w, src_h, buf_xsize=draw_w, buf_ysize=draw_h)
        except Exception as e:
            print("GDAL Read Error:", e)
            return
            
        if img_data is None:
            return

        # Convert to QImage
        if len(img_data.shape) == 3:
            img_data = img_data.transpose(1, 2, 0)
            if img_data.shape[2] >= 3:
                img_data = img_data[:, :, :3]
        elif len(img_data.shape) == 2:
            img_data = np.stack([img_data, img_data, img_data], axis=-1)
            
        if img_data.dtype != np.uint8:
            img_data = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            img_data = np.ascontiguousarray(img_data)
            
        # img_data is now H x W x 3, RGB/BGR? GDAL usually reads in RGB if it's a 3-band color image
        # PySide6 QImage.Format_RGB888 expects RGB
        qimg = QImage(img_data.data, draw_w, draw_h, img_data.strides[0], QImage.Format_RGB888)

        # Draw the cropped raster image
        painter.drawImage(0, 0, qimg)

        # Draw vector overlay if requested
        if self._showVectorOverlay and self._polygons:
            painter.save()
            # Set up the painter to match the raster coordinate system
            # The top-left of the QImage corresponds to (src_x, src_y) in raster pixel coords
            # First scale by zoom
            painter.scale(self._zoomScale, self._zoomScale)
            # Then translate back by the source offset
            painter.translate(-src_x, -src_y)
            
            pen = QPen(QColor(0, 255, 0))
            pen.setWidthF(2.0 / self._zoomScale) # Keep stroke width visually constant regardless of zoom
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            
            # Use anti-aliasing for vectors
            painter.setRenderHint(QPainter.Antialiasing)
            
            for poly in self._polygons:
                # Optionally, we could clip to the visible rectangle to avoid drawing offscreen polygons
                painter.drawPolygon(poly)
                
            painter.restore()
