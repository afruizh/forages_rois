"""Edit detections with the mouse: erase them by tracing over them, add boxes by dragging, with
undo, and save the result into the shapefile.

Independent of Qt and of the preview: the polygons, the traced segments and the added boxes are
given in the same coordinates (the preview uses raster pixels). Each feature is kept by its FID,
so saving writes the input shapefile without the erased features, with all their fields, plus the
added boxes (converted to map coordinates by the to_map function given to save).
"""
import os
from shapely.geometry import LineString, Point, Polygon, box
from shapely.strtree import STRtree


class DetectionEditor:

    def __init__(self, fids, polygons):
        """fids: feature id of each polygon in the shapefile; polygons: list of [(x, y), ...]."""
        self.fids = list(fids)
        geoms = [Polygon(p) if len(p) >= 3 else Point(p[0]) for p in polygons]
        self._tree = STRtree(geoms) if geoms else None
        self.erased = set()        # indexes (into fids) of the erased features
        self.added = []            # boxes added with the mouse: {"polygon": [(x, y), ...], "fields": {...}}
        self.erased_added = set()  # indexes (into added) of added boxes erased again
        self._history = []         # one entry per stroke or added box, for undo
        self._stroke = None

    # ---- Erase ----

    def begin_stroke(self):
        self._stroke = (set(), set())

    def erase_segment(self, x0, y0, x1, y1, tolerance=0.0):
        """Erase the features and added boxes that the segment touches (within tolerance); returns
        how many were erased."""
        line = Point(x0, y0) if (x0, y0) == (x1, y1) else LineString([(x0, y0), (x1, y1)])
        area = line.buffer(tolerance) if tolerance > 0 else line
        hits = []
        if self._tree is not None:
            hits = [int(i) for i in self._tree.query(area, predicate="intersects") if int(i) not in self.erased]
        hits_added = [k for k, b in enumerate(self.added)
                      if k not in self.erased_added and Polygon(b["polygon"]).intersects(area)]
        self.erased.update(hits)
        self.erased_added.update(hits_added)
        if self._stroke is not None:
            self._stroke[0].update(hits)
            self._stroke[1].update(hits_added)
        return len(hits) + len(hits_added)

    def end_stroke(self):
        """Close the stroke: everything it erased is undone together."""
        if self._stroke and (self._stroke[0] or self._stroke[1]):
            self._history.append(("erase", self._stroke))
        self._stroke = None

    # ---- Add ----

    def add_box(self, x0, y0, x1, y1, fields=None):
        """Add a box with corners (x0, y0) and (x1, y1) and the given field values; returns its index."""
        rect = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self.added.append({"polygon": list(rect.exterior.coords)[:-1], "fields": dict(fields or {})})
        self._history.append(("add", len(self.added) - 1))
        return len(self.added) - 1

    def added_boxes(self):
        """[(index, polygon)] of the added boxes that are not erased."""
        return [(k, b["polygon"]) for k, b in enumerate(self.added) if k not in self.erased_added]

    # ---- Undo and state ----

    def undo(self):
        """Undo the last stroke or added box; returns False when there is nothing to undo."""
        if not self._history:
            return False
        kind, entry = self._history.pop()
        if kind == "add":
            self.added.pop()           # always the last one, since undo goes in reverse order
            self.erased_added.discard(entry)
        else:
            self.erased -= entry[0]
            self.erased_added -= entry[1]
        return True

    @property
    def can_undo(self):
        return bool(self._history)

    @property
    def added_count(self):
        return len(self.added) - len(self.erased_added)

    @property
    def has_edits(self):
        return bool(self.erased) or self.added_count > 0

    # ---- Save ----

    def save(self, input_shp, output_shp, to_map=None):
        """Write input_shp without the erased features and with the added boxes to output_shp (which
        may be input_shp), with the QGIS style (.qml) of the input if there is one. to_map(x, y)
        converts the editor coordinates of the added boxes to map coordinates. An added box gets
        its own field values, grid_id 0 (not numbered, see plot_numbering), status "plant", and the
        crop_id of the nearest kept feature. Returns output_shp."""
        from osgeo import ogr
        added = [self.added[k] for k, _ in self.added_boxes()]
        if added and to_map is None:
            raise ValueError("Saving added boxes needs to_map (editor to map coordinates).")
        erased_fids = {self.fids[i] for i in self.erased}
        src = ogr.Open(input_shp)
        layer = src.GetLayer()
        defn = layer.GetLayerDefn()
        fields = [defn.GetFieldDefn(i) for i in range(defn.GetFieldCount())]
        fields = [(f.GetName(), f.GetType(), f.GetWidth(), f.GetPrecision()) for f in fields]
        srs, geom_type = layer.GetSpatialRef(), layer.GetGeomType()
        srs = srs.Clone() if srs else None
        kept = [f.Clone() for f in layer if f.GetFID() not in erased_fids]
        src = None
        style = os.path.splitext(input_shp)[0] + ".qml"
        style_text = open(style, encoding="utf-8").read() if os.path.exists(style) else None

        driver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(output_shp):
            driver.DeleteDataSource(output_shp)
        out = driver.CreateDataSource(output_shp)
        out_layer = out.CreateLayer(os.path.splitext(os.path.basename(output_shp))[0], srs, geom_type)
        for name, ftype, width, precision in fields:
            fd = ogr.FieldDefn(name, ftype)
            fd.SetWidth(width)
            fd.SetPrecision(precision)
            out_layer.CreateField(fd)
        for f in kept:
            new = ogr.Feature(out_layer.GetLayerDefn())
            new.SetFrom(f)
            out_layer.CreateFeature(new)
        names = {name for name, *_ in fields}
        for b in added:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for x, y in b["polygon"] + b["polygon"][:1]:
                ring.AddPoint_2D(*to_map(x, y))
            geom = ogr.Geometry(ogr.wkbPolygon)
            geom.AddGeometry(ring)
            values = {"grid_id": 0, "status": "plant", "Type": "forage_plant"}
            if "crop_id" in names and kept:
                centre = geom.Centroid()
                nearest = min(kept, key=lambda f: f.GetGeometryRef().Centroid().Distance(centre))
                values["crop_id"] = nearest.GetField("crop_id")
            values.update(b["fields"])
            new = ogr.Feature(out_layer.GetLayerDefn())
            for name, value in values.items():
                if name in names and value is not None:
                    new.SetField(name, value)
            new.SetGeometry(geom)
            out_layer.CreateFeature(new)
        out = None
        if style_text is not None:
            with open(os.path.splitext(output_shp)[0] + ".qml", "w", encoding="utf-8") as fh:
                fh.write(style_text)
        return output_shp
