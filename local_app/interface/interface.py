import os
import json
import webbrowser

from PySide6.QtCore import QObject, Slot, Signal
from PySide6.QtCore import QThread

from .processor import Processor

class Worker(QThread):
    finished = Signal(dict)  # Signal emitted when the thread finishes processing
    progressUpdated = Signal(dict) # Add signal for progress (current, total)

    def __init__(self, params):
        super().__init__()
        self.params = params
        # Add a flag for interruption
        self._is_interruption_requested = False

    def run(self):
        """Long-running task."""
        import time
        print("Processing started...")

        processor = Processor(self.params
                              , progress_callback = self.progressUpdated.emit
                              , interruption_check = self.isInterruptionRequested)
        results = processor.run()

        if not self.isInterruptionRequested():
            print("Processing finished!")
            #self.finished.emit({})
            print(results)
            self.finished.emit(results)
        else:
            print("Processing cancelled!")
            # Optionally emit a different signal or specific info for cancellation
            results.update({"status": "cancelled"})
            self.progressUpdated.emit({"status": "Cancelled..."})
            self.finished.emit(results)

    # Override isInterruptionRequested to use our flag
    def isInterruptionRequested(self):
        return self._is_interruption_requested
    
    # Add a method to request interruption
    def requestInterruption(self):
        self._is_interruption_requested = True

class ProcessorInterface(QObject):
    """Interface for processing tasks."""

    msg = Signal(str)
    finished = Signal(dict)
    progressUpdated = Signal(dict) # Relay progress signal
    visualizationReady = Signal(str) # New signal for QML visualization
    rasterPreviewReady = Signal(str) # New signal for raw raster preview

    def initialize(self):
        pass

    @Slot()
    def execute(self):
        print('Execute')

    @Slot()
    def click(self):
        print('click')

    @Slot()
    def download(self):
        print('download')

    @Slot()
    def cancelProcessing(self):
        """Request cancellation of the running worker."""
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            print("Requesting processing cancellation...")
            self.worker.requestInterruption() # Use the new method

    @Slot(dict)
    def onProcessFinished(self, info):
        """Handle the process completion."""
        self.finished.emit(info)
        print("Process finished signal emitted.")
        
        # Try to generate visualization
        self.generateVisualization()

        # Clean up worker reference
        self.worker = None

    @Slot(str)
    def previewRaster(self, input_raster):
        """Generates a quick thumbnail preview of the input raster without any shapefiles."""
        import traceback
        try:
            if not input_raster or not os.path.exists(input_raster):
                return
            
            import cv2
            import numpy as np
            from osgeo import gdal
            
            ds = gdal.Open(input_raster)
            if not ds:
                return
                
            thumbnail_size = (800, 800)
            w = ds.RasterXSize
            h = ds.RasterYSize
            
            scale_x = thumbnail_size[0] / w
            scale_y = thumbnail_size[1] / h
            scale = min(scale_x, scale_y)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            img = ds.ReadAsArray(0, 0, w, h, buf_xsize=new_w, buf_ysize=new_h)
            if len(img.shape) == 3:
                img = img.transpose(1, 2, 0)
                if img.shape[2] >= 3:
                    img = img[:, :, :3]
            elif len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
                
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
            vis_path = os.path.join(os.path.expanduser("~"), ".cache_forages_rois", "preview_raster.png")
            os.makedirs(os.path.dirname(vis_path), exist_ok=True)
            
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_path, img_bgr)
            
            self.rasterPreviewReady.emit(vis_path)
            print(f"Raster preview saved to {vis_path}")
        except Exception as e:
            print(f"Raster preview error: {e}")
            traceback.print_exc()

    def generateVisualization(self):
        import traceback
        try:
            if not hasattr(self, 'last_params') or not self.last_params:
                return

            input_raster = self.last_params.get("input_file")
            output_folder = self.last_params.get("output_folder")
            
            if not input_raster or not output_folder:
                return
                
            basename = os.path.splitext(os.path.basename(input_raster))[0]
            
            if output_folder.lower().endswith(".shp"):
                output_shp = output_folder
            else:
                # We expect custom_processor.tile_inference to write to output_folder/basename.shp or similar
                output_shp = os.path.join(output_folder, f"{basename}.shp")
                if not os.path.exists(output_shp):
                    output_shp = os.path.join(output_folder, "merged.shp")
            
            if not os.path.exists(output_shp):
                print(f"Cannot visualize: Output shapefile not found at {output_shp}")
                return

            import cv2
            import numpy as np
            from osgeo import gdal, ogr
            
            ds = gdal.Open(input_raster)
            if not ds:
                return
                
            thumbnail_size = (800, 800)
            
            w = ds.RasterXSize
            h = ds.RasterYSize
            
            scale_x = thumbnail_size[0] / w
            scale_y = thumbnail_size[1] / h
            scale = min(scale_x, scale_y)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            img = ds.ReadAsArray(0, 0, w, h, buf_xsize=new_w, buf_ysize=new_h)
            if len(img.shape) == 3:
                img = img.transpose(1, 2, 0)
                if img.shape[2] >= 3:
                    img = img[:, :, :3]
            elif len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
                
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
            shp_ds = ogr.Open(output_shp)
            if shp_ds is not None:
                layer = shp_ds.GetLayer()
                gt = ds.GetGeoTransform()
                inv_gt = gdal.InvGeoTransform(gt)
                
                for feature in layer:
                    geom = feature.GetGeometryRef()
                    if geom is not None:
                        # Only processing the outer ring for simplicity in preview
                        ring = geom.GetGeometryRef(0) if geom.GetGeometryCount() > 0 else geom
                        if ring:
                            pts = []
                            for i in range(ring.GetPointCount()):
                                x, y, _ = ring.GetPoint(i)
                                px = int(inv_gt[0] + inv_gt[1]*x + inv_gt[2]*y)
                                py = int(inv_gt[3] + inv_gt[4]*x + inv_gt[5]*y)
                                pts.append([int(px * scale), int(py * scale)])
                            
                            if pts:
                                pts = np.array(pts, np.int32)
                                pts = pts.reshape((-1, 1, 2))
                                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                
            vis_path = os.path.join(os.path.expanduser("~"), ".cache_forages_rois", "vis.png")
            os.makedirs(os.path.dirname(vis_path), exist_ok=True)
            
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_path, img_bgr)
            
            self.visualizationReady.emit(vis_path)
            print(f"Visualization saved to {vis_path}")
            
        except Exception as e:
            print(f"Visualization error: {e}")
            traceback.print_exc()

    @Slot(str)
    def openOutputFile(self, output_file):
        """Open the output file in Excel."""
        output_file = output_file.replace("file:///","")
        if os.path.exists(output_file):
            try:
                import subprocess
                if os.name == 'nt':  # Windows
                    os.startfile(output_file)
                elif os.name == 'posix':  # macOS/Linux
                    subprocess.run(["open", output_file])  # macOS
            except Exception as e:
                print(f"Failed to open file: {e}")
        else:
            print(f"Output file not found: {output_file}")

    @Slot(str)
    def openOutputFolder(self, output_folder):
        """Open the output folder in the default file explorer."""
        if os.path.isdir(output_folder): # Check if it's a valid directory
            try:
                import subprocess
                if os.name == 'nt':  # Windows
                    os.startfile(output_folder)
                elif os.name == 'posix':  # macOS/Linux
                    if sys.platform == "darwin": # macOS
                        subprocess.run(["open", output_folder])
                    else: # Linux
                        subprocess.run(["xdg-open", output_folder])
                else:
                    print(f"Unsupported OS: {os.name}")
            except Exception as e:
                print(f"Failed to open folder: {e}")
        else:
            print(f"Output folder not found or is not a directory: {output_folder}")

    @Slot(str)
    def open_url(self, url):
        """Open website in default web browser"""
        webbrowser.open(url)

    @Slot(str, str)
    def saveParametersJson(self, file_path, json_data_string):
        """Saves the provided JSON string to the specified file path."""
        try:
            params = json.loads(json_data_string)
            with open(file_path, 'w') as f:
                json.dump(params, f, indent=4) # Write with indentation
            print(f"Parameters saved to: {file_path}")
        except Exception as e:
            print(f"Error saving parameters to {file_path}: {e}")
            self.saveLoadError.emit(f"Error saving parameters: {e}")

    @Slot(str)
    def loadParametersJson(self, file_path):
        """Loads parameters from the specified JSON file path."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Parameter file not found: {file_path}")
            with open(file_path, 'r') as f:
                params = json.load(f)
            if "inputFolder" not in params or "outputFolder" not in params:
                 raise ValueError("Invalid parameter file format.")
            self.parametersLoaded.emit(params) 
            print(f"Parameters loaded from: {file_path}")
        except Exception as e:
            print(f"Error loading parameters from {file_path}: {e}")
            self.saveLoadError.emit(f"Error loading parameters: {e}")

    @Slot(dict)
    def process(self, params):
        self.last_params = params
        self.worker = Worker(params)
        self.worker.finished.connect(self.onProcessFinished)
        self.worker.progressUpdated.connect(self.progressUpdated)
        self.worker.start()

        