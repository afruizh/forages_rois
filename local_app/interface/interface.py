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
        import traceback
        print("Processing started...")

        try:
            processor = Processor(self.params
                                  , progress_callback = self.progressUpdated.emit
                                  , interruption_check = self.isInterruptionRequested)
            results = processor.run()
        except Exception as e:
            traceback.print_exc()
            results = {"status": "error", "message": str(e)}
            self.progressUpdated.emit({"status": "Error occurred.", "progress": 0})

        self.results = results  # read by the CLI after wait() (no event loop there)
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

class ModelsDownloadWorker(QThread):
    """Downloads the missing models in the background (see custom_processor.download_models)."""
    finished = Signal(dict)
    progressUpdated = Signal(dict)

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        import custom_processor
        try:
            names = custom_processor.download_models(self.folder, progress_callback=self.progressUpdated.emit,
                                                     interruption_check=self.isInterruptionRequested)
            results = {"status": "completed", "downloaded": names}
        except Exception as e:
            results = {"status": "error", "message": str(e)}
        self.finished.emit(results)

class ProcessorInterface(QObject):
    """Interface for processing tasks."""

    msg = Signal(str)
    finished = Signal(dict)
    progressUpdated = Signal(dict) # Relay progress signal
    visualizationReady = Signal(str) # New signal for QML visualization
    rasterPreviewReady = Signal(str, str) # (preview image, raster shown: the RGB view of the chosen bands)
    rasterBandsInfo = Signal("QVariantList", "QVariantList", bool) # (band labels, [r, g, b] used, user should choose)
    visualizationPathsReady = Signal(str, str) # Emits (raster_path, shp_path)
    outlinesReady = Signal(str)  # plant outlines layer (plant_traits output) for the preview
    modelsStatus = Signal(str, str, str) # Emits (level "ok"/"warning"/"error", models folder, message)

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
        self.last_results = info  # holds "output_file" for the visualization
        self.finished.emit(info)
        print("Process finished signal emitted.")
        
        # Try to generate visualization only if successful
        if info.get("status") != "error":
            self.generateVisualization()

        # Clean up worker reference
        self.worker = None

    def previewImagePath(self, name, folder):
        """Path for a preview image in the cache of folder (the output folder), or in the default
        cache before one is chosen or if folder cannot be written (see cachemanager)."""
        from utils.cachemanager import cache_folder
        for base in (cache_folder(folder), cache_folder(None)):
            path = os.path.join(base, name)
            try:
                os.makedirs(base, exist_ok=True)
                return path
            except OSError:
                continue
        return path

    def rgbView(self, input_raster, folder, bands=""):
        """RGB view of the raster with the chosen bands (see rgb_bands); the raster itself when it
        is already 8 bit RGB. The view goes to the same cache as the previews."""
        from processing.rgb_bands import rgb_raster, parse_bands
        cache = os.path.dirname(self.previewImagePath("x", folder))
        return rgb_raster(input_raster, parse_bands(bands), cache)

    @Slot(str, str)
    @Slot(str, str, str)
    def previewRaster(self, input_raster, output_folder="", bands=""):
        """Generates a quick thumbnail preview of the input raster without any shapefiles.
        bands: "r,g,b" chosen in the GUI, or empty to guess them."""
        import traceback
        try:
            if not input_raster or not os.path.exists(input_raster):
                return

            import cv2
            import numpy as np
            from osgeo import gdal
            from processing.rgb_bands import band_labels, guess_rgb_bands, parse_bands, choose_bands

            used = parse_bands(bands) or guess_rgb_bands(input_raster)
            self.rasterBandsInfo.emit(band_labels(input_raster), used, choose_bands(input_raster))
            shown_raster = self.rgbView(input_raster, output_folder, ",".join(map(str, used)))
            ds = gdal.Open(shown_raster)
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
            else:
                img = np.ascontiguousarray(img)
                
            vis_path = self.previewImagePath("preview_raster.png", output_folder)
            
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_path, img_bgr)
            
            self.rasterPreviewReady.emit(vis_path, shown_raster)
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
            
            task = self.last_params.get("task", "")
            if task == "export_coco":
                return  # writes a dataset folder, not a layer: the preview keeps what it shows

            # Prefer the output path reported by the processor; fall back to the naming convention
            output_shp = (getattr(self, "last_results", None) or {}).get("output_file")
            if output_shp and os.path.exists(output_shp):
                pass
            elif output_folder.lower().endswith(".shp"):
                output_shp = output_folder
            else:
                if task == "crop_detection":
                    output_shp = os.path.join(output_folder, f"{basename}_crops.shp")
                elif task == "detection":
                    output_shp = os.path.join(output_folder, f"{basename}_boxes.shp")
                elif task == "tiling_detection_on_crop":
                    output_shp = os.path.join(output_folder, f"{basename}_on_crop.shp")
                elif task == "plot_numbering":
                    output_shp = self.last_params.get("input_shapefile") or input_raster  # numbered in place
                elif task == "postprocessing":
                    output_shp = os.path.join(output_folder, f"{basename}_postprocess.shp")
                else:
                    output_shp = os.path.join(output_folder, f"{basename}.shp")
                    
                if not os.path.exists(output_shp):
                    output_shp = os.path.join(output_folder, "merged.shp")
            
            if not os.path.exists(output_shp):
                print(f"Cannot visualize: Output shapefile not found at {output_shp}")
                return

            import cv2
            import numpy as np
            from osgeo import gdal, ogr

            # Same RGB view (chosen bands) as the processing and the raster preview
            input_raster = self.rgbView(input_raster, output_folder, self.last_params.get("bands", ""))
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
            else:
                img = np.ascontiguousarray(img)
                
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
                
            vis_path = self.previewImagePath("vis.png", output_folder)
            
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_path, img_bgr)
            
            self.visualizationReady.emit(vis_path)
            # plant_traits: keep the input boxes as the detections layer and add the outlines as a
            # second layer, so both can be shown; other tasks replace the detections layer
            boxes = self.last_params.get("input_shapefile") if task == "plant_traits" else ""
            if boxes and os.path.exists(boxes):
                self.visualizationPathsReady.emit(input_raster, boxes)
                self.outlinesReady.emit(output_shp)
            else:
                self.visualizationPathsReady.emit(input_raster, output_shp)
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
        """Open the output folder in the default file explorer (for a file path, its parent folder)."""
        output_folder = output_folder.replace("file:///", "")
        if not os.path.isdir(output_folder):
            output_folder = os.path.dirname(output_folder)
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

    def check_models_folder(self, folder):
        """Return (level, message) for a models folder: "ok", "warning" (some models missing)
        or "error" (folder missing or no models at all)."""
        import custom_processor
        if not folder or not os.path.isdir(folder):
            return "error", f"The models folder does not exist:\n{folder}"
        missing = custom_processor.missing_models(folder)
        if len(missing) == len(custom_processor.REQUIRED_MODELS):
            return "error", f"No models were found in:\n{folder}\n\nExpected files:\n" + "\n".join(
                f"- {name}: {use}" for name, use in custom_processor.REQUIRED_MODELS.items())
        if missing:
            return "warning", f"Some models are missing in:\n{folder}\n\nTasks that need them will fail:\n" + "\n".join(
                f"- {name}: {custom_processor.REQUIRED_MODELS[name]}" for name in missing)
        return "ok", f"Models folder:\n{folder}\n\nAll models found."

    @Slot(str)
    def setModelsFolder(self, folder):
        """Use a new models folder (File > Set Models Folder). It is saved for next sessions
        unless no models are found in it."""
        import custom_processor
        from utils.app_settings import settings
        level, message = self.check_models_folder(folder)
        if level != "error":
            custom_processor.MODEL_PATH = folder
            settings().setValue("models_folder", folder)
        else:
            message += f"\n\nThe current models folder is kept:\n{custom_processor.MODEL_PATH}"
        self.modelsStatus.emit(level, custom_processor.MODEL_PATH, message)

    @Slot()
    def loadModelsFolder(self):
        """At startup: use the saved models folder if it still exists, otherwise the default one."""
        import custom_processor
        from utils.app_settings import settings
        saved = settings().value("models_folder", "")
        if saved and os.path.isdir(saved):
            custom_processor.MODEL_PATH = saved
        level, message = self.check_models_folder(custom_processor.MODEL_PATH)
        if level != "ok":
            message += ("\n\nUse File > Download Models to download them from Hugging Face (about 100 MB), "
                        "or File > Set Models Folder to select the folder that contains them.")
        print(message)
        self.modelsStatus.emit(level, custom_processor.MODEL_PATH, message)

    def writable_models_folder(self, folder):
        """folder if models can be written there, otherwise a models folder in the user's
        application data (for an executable unzipped where the user cannot write)."""
        try:
            os.makedirs(folder, exist_ok=True)
            test = os.path.join(folder, ".write_test")
            open(test, "w").close()
            os.remove(test)
            return folder
        except OSError:
            from PySide6.QtCore import QStandardPaths
            return os.path.join(QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation), "models")

    @Slot()
    def downloadModels(self):
        """Download the missing models into the models folder (File > Download Models). When it ends,
        the folder is checked and saved like with Set Models Folder; errors go to the error dialog."""
        import custom_processor
        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            self.finished.emit({"status": "error", "message": "Wait until the current task ends before downloading the models."})
            return
        self.worker = ModelsDownloadWorker(self.writable_models_folder(custom_processor.MODEL_PATH))
        self.worker.progressUpdated.connect(self.progressUpdated)
        self.worker.finished.connect(self.onModelsDownloaded)
        self.worker.start()

    @Slot(dict)
    def onModelsDownloaded(self, info):
        folder = self.worker.folder
        self.worker = None
        if info.get("status") == "error":
            self.finished.emit(info)
            return
        self.progressUpdated.emit({"progress": 100, "status": "Models downloaded." if info.get("downloaded") else "All models found."})
        self.setModelsFolder(folder)

    # ---- Cache (File > Cache): default cache folder and the cache of the output folder ----

    @Slot(str, result=str)
    def cacheFolder(self, output_folder):
        """The cache of output_folder, or the default cache for an empty string."""
        from utils.cachemanager import cache_folder
        return cache_folder(output_folder)

    @Slot(str, result=str)
    def cacheSizeText(self, folder):
        from utils.cachemanager import folder_size
        size = folder_size(folder)
        return f"{size / 2**30:.1f} GB" if size >= 2**30 else f"{size / 2**20:.0f} MB"

    @Slot(str)
    def openCacheFolder(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.openOutputFolder(folder)

    # ---- Last folder used in the file dialogs (see app_settings) ----

    @Slot(str, str, str, result=str)
    def startFolderUrl(self, kind, current_path, preferred_folder):
        """URL of the folder a dialog of this kind should open in (works for network paths too)."""
        from PySide6.QtCore import QUrl
        from utils.app_settings import start_folder
        return QUrl.fromLocalFile(start_folder(kind, current_path, preferred_folder)).toString()

    @Slot(str, str)
    def rememberFolder(self, kind, path):
        from utils.app_settings import remember_folder
        remember_folder(kind, path)

    @Slot(str)
    def setDefaultCacheFolder(self, folder):
        """Default cache in folder/.cache_forages_rois from now on (saved for next sessions)."""
        from utils.app_settings import settings
        from utils.cachemanager import default_cache_folder
        settings().setValue("cache_folder", folder)
        self.progressUpdated.emit({"status": f" Default cache: {default_cache_folder()}"})

    @Slot(str)
    def clearCache(self, folder):
        """Delete a cache folder (only ForagesROIs cache folders, see cachemanager.clear_cache)."""
        from utils.cachemanager import clear_cache
        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            self.finished.emit({"status": "error", "message": "Wait until the current task ends before clearing the cache."})
            return
        try:
            clear_cache(folder)
            self.progressUpdated.emit({"progress": 0, "status": f" Cache cleared: {folder}"})
        except Exception as e:
            self.finished.emit({"status": "error", "message": f"Could not clear the cache:\n{folder}\n\n{e}"})

    @Slot(result=str)
    def getModelsFolder(self):
        import custom_processor
        return custom_processor.MODEL_PATH

    @Slot(dict)
    def process(self, params):
        self.last_params = params
        self.worker = Worker(params)
        self.worker.finished.connect(self.onProcessFinished)
        self.worker.progressUpdated.connect(self.progressUpdated)
        self.worker.start()

        