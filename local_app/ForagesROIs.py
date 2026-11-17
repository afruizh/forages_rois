import os
import sys
# Add the directory of the executable to PATH at runtime if frozen
# (sys.frozen is set by PyInstaller, __compiled__ by Nuitka)
if getattr(sys, 'frozen', False) or "__compiled__" in globals():
    exe_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    # Opened without a console (double click), there is no stdout or stderr: print() must not fail
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")

    # Force correct environment
    os.environ["PROJ_LIB"] = os.path.join(exe_dir, "proj_dir", "share", "proj")
    os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"]  # name used by PROJ 9.1+
    os.environ["GDAL_DATA"] = os.path.join(exe_dir, "gdal_data")
    os.environ["PATH"] = exe_dir + os.pathsep + os.environ.get("PATH", "")

    print("PROJ_LIB", os.environ["PROJ_LIB"])
    print("GDAL_DATA", os.environ["GDAL_DATA"])
    print("PATH", os.environ["PATH"])

    from ctypes import windll
    try:
        windll.kernel32.SetDllDirectoryW(exe_dir)
    except Exception as e:
        print("Failed to set DLL directory:", e)

import sys
# import time
import argparse

# from PySide6.QtWidgets import QApplication
# from PySide6.QtGui import QIcon
# from PySide6.QtGui import QPixmap
# from PySide6.QtWidgets import QSplashScreen
# from PySide6.QtQml import QQmlApplicationEngine

# interface.interface (onnxruntime, GDAL, OpenCV) is imported in run_gui after the splash is shown
# and in run_cli, since it takes most of the startup time


USE_RESOURCES = True  # Set to True to use resources.qrc

RES_PREFIX = ""

if USE_RESOURCES:
    import rc_resources
    RES_PREFIX = ":/"

def run_gui(input_path=None, output_folder=None):
    """Start the GUI. input_path (raster or detections shapefile) and output_folder prefill the
    form, e.g. when the app is opened from the QGIS plugin."""
    import os
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Fusion"

    # Set explicit AppUserModelID on Windows so taskbar uses the app icon
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("CIAT.TropicalForages.ForagesROIs.1.0")
        except Exception:
            pass

    from PySide6.QtWidgets import QApplication, QSplashScreen
    from PySide6.QtGui import QIcon, QPixmap, QColor
    from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterType
    from PySide6.QtCore import QUrl, Qt, QTimer

    app = QApplication(sys.argv)

    app.setApplicationName("ForagesROIs")
    app.setOrganizationName("Tropical Forages Program | Alliance Bioversity International & CIAT")
    app.setApplicationVersion("1.0.0")

    app.styleHints().setColorScheme(Qt.ColorScheme.Light)

    # Icon, splash and view.qml come from resources.qrc (":/...") or from files next to this script
    from PySide6.QtCore import QFile
    base_dir = os.path.dirname(os.path.abspath(__file__))
    def res_path(name):
        return RES_PREFIX + name if USE_RESOURCES else os.path.join(base_dir, name)

    icon_path = res_path("res/icon.png")
    if QFile.exists(icon_path):  # QFile also checks ":/" resource paths
        app.setWindowIcon(QIcon(icon_path))

    # Splash screen with the current startup step, written in the free space at the bottom left
    class Splash(QSplashScreen):
        def drawContents(self, painter):
            painter.setPen(QColor(200, 210, 205))
            painter.drawText(self.rect().adjusted(70, 0, 0, -25), Qt.AlignBottom | Qt.AlignLeft, self.message())

    splash = None
    splash_path = res_path("res/splash.png")
    if QFile.exists(splash_path):
        pixmap = QPixmap(splash_path)
        splash = Splash(pixmap, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        splash.show()
        splash.showMessage("Loading libraries...")
        app.processEvents()

    from interface.interface import ProcessorInterface
    from interface.raster_preview_item import RasterPreviewItem
    processorInterface = ProcessorInterface()

    if splash:
        splash.showMessage("Loading interface...")
    qmlRegisterType(RasterPreviewItem, "CustomComponents", 1, 0, "RasterPreviewItem")

    engine = QQmlApplicationEngine()
    engine.quit.connect(app.quit)
    engine.rootContext().setContextProperty("processorInterface", processorInterface)

    if USE_RESOURCES:
        engine.load(QUrl("qrc:/view.qml"))
    else:
        engine.load(QUrl.fromLocalFile(res_path("view.qml")))

    root_objects = engine.rootObjects()
    if not root_objects:
        print("QML load failed")
        if splash:
            splash.close()
        sys.exit(1)

    # Prefill the form from the command line (--input, --output without --cli)
    root = root_objects[0]
    if output_folder:
        root.setProperty("outputFolderPath", os.path.normpath(output_folder))
    if input_path:
        input_path = os.path.normpath(input_path)
        if input_path.lower().endswith(".shp"):
            root.setProperty("inputShapefilePath", input_path)
        else:
            root.setProperty("inputRasterPath", input_path)
            QTimer.singleShot(0, lambda: processorInterface.previewRaster(input_path, output_folder or ""))

    # Close the splash as soon as the window is ready
    if splash:
        main_win = root_objects[0]
        splash.close()
        main_win.show()
        main_win.raise_()
        main_win.requestActivate()

    sys.exit(app.exec())

def run_cli(args):
    print("working")

    from interface.interface import ProcessorInterface
    processorInterface = ProcessorInterface()
    processorInterface.loadModelsFolder()  # same models folder as the GUI (saved setting or default)

    if args.download_models:
        import custom_processor
        folder = processorInterface.writable_models_folder(custom_processor.MODEL_PATH)
        shown = set()
        def report(info):  # one line every 10 %
            step = info["progress"] // 10
            if step not in shown:
                shown.add(step)
                print(info["status"])
        try:
            names = custom_processor.download_models(folder, progress_callback=report)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        processorInterface.setModelsFolder(folder)  # use and remember this folder, also for the GUI
        print(f"Downloaded: {', '.join(names)}" if names else "All models found.", f"Models folder: {folder}", sep="\n")
        if not args.cli:
            return

    # parameters = {
    #     "task": "detection",
    #     "input_file": args.input,
    #     "output_folder": args.output
    # }

    #python ForagesROIs.py --cli --input "\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples\146.tif" --output "\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples_outputs"
    #python ForagesROIs.py --cli --input "\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples\187.tif" --output "\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples_outputs"
    #python ForagesROIs.py --cli --input "\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples\187.tif" --output "\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples_outputs\current.shp"
    #python ForagesROIs.py --cli --input "\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples\full\annotation_0_clipped_RGB.tif" --output "\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples_outputs\current.shp"
    #python ForagesROIs.py --cli --input "\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples\full\annotation_0_clipped_RGB.tif" --output "\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples_outputs\current_final.shp"
    #python ForagesROIs.py --cli --input "\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples\full\annotation_0_clipped_RGB.tif" --output "\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples_outputs\output.shp"
    #python ForagesROIs.py --cli --input "//catalogue.cgiarad.org/AcceleratedBreedingInitiative/1.Data/36. Dataset ROI/training/samples/full/annotation_0_clipped_RGB.tif" --output "//catalogue.cgiarad.org/AcceleratedBreedingInitiative/1.Data/36. Dataset ROI/training/samples_outputs/output_qgis_python.shp"
    #python ForagesROIs.py --cli --task plot_numbering --input "//catalogue.cgiarad.org/AcceleratedBreedingInitiative/1.Data/36. Dataset ROI/training/samples_outputs/edited.shp" --output "//catalogue.cgiarad.org/AcceleratedBreedingInitiative/1.Data/36. Dataset ROI/training/samples_outputs/enumeration.shp"
    #python ForagesROIs.py --cli --task tiling_detection --input "\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples\full\SOC_2024-11-18.tif"  --output "//catalogue.cgiarad.org/AcceleratedBreedingInitiative/1.Data/36. Dataset ROI/training/samples_outputs/SOC_2024-11-18_crop.shp"
    #python ForagesROIs.py --cli --input "//catalogue.cgiarad.org/AcceleratedBreedingInitiative/1.Data/36. Dataset ROI/training/samples/full/annotation_2_clipped_RGB.tif" --output "//catalogue.cgiarad.org/AcceleratedBreedingInitiative/1.Data/36. Dataset ROI/training/samples_outputs/output_2.shp"
    #python ForagesROIs.py --cli --input "\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples\full\annotation_0_clipped_RGB.tif" --output "\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples_outputs\exec.shp"

    #python ForagesROIs.py --cli --task tiling_detection_only --input "D:\local_mydata\rois\sample\annotation_0_clipped_RGB.tif" --output "D:\local_mydata\rois\samples_outputs_tests\00_detection.shp"


    #python ForagesROIs.py --cli --task postprocessing --input "D:\local_mydata\rois\samples_outputs_tests\00_detection.shp" --output "D:\local_mydata\rois\samples_outputs_tests\00_postprocess.shp"

    #python ForagesROIs.py --cli --task plot_numbering --align --serpentine --input "D:\local_mydata\rois\samples_outputs_tests\00_postprocess.shp" --output "D:\local_mydata\rois\samples_outputs_tests\00_numbering.shp"

    #python ForagesROIs.py --cli --task plot_numbering --align --input "D:\local_mydata\rois\samples_outputs_tests\00_postprocess.shp" --output "D:\local_mydata\rois\samples_outputs_tests\00_numbering_normal.shp"

    #python ForagesROIs.py --cli --task plot_numbering --align --serpentine --input "D:\local_mydata\rois\samples_outputs_tests\00_postprocess.shp" --output "D:\local_mydata\rois\samples_outputs_tests\00_numbering_serpentine.shp"


    #python ForagesROIs.py --cli --task tiling_detection --input "D:\local_mydata\rois\sample\annotation_0_clipped_RGB.tif" --output "D:\local_mydata\rois\samples_outputs_tests\00_full.shp"



    #python ForagesROIs.py --cli --task tiling_detection --input "D:\local_mydata\rois\sample\annotation_7_clipped_RGB.tif" --output "D:\local_mydata\rois\samples_outputs_tests\07_full.shp"

    #python ForagesROIs.py --cli --task tiling_detection --input "D:\local_mydata\rois\sample\annotation_2_clipped_RGB.tif" --output "D:\local_mydata\rois\samples_outputs_tests\02_full.shp"
        

    # parameters = {
    #     "task": "detection",
    #     "input_file": r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples\146.tif",
    #     "output_folder": r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples_outputs"
    # }

    parameters = {
        "task": args.task,
        "input_file": args.input,
        "output_folder": args.output
    }

    extra_parameters = {k: v for k, v in vars(args).items() if v is not None}

    parameters.update(extra_parameters)

    print(parameters)

    processorInterface.process(parameters)

    # Wait for the worker to finish to avoid QThread destruction error
    results = {}
    if hasattr(processorInterface, 'worker') and processorInterface.worker is not None:
        processorInterface.worker.wait()
        results = getattr(processorInterface.worker, "results", {}) or {}
    results.setdefault("task", args.task)  # also on errors, where the result has only a message

    # --json: the last line of the output is a JSON summary (status, output files, counts), for
    # scripts and AI agents. The exit code is 0 when the task completed and 1 otherwise.
    from utils.result_summary import summarize
    summary = summarize(results)
    if args.json:
        import json
        print(json.dumps(summary, ensure_ascii=False))
    sys.exit(0 if summary["status"] == "completed" else 1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="ForagesROIs: detect, number and measure forage plants in UAV orthomosaics (GeoTIFF, "
                    "1 to 3 cm per pixel). Without --cli the window opens.",
        epilog="Tasks (--task, with --cli):\n"
               "  crop_detection            raster -> <raster>_crops.shp (field polygons)\n"
               "  tiling_detection_on_crop  raster -> <raster>_on_crop.shp (plants inside <raster>_crops.shp of the output folder, numbered per field)\n"
               "  tiling_detection          raster -> <raster>.shp (plants on the whole raster, numbered)\n"
               "  tiling_detection_only     raster -> <raster>.shp (plants, not numbered)\n"
               "  plot_numbering            shapefile -> the same shapefile, numbered again with other options (a .shp --output writes a new file)\n"
               "  postprocessing            shapefile -> <input>_postprocess.shp (duplicates removed, not numbered)\n"
               "  plant_traits              --input shapefile --raster raster -> <input>_traits.shp and .csv\n"
               "  export_coco               --input shapefile --raster raster -> <input>_coco/ (COCO dataset of 1024 px RGB tiles with the boxes)\n"
               "\n"
               "Typical run: crop_detection, then tiling_detection_on_crop, then plant_traits, with the same --output folder.\n"
               "Add --json for a JSON summary as the last line; the exit code is 0 on success.\n"
               "Example: ForagesROIs.exe --cli --task crop_detection --input D:\\flights\\trial.tif --output D:\\results --json\n"
               "Guide for scripts and AI agents: https://afruizh.github.io/forages_rois/agents")
    parser.add_argument("--cli", action="store_true", help="Run a task without the window (see Tasks below).")
    parser.add_argument("--task", type=str, help="Task for --cli (default tiling_detection); see Tasks below.")
    parser.add_argument("--input", type=str, help="Input for CLI processing; without --cli it prefills the GUI input (raster or shapefile).")
    parser.add_argument("--output", type=str, help="Output for CLI processing (folder, or a .shp path); without --cli it prefills the GUI output folder.")
    parser.add_argument("--align", action="store_true", help="Kept for compatibility; has no effect.")
    parser.add_argument("--serpentine", action="store_true", help="For plot_numbering: reverse the direction every other row (off unless given).")
    parser.add_argument("--start_corner", type=str, choices=["NW", "NE", "SW", "SE"], help="Corner where plot numbering starts (default NW).")
    parser.add_argument("--swap_axes", action="store_true", help="Number along the other grid axis when rows are detected as columns.")
    parser.add_argument("--empty_plots", action="store_true", help="For tiling_detection, tiling_detection_on_crop and plot_numbering: number planting positions, so that a missing plant gets an empty plot with its own number (default: only the detected plants are numbered).")
    parser.add_argument("--raster", type=str, help="Orthomosaic for --task plant_traits and export_coco (the ROIs shapefile is --input).")
    parser.add_argument("--crop_shapefile", type=str, help="Crop fields for tiling_detection_on_crop and export_coco (default: <raster>_crops.shp in the output folder).")
    parser.add_argument("--index", type=str, choices=["RGBVI", "ExG", "NDVI", "NDRE", "GNDVI"], help="Vegetation index that separates plant from soil in plant_traits (default NDVI when the raster has a near infrared band, otherwise RGBVI). NDVI, NDRE and GNDVI need a near infrared band; NDRE also a red edge band.")
    parser.add_argument("--nir_band", type=int, help="For plant_traits: number of the near infrared band (default: found from the band names, such as NIR or 840 nm).")
    parser.add_argument("--rededge_band", type=int, help="For plant_traits: number of the red edge band (default: found from the band names, such as RedEdge or 730 nm).")
    parser.add_argument("--threshold", type=float, help="Fixed index threshold for plant_traits (default: automatic, Otsu per field).")
    parser.add_argument("--confidence", type=float, help="Plant detection confidence threshold, 0 to 1 (default 0.10).")
    parser.add_argument("--bands", type=str, help="Raster bands used as red, green and blue, for example 3,2,1 for a multispectral orthomosaic ordered blue, green, red (default: guessed from the band colours and names, otherwise 1,2,3).")
    parser.add_argument("--no_resample", action="store_true", help="For plant detection: do not resample rasters finer than the model resolution (by default rasters finer than 1.1 cm per pixel are resampled to 1.5 cm).")
    parser.add_argument("--clear_cache", type=str, choices=["default", "output"], help="Delete the default cache, or the cache of the --output folder; alone it only clears, with --cli it clears before the task.")
    parser.add_argument("--json", action="store_true", help="With --cli: print a JSON summary as the last line (status, task, message, output_file, skipped, counts of plants, empty plots, fields and classes; csv_file for plant_traits). The exit code is 0 on success and 1 on error.")
    parser.add_argument("--download_models", action="store_true", help="Download the missing models from Hugging Face into the models folder; alone it only downloads, with --cli it downloads before the task.")

    args = parser.parse_args()

    if args.clear_cache:
        from utils.cachemanager import cache_folder, clear_cache
        if args.clear_cache == "output" and not args.output:
            print("Error: --clear_cache output needs --output (the output folder).")
            sys.exit(1)
        folder = cache_folder(args.output if args.clear_cache == "output" else None)
        clear_cache(folder)
        print(f"Cache cleared: {folder}")
        if not args.cli and not args.download_models:
            sys.exit(0)

    if args.download_models and not args.cli:
        run_cli(args)  # only download the models
    elif args.cli:
        if not args.task:
            args.task = "tiling_detection"
        if not args.input:
            print("Error: --input is required in CLI mode.")
            sys.exit(1)
        elif not args.output:
            print("Error: --output is required in CLI mode.")
            sys.exit(1)
        run_cli(args)
    else:
        run_gui(input_path=args.input, output_folder=args.output)







    


