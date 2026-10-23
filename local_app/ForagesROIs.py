import os
import sys
# Add the directory of the executable to PATH at runtime if frozen
if getattr(sys, 'frozen', False):
    exe_dir = os.path.dirname(sys.executable)
    
    # Force correct environment
    os.environ["PROJ_LIB"] = os.path.join(exe_dir, "proj_dir", "share", "proj")
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

from pyproj import datadir
print("proj.db path:", datadir.get_data_dir())

import sys
# import time
import argparse

# from PySide6.QtWidgets import QApplication
# from PySide6.QtGui import QIcon
# from PySide6.QtGui import QPixmap
# from PySide6.QtWidgets import QSplashScreen
# from PySide6.QtQml import QQmlApplicationEngine

from interface.interface import ProcessorInterface


USE_RESOURCES = False  # Set to True to use resources.qrc

RES_PREFIX = ""

if USE_RESOURCES:
    import rc_resources
    RES_PREFIX = ":/"

def run_gui():
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
    from PySide6.QtGui import QIcon, QPixmap
    from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterType
    from PySide6.QtCore import QUrl, Qt, QTimer
    from interface.raster_preview_item import RasterPreviewItem

    app = QApplication(sys.argv)

    app.setApplicationName("ForagesROIs")
    app.setOrganizationName("Tropical Forages Program | Alliance Bioversity International & CIAT")
    app.setApplicationVersion("1.0.0")

    app.styleHints().setColorScheme(Qt.ColorScheme.Light)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(base_dir, "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Professional splash screen
    splash = None
    splash_path = os.path.join(base_dir, "splash.png")
    if os.path.exists(splash_path):
        pixmap = QPixmap(splash_path)
        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        splash.show()
        app.processEvents()

    processorInterface = ProcessorInterface()

    qmlRegisterType(RasterPreviewItem, "CustomComponents", 1, 0, "RasterPreviewItem")

    engine = QQmlApplicationEngine()
    engine.quit.connect(app.quit)
    engine.rootContext().setContextProperty("processorInterface", processorInterface)

    qml_path = os.path.join(base_dir, "view.qml")
    engine.load(QUrl.fromLocalFile(qml_path))

    root_objects = engine.rootObjects()
    if not root_objects:
        print("QML load failed")
        if splash:
            splash.close()
        sys.exit(1)

    # Smooth splash screen transition
    if splash:
        main_win = root_objects[0]
        def close_splash():
            splash.close()
            main_win.show()
            main_win.raise_()
            main_win.requestActivate()

        QTimer.singleShot(1800, close_splash)

    sys.exit(app.exec())

def run_cli(args):
    print("working")

    processorInterface = ProcessorInterface()

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
    if hasattr(processorInterface, 'worker') and processorInterface.worker is not None:
        processorInterface.worker.wait()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the application in GUI or CLI mode.")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode.")
    parser.add_argument("--task", type=str, help="Task performed for CLI processing (used only with --cli).")
    parser.add_argument("--input", type=str, help="Input for CLI processing (used only with --cli).")
    parser.add_argument("--output", type=str, help="Output for CLI processing (used only with --cli).")
    parser.add_argument("--align", action="store_true", help="Run in CLI mode.")
    parser.add_argument("--serpentine", action="store_true", help="Run in CLI mode.")
    parser.add_argument("--start_corner", type=str, choices=["NW", "NE", "SW", "SE"], help="Corner where plot numbering starts (default NW).")
    parser.add_argument("--swap_axes", action="store_true", help="Number along the other grid axis when rows are detected as columns.")

    args = parser.parse_args()

    if args.cli:
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
        run_gui()







    


