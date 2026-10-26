# -*- coding: utf-8 -*-
"""
ForagesROIs Launcher: a light QGIS plugin for the ForagesROIs desktop application.

All processing happens in the ForagesROIs application (installed separately), so this plugin
has no dependencies beyond QGIS. It can:
  - Open in ForagesROIs: start the application with the selected raster layer prefilled.
  - Load ForagesROIs results: add output shapefiles to the project. QGIS applies the style
    automatically because the application writes a .qml file next to each output.
  - Set ForagesROIs application: choose ForagesROIs.exe (saved in the QGIS settings).
"""

import os

from qgis.PyQt.QtCore import QProcess
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QFileDialog, QMessageBox
from qgis.core import QgsProject, QgsRasterLayer, QgsSettings, QgsVectorLayer

MENU = "&ForagesROIs"
SETTINGS_EXE = "foragesrois_launcher/exe_path"
SETTINGS_RESULTS_DIR = "foragesrois_launcher/results_dir"


def check_exe(path):
    """Return an error message for an invalid application path, or None if it is valid."""
    if not path:
        return "The ForagesROIs application is not set."
    if not os.path.isfile(path):
        return f"The ForagesROIs application was not found:\n{path}"
    if not path.lower().endswith(".exe"):
        return f"Select the ForagesROIs executable (ForagesROIs.exe), not:\n{path}"
    return None


def build_arguments(input_path=None, output_folder=None):
    """Command line arguments that open the GUI with the input and output prefilled."""
    args = []
    if input_path:
        args += ["--input", os.path.normpath(input_path)]
    if output_folder:
        args += ["--output", os.path.normpath(output_folder)]
    return args


def load_results(paths, project=None):
    """Add result shapefiles to the project. Returns (loaded layers, paths that failed)."""
    project = project or QgsProject.instance()
    loaded, failed = [], []
    for path in paths:
        # The default style (same name, .qml) is loaded automatically by QgsVectorLayer
        layer = QgsVectorLayer(path, os.path.splitext(os.path.basename(path))[0], "ogr")
        if layer.isValid():
            project.addMapLayer(layer)
            loaded.append(layer)
        else:
            failed.append(path)
    return loaded, failed


class ForagesROIsLauncherPlugin(object):

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.toolbar = None

    def initGui(self):
        icon = QIcon(os.path.join(self.plugin_dir, "icon.png"))
        self.toolbar = self.iface.addToolBar("ForagesROIs Launcher")
        self.toolbar.setObjectName("ForagesROIsLauncher")

        for text, slot, on_toolbar in [
            ("Open in ForagesROIs", self.open_app, True),
            ("Load ForagesROIs results...", self.load_results_dialog, True),
            ("Set ForagesROIs application...", self.set_exe_dialog, False),
        ]:
            action = QAction(icon, text, self.iface.mainWindow())
            action.triggered.connect(slot)
            self.iface.addPluginToMenu(MENU, action)
            if on_toolbar:
                self.toolbar.addAction(action)
            self.actions.append(action)

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(MENU, action)
            self.iface.removeToolBarIcon(action)
        self.actions = []
        if self.toolbar:
            self.toolbar.deleteLater()
            self.toolbar = None

    # ---------------- Settings ----------------

    def exe_path(self):
        return QgsSettings().value(SETTINGS_EXE, "", type=str)

    def set_exe_dialog(self):
        """Ask for ForagesROIs.exe. Returns the valid path, or None if cancelled or invalid."""
        current = self.exe_path()
        path, _ = QFileDialog.getOpenFileName(
            self.iface.mainWindow(), "Select the ForagesROIs application",
            os.path.dirname(current) if current else "", "ForagesROIs (ForagesROIs.exe);;Executables (*.exe)")
        if not path:
            return None
        error = check_exe(path)
        if error:
            QMessageBox.warning(self.iface.mainWindow(), "ForagesROIs", error)
            return None
        QgsSettings().setValue(SETTINGS_EXE, path)
        self.iface.messageBar().pushSuccess("ForagesROIs", f"Application set: {path}")
        return path

    # ---------------- Actions ----------------

    def open_app(self):
        """Start ForagesROIs as a separate process with the selected raster layer prefilled."""
        exe = self.exe_path()
        if check_exe(exe):
            QMessageBox.information(self.iface.mainWindow(), "ForagesROIs",
                                    "Select the ForagesROIs application (ForagesROIs.exe) first.")
            exe = self.set_exe_dialog()
            if not exe:
                return

        layer = self.iface.activeLayer()
        input_path = None
        if isinstance(layer, QgsRasterLayer) and os.path.isfile(layer.source()):
            input_path = layer.source()
        else:
            self.iface.messageBar().pushInfo(
                "ForagesROIs", "No raster layer selected: the application opens without an input.")

        # Start detached so QGIS stays usable; the app runs from its own folder
        started = QProcess.startDetached(exe, build_arguments(input_path), os.path.dirname(exe))
        if isinstance(started, tuple):  # PyQt returns (ok, pid)
            started = started[0]
        if not started:
            QMessageBox.critical(self.iface.mainWindow(), "ForagesROIs", f"Could not start:\n{exe}")
            return
        self.iface.messageBar().pushInfo(
            "ForagesROIs", "ForagesROIs opened. When processing finishes, use 'Load ForagesROIs results'.")

    def load_results_dialog(self):
        """Pick output shapefiles and add them to the project with their style."""
        settings = QgsSettings()
        start_dir = settings.value(SETTINGS_RESULTS_DIR, "", type=str)
        layer = self.iface.activeLayer()
        if not start_dir and isinstance(layer, QgsRasterLayer):
            start_dir = os.path.dirname(layer.source())

        paths, _ = QFileDialog.getOpenFileNames(
            self.iface.mainWindow(), "Load ForagesROIs results", start_dir, "Shapefiles (*.shp)")
        if not paths:
            return
        settings.setValue(SETTINGS_RESULTS_DIR, os.path.dirname(paths[0]))

        loaded, failed = load_results(paths)
        if loaded:
            self.iface.messageBar().pushSuccess("ForagesROIs", f"Loaded {len(loaded)} layer(s).")
        if failed:
            QMessageBox.warning(self.iface.mainWindow(), "ForagesROIs",
                                "These files could not be loaded:\n" + "\n".join(failed))
