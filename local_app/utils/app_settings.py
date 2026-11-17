"""Settings kept between sessions: models folder, default cache folder and the last folder used
for each kind of file. On Windows they are stored in the registry under
HKEY_CURRENT_USER\\Software\\ForagesROIs\\ForagesROIs."""
import os


def settings():
    from PySide6.QtCore import QSettings
    return QSettings("ForagesROIs", "ForagesROIs")


def last_folder(kind):
    """Last folder used for a kind of file ("raster", "shapefile", "output", ...), or "" if none
    was saved or it no longer exists (for example a disconnected disk)."""
    folder = settings().value(f"last_folders/{kind}", "")
    return folder if folder and os.path.isdir(folder) else ""


def remember_folder(kind, path):
    """Save the folder of path (a file or a folder) as the last one for kind."""
    if not path:
        return
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    if folder:
        settings().setValue(f"last_folders/{kind}", os.path.normpath(folder))


def start_folder(kind, current_path="", preferred_folder=""):
    """Folder a file dialog should open in: the folder of the path already in the field, then the
    preferred folder (for example the output folder for shapefiles), then the last folder used for
    kind, then Documents."""
    for path in (current_path, preferred_folder):
        if path:
            folder = path if os.path.isdir(path) else os.path.dirname(path)
            if folder and os.path.isdir(folder):
                return folder
    from PySide6.QtCore import QStandardPaths
    return last_folder(kind) or QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
