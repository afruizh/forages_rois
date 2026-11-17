import os
import shutil
import hashlib

# Where cached files go (tiles, detections, previews, RGB views):
# - with an output folder: <output folder>/.cache_forages_rois, so a rerun reuses it;
# - without one: the default cache, %LOCALAPPDATA%\ForagesROIs\cache (local application data, as in
#   most desktop software; not Documents, which is often synced to the cloud), or
#   <folder>/.cache_forages_rois when the user chose a folder (File > Cache > Set Default Cache Folder).
# Never next to the input raster.
CACHE_DIR_NAME = ".cache_forages_rois"


def default_cache_folder():
    saved = ""
    try:
        from utils.app_settings import settings
        saved = settings().value("cache_folder", "")
    except ImportError:
        pass
    if saved:
        return os.path.normpath(os.path.join(saved, CACHE_DIR_NAME))
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.normpath(os.path.join(base, "ForagesROIs", "cache"))


def cache_folder(output_folder=None):
    """Cache of an output folder (a .shp path means its folder), or the default cache without one."""
    if output_folder:
        if output_folder.lower().endswith(".shp"):
            output_folder = os.path.dirname(output_folder)
        return os.path.normpath(os.path.join(output_folder, CACHE_DIR_NAME))
    return default_cache_folder()


def folder_size(path):
    """Total size in bytes of the files under path (0 if it does not exist)."""
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def clear_cache(path):
    """Delete a cache folder. Only folders named .cache_forages_rois or the default cache are
    deleted, so a wrong path can never remove other data."""
    path = os.path.normpath(path)
    if os.path.basename(path) != CACHE_DIR_NAME and path != default_cache_folder():
        raise ValueError(f"Not a ForagesROIs cache folder:\n{path}")
    if os.path.exists(path):
        shutil.rmtree(path)


class CacheManager():

    def __init__(self, project_path=None, cache_dir_name=CACHE_DIR_NAME, temp_raster_name="_temp_raster.tif", parameters=None):

        self.cache_dir_name = cache_dir_name
        self.temp_raster_name = temp_raster_name
        self.parameters = parameters

        if project_path:
            self.base_path = os.path.join(project_path, self.cache_dir_name)
            self.base_path = os.path.normpath(self.base_path)
        else:
            self.base_path = default_cache_folder()

    def compute_key(self, *args):
        return hashlib.md5("_".join(map(str, args)).encode()).hexdigest()

    def get_cache_folder_path(self, stage, key):
        folder = os.path.join(self.base_path, stage, key)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    def clean_cache_folder_path(self, stage, key):

        folder = self.get_cache_folder_path(stage, key)
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(folder)

        return True

    def exists(self, stage, key):
        folder = self.get_cache_folder_path(stage, key)
        return os.path.exists(folder) and len(os.listdir(folder)) > 0
    
    def clean_cache_folder(self):

        if os.path.exists(self.base_path):
            for root, dirs, files in os.walk(self.base_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.base_path)

        return True
    
    def get_cache_folder(self):
        os.makedirs(self.base_path, exist_ok=True)
        return self.base_path
    
    def get_temp_raster_path(self):
        return os.path.join(self.base_path, self.temp_raster_name)
    
    def remove_temp_raster(self):
        temp_raster_path = self.get_temp_raster_path()
        if os.path.exists(temp_raster_path):
            os.remove(temp_raster_path)
        return True
