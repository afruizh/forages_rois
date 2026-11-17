"""File > Preferences: models folder, plant and crop models, GPU, what to do when an output already
exists, and the default cache folder. The values are kept in QSettings (see utils/app_settings);
an empty value means the default. apply_saved_preferences() puts them into custom_processor at
startup, for the GUI and the CLI (where --plant_model, --crop_model and --cpu override them for one run).
The dialog is interface/PreferencesDialog.qml; this object is its "preferences" context property."""
import os

from PySide6.QtCore import QObject, Property, Signal, Slot

from custom_processor import MODEL_FILENAME, CROP_MODEL_FILENAME

# Settings key of each model, with the default file it replaces
MODEL_KEYS = {"plant_model": MODEL_FILENAME, "crop_model": CROP_MODEL_FILENAME}
MODEL_LABELS = {"plant_model": "plant detection model", "crop_model": "crop field model"}
WHEN_OUTPUT_EXISTS = ("ask", "skip", "overwrite")


def when_output_exists():
    """"ask" (default), "skip" or "overwrite"."""
    from utils.app_settings import settings
    value = settings().value("when_output_exists", "ask")
    return value if value in WHEN_OUTPUT_EXISTS else "ask"


def apply_saved_preferences():
    """Put the saved models folder, models and GPU choice into custom_processor. Returns notes on
    saved values that could not be used (for example a model file that was moved)."""
    import custom_processor as cp
    from utils.app_settings import settings
    s, notes = settings(), []
    folder = s.value("models_folder", "")
    if folder and os.path.isdir(folder):
        cp.MODEL_PATH = folder
    for key, name in MODEL_KEYS.items():
        path = s.value(key, "")
        cp.CUSTOM_MODELS[name] = path if path and os.path.isfile(path) else None
        if path and not os.path.isfile(path):
            notes.append(f"The {MODEL_LABELS[key]} chosen in Preferences was not found, the default is used:\n{path}")
    cp.USE_GPU = s.value("use_gpu", True, type=bool)
    return notes


def models_summary():
    """One line with the models and the device in use, for the main window."""
    import custom_processor as cp
    parts = []
    for key, name in MODEL_KEYS.items():
        custom = cp.CUSTOM_MODELS.get(name)
        label = "Plant model" if key == "plant_model" else "Crop model"
        parts.append(f"{label}: {os.path.basename(custom)} (custom)" if custom else f"{label}: default")
    parts.append("GPU when available" if cp.USE_GPU else "CPU only")
    return "   |   ".join(parts)


class Preferences(QObject):
    """Bridge between PreferencesDialog.qml and the settings."""

    summaryChanged = Signal()

    @Property(str, notify=summaryChanged)
    def summary(self):
        return models_summary()

    @Slot(result="QVariantMap")
    def load(self):
        """Current values for the dialog, with the defaults and the GPU state."""
        import onnxruntime as ort
        import custom_processor as cp
        from utils.app_settings import settings
        from utils.cachemanager import default_cache_folder
        s = settings()
        return {
            "models_folder": cp.MODEL_PATH,
            "default_models_folder": cp.DEFAULT_MODEL_PATH,
            "plant_model": cp.CUSTOM_MODELS.get(MODEL_KEYS["plant_model"]) or "",
            "crop_model": cp.CUSTOM_MODELS.get(MODEL_KEYS["crop_model"]) or "",
            "plant_default": MODEL_KEYS["plant_model"],
            "crop_default": MODEL_KEYS["crop_model"],
            "use_gpu": cp.USE_GPU,
            "cuda_available": "CUDAExecutionProvider" in ort.get_available_providers(),
            "last_device": s.value("last_device", ""),
            "when_output_exists": when_output_exists(),
            "cache_folder": s.value("cache_folder", ""),
            "default_cache": default_cache_folder(),
        }

    @Slot(str, str, str, result="QVariantMap")
    def checkModel(self, path, key, folder):
        """Check a model file (path), or the default file of key in folder when path is empty."""
        import custom_processor as cp
        name = MODEL_KEYS[key]
        path = path or os.path.join(folder or cp.MODEL_PATH, name)
        level, text = cp.check_model(path, name)
        return {"level": level, "text": text, "path": path}

    @Slot("QVariantMap", result=str)
    def save(self, values):
        """Save and apply the values of the dialog. Returns an error message, or "" when saved."""
        import custom_processor as cp
        from utils.app_settings import settings
        folder = values.get("models_folder", "") or cp.DEFAULT_MODEL_PATH
        if not os.path.isdir(folder):
            return f"The models folder does not exist:\n{folder}"
        for key in MODEL_KEYS:
            path = values.get(key, "")
            if path and not os.path.isfile(path):
                return f"The {MODEL_LABELS[key]} file does not exist:\n{path}"
        cache = values.get("cache_folder", "")
        if cache and not os.path.isdir(cache):
            return f"The cache folder does not exist:\n{cache}"
        s = settings()
        s.setValue("models_folder", "" if folder == cp.DEFAULT_MODEL_PATH else folder)
        for key in MODEL_KEYS:
            s.setValue(key, values.get(key, ""))
        s.setValue("use_gpu", bool(values.get("use_gpu", True)))
        when = values.get("when_output_exists", "ask")
        s.setValue("when_output_exists", when if when in WHEN_OUTPUT_EXISTS else "ask")
        s.setValue("cache_folder", cache)
        cp.MODEL_PATH = folder
        apply_saved_preferences()
        self.summaryChanged.emit()
        return ""

    @Slot(str, result=str)
    def localPath(self, url):
        """Local path of a file:// URL from a file dialog (network paths too)."""
        from PySide6.QtCore import QUrl
        return os.path.normpath(QUrl(url).toLocalFile()) if url else ""

    @Slot(str)
    def setWhenOutputExists(self, value):
        """From the question shown when an output exists (Don't ask again)."""
        from utils.app_settings import settings
        if value in WHEN_OUTPUT_EXISTS:
            settings().setValue("when_output_exists", value)

    @Slot(result=str)
    def whenOutputExists(self):
        return when_output_exists()
