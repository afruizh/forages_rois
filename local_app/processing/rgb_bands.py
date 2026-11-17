"""RGB view of rasters that are not 8 bit RGB (for example multispectral orthomosaics with 5 bands,
or 16 bit and reflectance rasters).

The rest of the app reads bands 1 to 3 as R, G, B in 8 bit. Instead of changing every reader, a
small GDAL virtual raster (VRT, an XML file without pixels) is written with the 3 chosen bands
stretched to 8 bit, and its path is used in place of the original raster. Normal 8 bit RGB or RGBA
rasters are used as they are. For plant detection the view can also be resampled to the resolution
the model was trained on, when the raster is much finer (custom_processor.MODEL_RESOLUTION).
"""
import os
import re
import numpy as np
from osgeo import gdal

gdal.DontUseExceptions()

# Each band is stretched from its minimum to its maximum over the valid pixels (not masked and not
# the largest value of the type, used as nodata, for example 65535), as the training tiles were made
# (scale_and_save_raster in project_roi/test_training/preprocessing_qgis.ipynb, bands 3, 2, 1).
STATS_SIZE = 1024               # longest side of the reduced read used for the minimum and maximum


def _band_name(band):
    name = band.GetDescription().strip()
    if not name:
        name = gdal.GetColorInterpretationName(band.GetColorInterpretation())
    return "" if name in ("Undefined", "Gray") else name


def band_labels(path):
    """Labels for the band lists of the GUI, for example ["1: Blue", "2: Green", "3: Red", "4: 4"]."""
    ds = gdal.Open(path)
    if ds is None:
        return []
    return [f"{i}: {_band_name(ds.GetRasterBand(i)) or i}" for i in range(1, ds.RasterCount + 1)]


def _colour_from_name(name):
    """'red', 'green', 'blue' or None from a band description or colour interpretation."""
    text = name.lower()
    if any(word in text for word in ("edge", "nir", "infrared")):
        return None
    for colour, letter in (("red", "r"), ("green", "g"), ("blue", "b")):
        if re.search(rf"(^|[^a-z])({colour}|{letter})([^a-z]|$)", text):
            return colour
    match = re.search(r"(\d{3})\s*nm", text)
    if match:
        nm = int(match.group(1))
        return "red" if 620 <= nm <= 700 else "green" if 500 <= nm < 600 else "blue" if 420 <= nm < 500 else None
    return None


def guess_rgb_bands(path):
    """Most likely [red, green, blue] band numbers (1 based): from the colour interpretation of
    the bands, then from their names or wavelengths, otherwise [1, 2, 3] ([1, 1, 1] for one band)."""
    ds = gdal.Open(path)
    if ds is None:
        return [1, 2, 3]
    n = ds.RasterCount
    if n < 3:
        return [1, 1, 1]
    interp = {gdal.GCI_RedBand: "red", gdal.GCI_GreenBand: "green", gdal.GCI_BlueBand: "blue"}
    for read in (lambda b: interp.get(b.GetColorInterpretation()), lambda b: _colour_from_name(b.GetDescription())):
        found = {}
        for i in range(1, n + 1):
            colour = read(ds.GetRasterBand(i))
            if colour and colour not in found:
                found[colour] = i
        if len(found) == 3:
            return [found["red"], found["green"], found["blue"]]
    return [1, 2, 3]


def parse_bands(text):
    """'3,2,1' -> [3, 2, 1]; empty or None -> None (use the guess)."""
    if not text:
        return None
    bands = [int(b) for b in str(text).replace(" ", "").split(",") if b]
    if len(bands) != 3:
        raise ValueError(f"Bands must be three numbers (red, green, blue), for example 3,2,1; got: {text}")
    return bands


def choose_bands(path):
    """True when the user should choose the bands: more than 3 bands, unless band 4 is an alpha band."""
    ds = gdal.Open(path)
    if ds is None:
        return False
    n = ds.RasterCount
    return n > 4 or (n == 4 and ds.GetRasterBand(4).GetColorInterpretation() != gdal.GCI_AlphaBand)


def needs_rgb_view(path, bands):
    """False for an 8 bit raster read as bands 1, 2, 3 (what the app expects)."""
    ds = gdal.Open(path)
    return ds is not None and not (bands == [1, 2, 3] and ds.GetRasterBand(1).DataType == gdal.GDT_Byte)


def _valid_values(ds, band_number):
    """Values of one band from a reduced read, without masked pixels and without the largest value
    of an integer type (nodata in multispectral orthomosaics)."""
    band = ds.GetRasterBand(band_number)
    scale = min(1.0, STATS_SIZE / max(ds.RasterXSize, ds.RasterYSize))
    w, h = max(1, int(ds.RasterXSize * scale)), max(1, int(ds.RasterYSize * scale))
    values = band.ReadAsArray(buf_xsize=w, buf_ysize=h)
    valid = band.GetMaskBand().ReadAsArray(buf_xsize=w, buf_ysize=h) > 0
    if np.issubdtype(values.dtype, np.integer):
        valid &= values != np.iinfo(values.dtype).max
    else:
        valid &= np.isfinite(values)
    return values[valid].astype(np.float64)


def _stretch(ds, band_number):
    """(minimum, maximum) of one band from a reduced read, without masked pixels and without the
    largest value of an integer type (nodata in multispectral orthomosaics)."""
    values = _valid_values(ds, band_number)
    if values.size == 0:
        return 0.0, 255.0
    low, high = values.min(), values.max()
    return float(low), float(high if high > low else low + 1)


def rgb_raster(path, bands=None, folder=None, pixel_m=None, target_m=None):
    """Path of an 8 bit RGB view of path with the given bands ([r, g, b], 1 based; None: guess),
    resampled to target_m metres per pixel when pixel_m and target_m are given (see
    custom_processor.MODEL_RESOLUTION). Returns path itself when no view is needed. The VRT is
    written in folder (default: the default cache) /rgb_<r>_<g>_<b>[_<target>mm]/ with the same
    file name as the raster, so output names do not change, and it is reused while it is newer
    than the raster."""
    from utils.cachemanager import cache_folder
    bands = bands or guess_rgb_bands(path)
    resample = bool(pixel_m and target_m)
    if not resample and not needs_rgb_view(path, bands):
        return path
    ds = gdal.Open(path)
    if any(b < 1 or b > ds.RasterCount for b in bands):
        raise ValueError(f"The raster has {ds.RasterCount} bands; bands {bands} do not exist.")
    name = "rgb_" + "_".join(str(b) for b in bands) + (f"_{target_m * 1000:.0f}mm" if resample else "")
    folder = os.path.join(folder or cache_folder(None), name)
    os.makedirs(folder, exist_ok=True)
    vrt = os.path.join(folder, os.path.splitext(os.path.basename(path))[0] + ".vrt")
    if os.path.exists(vrt) and os.path.getmtime(vrt) >= os.path.getmtime(path):
        return vrt
    # No overviews on purpose: crop detection reads the whole raster reduced, and averaged
    # overviews changed its result (an extra field on the 5 band test); the preview keeps its
    # own reduced copy instead (raster_preview_item.BASE_SIZE)
    options = ["-of", "VRT", "-ot", "Byte", "-a_nodata", "none", "-colorinterp", "red,green,blue"]
    for b in bands:
        options += ["-b", str(b)]
    if ds.GetRasterBand(bands[0]).DataType == gdal.GDT_Byte:
        options += ["-scale", "0", "255", "0", "255"]  # 8 bit: values kept as they are
    else:
        for i, b in enumerate(bands, start=1):
            low, high = _stretch(ds, b)
            options += [f"-scale_{i}", repr(low), repr(high), "0", "255"]
    if resample:
        scale = pixel_m / target_m
        options += ["-outsize", str(max(1, round(ds.RasterXSize * scale))), str(max(1, round(ds.RasterYSize * scale))),
                    "-r", "average"]
    out = gdal.Translate(vrt, ds, options=options)
    if out is None:
        raise ValueError(f"Could not create the RGB view of {path} with bands {bands}.")
    out = None  # close to write the VRT
    print(f"RGB view of {os.path.basename(path)} with bands {bands}"
          + (f", resampled from {pixel_m * 100:.2f} to {target_m * 100:.2f} cm per pixel" if resample else "") + f": {vrt}")
    return vrt
