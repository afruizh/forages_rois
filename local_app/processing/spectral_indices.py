"""Vegetation indices of the original raster bands, for plant traits.

The detection works on an 8 bit RGB view (rgb_bands), but plant traits read the original raster, so
that a multispectral orthomosaic also gives the indices of its red edge and near infrared bands.
The bands are found from their names or wavelengths, as rgb_bands does for red, green and blue.

    bands = find_bands(path, rgb=[3, 2, 1])      # {"red": 3, "green": 2, "blue": 1, "rededge": 4, "nir": 5}
    arrays = read_bands(ds, bands, x0, y0, w, h) # float arrays by band name, and the valid pixels
    ndvi = compute_index("NDVI", arrays)
"""
import re
import numpy as np
from osgeo import gdal

gdal.DontUseExceptions()

# name: (bands it needs, formula). RGB indices first, so they stay the default without NIR.
INDICES = {
    "RGBVI": (("red", "green", "blue"), lambda b: (b["green"] ** 2 - b["red"] * b["blue"]) / (b["green"] ** 2 + b["red"] * b["blue"])),
    "ExG":   (("red", "green", "blue"), lambda b: (2 * b["green"] - b["red"] - b["blue"]) / (b["red"] + b["green"] + b["blue"])),
    "NDVI":  (("nir", "red"),           lambda b: (b["nir"] - b["red"]) / (b["nir"] + b["red"])),
    "NDRE":  (("nir", "rededge"),       lambda b: (b["nir"] - b["rededge"]) / (b["nir"] + b["rededge"])),
    "GNDVI": (("nir", "green"),         lambda b: (b["nir"] - b["green"]) / (b["nir"] + b["green"])),
}
NIR_INDICES = ("NDVI", "NDRE", "GNDVI")   # their canopy means are added as columns when available
DEFAULT_RGB_INDEX, DEFAULT_NIR_INDEX = "RGBVI", "NDVI"


def _band_kind(name):
    """'nir', 'rededge' or None from a band description such as 'NIR', 'Red Edge' or '842 nm'."""
    text = name.lower().replace("-", " ").replace("_", " ")
    if re.search(r"(^|[^a-z])(red ?edge|re)([^a-z]|$)", text):
        return "rededge"
    if re.search(r"(^|[^a-z])(nir|near ?infra ?red|infrared)([^a-z]|$)", text):
        return "nir"
    match = re.search(r"(\d{3,4})\s*nm", text)
    if match:
        nm = int(match.group(1))
        return "rededge" if 700 < nm < 760 else "nir" if 760 <= nm <= 1000 else None
    return None


def find_bands(path, rgb=None, nir=None, rededge=None):
    """Band numbers (1 based) by name. rgb: [red, green, blue] (None: guessed as for the RGB view);
    nir and rededge: given numbers, otherwise found from the band names (left out when not found)."""
    from processing.rgb_bands import guess_rgb_bands
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Could not open raster: {path}")
    red, green, blue = rgb or guess_rgb_bands(path)
    bands = {"red": red, "green": green, "blue": blue}
    found = {}
    for i in range(1, ds.RasterCount + 1):
        kind = _band_kind(ds.GetRasterBand(i).GetDescription())
        if kind and kind not in found:
            found[kind] = i
    for kind, given in (("nir", nir), ("rededge", rededge)):
        number = given or found.get(kind)
        if number:
            if not 1 <= int(number) <= ds.RasterCount:
                raise ValueError(f"The raster has {ds.RasterCount} bands; band {number} ({kind}) does not exist.")
            bands[kind] = int(number)
    return bands


def available_indices(bands):
    """Index names that can be computed with the given bands."""
    return [name for name, (needs, _) in INDICES.items() if all(b in bands for b in needs)]


def default_index(bands):
    """NDVI when there is a near infrared band (separates plants from soil and dry straw), else RGBVI."""
    return DEFAULT_NIR_INDEX if "nir" in bands else DEFAULT_RGB_INDEX


def read_bands(ds, bands, x0, y0, w, h):
    """(arrays, valid) for a window: float arrays by band name, and the pixels with data in all of
    them (not masked, not the largest value of an integer type, not black background)."""
    arrays, valid = {}, np.ones((h, w), bool)
    total = np.zeros((h, w))
    for name, number in bands.items():
        band = ds.GetRasterBand(number)
        values = band.ReadAsArray(x0, y0, w, h)
        valid &= band.GetMaskBand().ReadAsArray(x0, y0, w, h) > 0
        if np.issubdtype(values.dtype, np.integer):
            valid &= values != np.iinfo(values.dtype).max
        nodata = band.GetNoDataValue()
        if nodata is not None:
            valid &= values != nodata
        arrays[name] = values.astype(np.float64)
        total += arrays[name]
    valid &= np.isfinite(total) & (total > 0)
    return arrays, valid


def compute_index(name, arrays):
    """Index of the arrays of read_bands; pixels where it is undefined get 0."""
    needs, formula = INDICES[name]
    missing = [b for b in needs if b not in arrays]
    if missing:
        raise ValueError(f"{name} needs the {', '.join(missing)} band; the raster does not have it "
                         f"(available indices: {', '.join(available_indices(arrays))}).")
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = formula(arrays)
    return np.nan_to_num(idx, nan=0.0, posinf=0.0, neginf=0.0)
