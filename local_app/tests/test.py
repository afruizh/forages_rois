import os
import sys
# Add the directory of the executable to PATH at runtime if frozen
if getattr(sys, 'frozen', False):
    exe_dir = os.path.dirname(sys.executable)
    os.environ["PATH"] = exe_dir + os.pathsep + os.environ.get("PATH", "")

from osgeo import ogr
import shutil

if __name__ == "__main__":

    # Load shapefile
    input_shp = r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples_outputs\edited.shp"
    
    # Simple reading test
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.Open(input_shp, 0)
    if ds is not None:
        layer = ds.GetLayer()
        print(f"Loaded shapefile with {layer.GetFeatureCount()} features.")
        ds = None
    else:
        print(f"Could not open {input_shp}")
    
    # Save shapefile (by copying)
    output_dir = "./local"
    os.makedirs(output_dir, exist_ok=True)
    output_shp = os.path.join(output_dir, "test.shp")
    
    # Actually saving using ogr
    ds_in = driver.Open(input_shp, 0)
    if ds_in is not None:
        if os.path.exists(output_shp):
            driver.DeleteDataSource(output_shp)
        ds_out = driver.CopyDataSource(ds_in, output_shp)
        print(f"Saved shapefile to {output_shp}")
        ds_in = None
        ds_out = None