import os
import sys
# Add the directory of the executable to PATH at runtime if frozen
if getattr(sys, 'frozen', False):
    exe_dir = os.path.dirname(sys.executable)
    os.environ["PATH"] = exe_dir + os.pathsep + os.environ.get("PATH", "")

import geopandas

if __name__ == "__main__":

    # Load shapefile
    gdf = geopandas.read_file(r"\\CATALOGUE.CGIARAD.ORG\AcceleratedBreedingInitiative\1.Data\36. Dataset ROI\training\samples_outputs\edited.shp")
    # save shapefile
    gdf.to_file("./local/test.shp")