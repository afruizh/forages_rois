import os
import subprocess

raster_path =r"D:\local_mydata\rois\sample\annotation_0_clipped_RGB.tif"
output_path = r"D:\local_mydata\rois\samples_outputs\00.shp"


cwd_path = r"C:\Users\afrhu\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\foragesrois\ForagesROIs"

env = os.environ.copy()
env["PATH"] = cwd_path
os.environ["PATH"] = cwd_path

os.environ["GDAL_DATA"] = os.path.join(cwd_path, "gdal_data")   # Must exist
os.environ["PROJ_LIB"] = os.path.join(cwd_path, "proj_data")       # Or "share/proj" depending on your layout


env["GDAL_DATA"] = os.path.join(cwd_path, "gdal_data")   # Must exist
env["PROJ_LIB"] = os.path.join(cwd_path, "proj_data")       # Or "share/proj" depending on your layout



cmd = ['ForagesROIs.exe', '--cli', '--input', os.path.normpath(raster_path), '--output', os.path.normpath(output_path)]
print(f'Running: {" ".join(cmd)}')
result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd_path, env=env, encoding='latin1')
print('STDOUT:')
print(result.stdout)
print('STDERR:')
print(result.stderr)
