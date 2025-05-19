import subprocess
import sys
import os


# The contents of crazy.py as a string, using f-string formatting for variables
raster_path = r"D:\local_mydata\rois\sample\annotation_0_clipped_RGB.tif"
output_path = r"D:\local_mydata\rois\samples_outputs\00.shp"
cwd_path = r"C:\Users\afrhu\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\foragesrois\ForagesROIs"

env = os.environ.copy()
env["PATH"] = cwd_path

env["GDAL_DATA"] = os.path.join(cwd_path, "gdal_data")
env["PROJ_LIB"] = os.path.join(cwd_path, "proj_data")

os.environ["GDAL_DATA"] = os.path.join(cwd_path, "gdal_data")
os.environ["PROJ_LIB"] = os.path.join(cwd_path, "proj_data")


crazy_py_code = f'''import os
import subprocess

raster_path = r"{raster_path}"
output_path = r"{output_path}"


print(os.environ["GDAL_DATA"])
print(os.environ["PROJ_LIB"])

cmd = ['ForagesROIs.exe', '--cli', '--input', os.path.normpath(raster_path), '--output', os.path.normpath(output_path)]
print(f'Running: {{" ".join(cmd)}}')
result = subprocess.run(cmd, capture_output=True, text=True, encoding='latin1')
print('STDOUT:')
print(result.stdout)
print('STDERR:')
print(result.stderr)
'''

result = subprocess.run([sys.executable, "-c", crazy_py_code], capture_output=True, text=True, env=env, cwd=cwd_path, encoding='latin1')
# creationflags=subprocess.CREATE_NEW_CONSOLE
#result = subprocess.run(["python", crazy_py_code], capture_output=True, text=True, env=env, cwd=cwd_path, encoding='latin1')

print("CRAZY.PY STDOUT:")
print(result.stdout)
print("CRAZY.PY STDERR:")
print(result.stderr)

