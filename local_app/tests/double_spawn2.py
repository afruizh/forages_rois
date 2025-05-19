import subprocess
import os

# The contents of crazy.py as a string, using f-string formatting for variables
raster_path = r"D:\local_mydata\rois\sample\annotation_0_clipped_RGB.tif"
output_path = r"D:\local_mydata\rois\samples_outputs\00.shp"
cwd_path = r"C:\Users\afrhu\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\foragesrois\ForagesROIs"
bat_path = r"C:\Users\afrhu\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\foragesrois\ForagesROIs\run.bat"

# env = os.environ.copy()
# env["PATH"] = cwd_path

# env["GDAL_DATA"] = os.path.join(cwd_path, "gdal_data")
# env["PROJ_LIB"] = os.path.join(cwd_path, "proj_data")

# os.environ["GDAL_DATA"] = os.path.join(cwd_path, "gdal_data")
# os.environ["PROJ_LIB"] = os.path.join(cwd_path, "proj_data")

# # Join commands with & so they run in sequence in a single cmd.exe call
# bat_commands = (
#     f'set PROJ_LIB={env["PROJ_LIB"]} & '
#     f'set GDAL_DATA={env["GDAL_DATA"]} & '
#     f'set PATH={cwd_path} & '
#     f'cd {cwd_path} & '
#     f'echo %PROJ_LIB% & '
#     f'ForagesROIs.exe --cli --input {raster_path} --output {output_path}'
# )

result = subprocess.run(
    #["cmd.exe", "/c", bat_commands],
    #["cmd.exe", "/c", "start", "/wait", "cmd.exe", "/c", bat_commands],
    ["cmd.exe", "/c", "start", "", "/wait",
    bat_path,
    raster_path, output_path],
    capture_output=True,
    text=True
    # env=env
    # cwd=cwd_path
    , encoding='latin1'
    # ,creationflags=subprocess.CREATE_NEW_CONSOLE
    # , shell=False
)

print("BATCH STDOUT:")
print(result.stdout)
print("BATCH STDERR:")
print(result.stderr)