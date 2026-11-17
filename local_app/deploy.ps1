# Build the standalone app and add the files Nuitka does not bundle.
#
# Run from local_app with the forages_rois_app environment active (see environment_app.yml):
#   .\deploy.ps1
#
# Output: dist\ForagesROIs.dist (ForagesROIs.exe plus everything it needs; zip the folder
# to distribute it). Added after the build:
#   models                 the ONNX models (the app looks for "models" next to the exe)
#   proj_dir\share\proj    PROJ data (proj.db), must match the PROJ version in the environment
#   gdal_data              GDAL data
#   CUDA, cuDNN and NVRTC DLLs, which onnxruntime and cuDNN load at run time, so Nuitka does
#   not see them. Without them the exe crashes during detection.
#   openblas.dll (numpy's BLAS, behind libblas.dll), copied to be sure it is there.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $env:CONDA_PREFIX -or -not (Test-Path "$env:CONDA_PREFIX\Library\share\proj\proj.db")) {
    throw "Activate the forages_rois_app environment first (conda activate forages_rois_app)."
}
if (Test-Path "ForagesROIs") {
    throw "Delete or move local_app\ForagesROIs (an earlier build); otherwise it is copied into the new build."
}
$lib = Join-Path $env:CONDA_PREFIX "Library"
$dist = Join-Path $PSScriptRoot "dist\ForagesROIs.dist"

# Resources (view.qml, icon, splash), compiled with this environment's rcc
pyside6-rcc resources.qrc -o rc_resources.py
if ($LASTEXITCODE -ne 0) { throw "pyside6-rcc failed." }

# pyside6-deploy adds every folder in local_app as data except a few names (dist, build,
# deployment, tests...), so the build goes to dist (exec_directory in pysidedeploy.spec)
pyside6-deploy -c pysidedeploy.spec -f
if ($LASTEXITCODE -ne 0 -or -not (Test-Path "$dist\ForagesROIs.exe")) { throw "pyside6-deploy failed." }

Write-Host "Copying models, PROJ and GDAL data..."
New-Item -ItemType Directory -Force "$dist\models", "$dist\proj_dir\share" | Out-Null
Copy-Item -Force "models\*.onnx" "$dist\models"
Remove-Item -Recurse -Force "$dist\proj_dir\share\proj", "$dist\gdal_data" -ErrorAction SilentlyContinue
Copy-Item -Recurse "$lib\share\proj" "$dist\proj_dir\share\proj"
Copy-Item -Recurse "$lib\share\gdal" "$dist\gdal_data"

Write-Host "Copying run time DLLs..."
$dlls = "cudart64_*.dll", "cublas*64_*.dll", "cufft64_*.dll", "cudnn*64_*.dll", "nvrtc*.dll", "openblas.dll"
foreach ($pattern in $dlls) {
    $files = Get-ChildItem "$lib\bin\$pattern"
    if (-not $files) { throw "Not found in the environment: $pattern" }
    $files | Copy-Item -Destination $dist -Force
}

Write-Host "Done: $dist\ForagesROIs.exe"
