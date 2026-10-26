ForagesROIs Launcher (QGIS plugin)

Light QGIS plugin for the ForagesROIs desktop application. Processing runs in the
application, installed separately, so the plugin needs nothing beyond QGIS.

Plugins > ForagesROIs (and toolbar):
- Open in ForagesROIs: starts ForagesROIs.exe with the selected raster layer prefilled
  as input. QGIS stays usable while the application processes.
- Load ForagesROIs results...: adds output shapefiles to the project. Each output has a
  .qml file with the same name, so QGIS shows the plot numbers and borders automatically.
- Set ForagesROIs application...: selects ForagesROIs.exe. Asked automatically the
  first time; saved in the QGIS settings.

Install: copy this folder (foragesrois_launcher) into the QGIS plugins folder, for example
%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins, or run "pb_tool deploy" here,
then enable "ForagesROIs Launcher" in Plugins > Manage and Install Plugins.

This plugin replaces the older processing plugin in plugins/foragesrois, which ran the
application from inside the plugin folder.
