import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import Qt5Compat.GraphicalEffects
import QtQuick.Effects
import CustomComponents 1.0

ApplicationWindow {
    id: mainWindow
    visible: true
    width: 1150
    height: 830
    title: qsTr(Qt.application.name + " - v" + Qt.application.version)
    minimumWidth: 630

    // Force standard desktop window decorations
    //flags: Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint

    property string inputRasterPath: ""
    property string outputFolderPath: ""
    property string inputCropShapefilePath: ""
    property string inputShapefilePath: ""
    property bool modelsFromMenu: false  // show the models dialog even when all is ok
    property bool taskRunning: false     // from Run Inference until finished (File > New is disabled meanwhile)
    // Bands of the input raster used as red, green, blue (lists shown for rasters with more than 3 bands)
    property var bandLabels: []
    property bool chooseBands: false

    function bandsText() {
        return chooseBands ? (redBandCombo.currentIndex + 1) + "," + (greenBandCombo.currentIndex + 1) + "," + (blueBandCombo.currentIndex + 1) : "";
    }
    // After a task, put its output where the next task reads it, to save time (the user can still
    // choose another file): crop fields -> Crop Shapefile; plants -> Input Shapefile (for
    // plot_numbering and plant_traits). plant_traits outputs are outlines, not inputs.
    function suggestNextInputs(task, outputFile) {
        if (!outputFile.toLowerCase().endsWith(".shp"))
            return;
        if (task === "crop_detection")
            inputCropShapefilePath = outputFile;
        else if (["tiling_detection", "tiling_detection_on_crop", "tiling_detection_only", "plot_numbering", "postprocessing"].indexOf(task) >= 0)
            inputShapefilePath = outputFile;
    }
    // Open a file dialog in the folder of the current path, else the preferred folder, else the last
    // folder used for this kind of file, else Documents (app_settings.start_folder)
    function openInFolder(dialog, kind, currentPath, preferredFolder) {
        dialog.currentFolder = processorInterface.startFolderUrl(kind, currentPath, preferredFolder || "");
        dialog.open();
    }
    // File > New: start again with an empty preview and empty inputs. The output folder, the task
    // and the options are kept; no file is deleted. Unsaved edits are handled by newSessionDialog.
    function newSession() {
        if (rasterItem.hasEdits) {
            newSessionDialog.open();
            return;
        }
        resetSession();
    }
    function resetSession() {
        eraseToggle.checked = false;
        addToggle.checked = false;
        rasterItem.clearAll();
        previewImage.source = "";
        inputRasterPath = "";
        inputShapefilePath = "";
        inputCropShapefilePath = "";
        bandLabels = [];
        chooseBands = false;
        previewContainer.fitPending = true;
        progressBar.value = 0;
        statusLabel.text = "Ready";
    }
    function refreshBands() {
        statusLabel.text = "Updating the preview with bands " + bandsText() + "...";
        processorInterface.previewRaster(inputRasterPath, outputFolderPath, bandsText());
    }

    // Check the models folder at startup (warns if models are missing)
    Component.onCompleted: processorInterface.loadModelsFolder()
    // Tasks that work on a detections shapefile instead of the raster
    property bool vectorTask: ["plot_numbering", "postprocessing", "plant_traits", "export_coco"].indexOf(taskCombo.currentText) >= 0
    // plant_traits and export_coco need both the shapefile and the raster; the other vector tasks use the raster only for the preview
    property bool rasterOptional: vectorTask && taskCombo.currentText !== "plant_traits" && taskCombo.currentText !== "export_coco"

    function urlToLocalPath(urlStr) {
        if (urlStr.startsWith("file:////")) {
            return "\\\\" + urlStr.substring(9).replace(/\//g, "\\");
        } else if (urlStr.startsWith("file:///")) {
            return urlStr.substring(8);
        } else if (urlStr.startsWith("file://")) {
            return "\\\\" + urlStr.substring(7).replace(/\//g, "\\");
        } else {
            return urlStr;
        }
    }

    menuBar: MenuBar {
        Menu {
            title: qsTr("&File")
            Action {
                text: qsTr("&New")
                shortcut: StandardKey.New
                enabled: !taskRunning
                onTriggered: newSession()
            }
            MenuSeparator {}
            Action {
                text: qsTr("Set &Models Folder...")
                onTriggered: {
                    modelsFolderDialog.currentFolder = "file:///" + processorInterface.getModelsFolder();
                    modelsFolderDialog.open();
                }
            }
            Action {
                text: qsTr("&Download Models")
                onTriggered: {
                    modelsFromMenu = true;
                    processorInterface.downloadModels();
                }
            }
            // Cache: default folder (used without an output folder) and the output folder's cache
            Menu {
                id: cacheMenu
                title: qsTr("&Cache")
                property string defaultFolder: ""
                property string outputFolder: ""
                property string defaultSize: ""
                property string outputSize: ""
                onAboutToShow: {
                    defaultFolder = processorInterface.cacheFolder("");
                    defaultSize = processorInterface.cacheSizeText(defaultFolder);
                    outputFolder = outputFolderPath !== "" ? processorInterface.cacheFolder(outputFolderPath) : "";
                    outputSize = outputFolder !== "" ? processorInterface.cacheSizeText(outputFolder) : "";
                }
                Action {
                    text: qsTr("Open Default Cache Folder")
                    onTriggered: processorInterface.openCacheFolder(cacheMenu.defaultFolder)
                }
                Action {
                    text: qsTr("Open Output Folder Cache")
                    enabled: cacheMenu.outputFolder !== ""
                    onTriggered: processorInterface.openCacheFolder(cacheMenu.outputFolder)
                }
                Action {
                    text: qsTr("Set Default Cache Folder...")
                    onTriggered: openInFolder(cacheFolderDialog, "cache", "", "")
                }
                MenuSeparator {}
                Action {
                    text: qsTr("Clear Default Cache (%1)").arg(cacheMenu.defaultSize)
                    onTriggered: {
                        clearCacheDialog.folder = cacheMenu.defaultFolder;
                        clearCacheDialog.text = qsTr("Delete the default cache (%1)?\n%2\n\nPreviews and RGB views are made again when needed.").arg(cacheMenu.defaultSize).arg(cacheMenu.defaultFolder);
                        clearCacheDialog.open();
                    }
                }
                Action {
                    text: cacheMenu.outputFolder !== "" ? qsTr("Clear Output Folder Cache (%1)").arg(cacheMenu.outputSize) : qsTr("Clear Output Folder Cache")
                    enabled: cacheMenu.outputFolder !== ""
                    onTriggered: {
                        clearCacheDialog.folder = cacheMenu.outputFolder;
                        clearCacheDialog.text = qsTr("Delete the cache of the output folder (%1)?\n%2\n\nTiles and detections are made again on the next run; the results in the output folder are kept.").arg(cacheMenu.outputSize).arg(cacheMenu.outputFolder);
                        clearCacheDialog.open();
                    }
                }
            }
            MenuSeparator {}
            Action {
                text: qsTr("&Quit")
                onTriggered: Qt.quit()
            }
        }
        Menu {
            title: qsTr("&Help")
            Action {
                text: qsTr("&About")
                onTriggered: aboutWindow.visible = true
            }
        }
    }

    Window {
        id: aboutWindow
        title: qsTr("About")
        width: 630
        height: 250
        modality: Qt.ApplicationModal
        flags: Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint

        ColumnLayout {
            anchors.fill: parent
            spacing: 1

            Rectangle {
                Layout.margins: 20
                Layout.preferredHeight: 150
                Layout.fillWidth: true
                color: "transparent"

                RowLayout {
                    spacing: 20
                    anchors.fill: parent

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        color: "transparent"

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 1

                            Text {
                                text: '<div style="text-align: left;"><h1>' + Qt.application.name + ' v' + Qt.application.version + '</h1><p>Automated forage ROIs detection from drone imagery.</p><h3>Authors</h3><p>' + Qt.application.organization + '.</p><h3>Acknowledgments</h3><p>This work was partially funded by Accelerated Breeding Initiative of CGIAR.</p><p></p></div>'
                                textFormat: Text.RichText
                            }
                        }
                    }

                    Rectangle {
                        Layout.preferredWidth: 200
                        Layout.fillHeight: true
                        color: "transparent"

                        Rectangle {
                            id: logoRectangle2
                            color: "white"
                            radius: 10
                            anchors.fill: parent
                            anchors.margins: 10

                            RowLayout {
                                anchors.fill: parent
                                Image {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    source: "res/tf.png"
                                    fillMode: Image.PreserveAspectFit
                                    mipmap: true
                                }
                            }
                        }
                        MultiEffect {
                            source: logoRectangle2
                            anchors.fill: logoRectangle2
                            shadowBlur: 1.0
                            shadowEnabled: true
                            shadowColor: "gray"
                            shadowVerticalOffset: 0
                            shadowHorizontalOffset: 0
                        }
                    }
                }
            }
        }
    }

    FileDialog {
        id: fileDialog
        title: qsTr("Please choose an input raster file")
        nameFilters: ["Raster files (*.tif *.tiff)", "All files (*)"]
        onAccepted: {
            inputRasterPath = urlToLocalPath(fileDialog.currentFile.toString());
            processorInterface.rememberFolder("raster", inputRasterPath);
            previewContainer.fitPending = true;
            loadingIndicator.running = true;
            rasterLoadTimer.start();
        }
    }

    FolderDialog {
        id: folderDialog
        title: qsTr("Please choose an output folder")
        onAccepted: {
            outputFolderPath = urlToLocalPath(folderDialog.currentFolder.toString());
            processorInterface.rememberFolder("output", outputFolderPath);
        }
    }

    FileDialog {
        id: cropFileDialog
        title: qsTr("Please choose a crop shapefile")
        nameFilters: ["Shapefiles (*.shp)", "All files (*)"]
        onAccepted: {
            inputCropShapefilePath = urlToLocalPath(cropFileDialog.currentFile.toString());
            processorInterface.rememberFolder("shapefile", inputCropShapefilePath);
        }
    }

    FileDialog {
        id: shapefileDialog
        title: qsTr("Please choose a detections shapefile")
        nameFilters: ["Shapefiles (*.shp)", "All files (*)"]
        onAccepted: {
            inputShapefilePath = urlToLocalPath(shapefileDialog.currentFile.toString());
            processorInterface.rememberFolder("shapefile", inputShapefilePath);
        }
    }

    FolderDialog {
        id: modelsFolderDialog
        title: qsTr("Please choose the folder that contains the models (.onnx)")
        onAccepted: {
            modelsFromMenu = true;
            processorInterface.setModelsFolder(urlToLocalPath(modelsFolderDialog.selectedFolder.toString()));
        }
    }

    FolderDialog {
        id: cacheFolderDialog
        title: qsTr("Choose a folder for the default cache (a .cache_forages_rois folder is made inside)")
        onAccepted: {
            processorInterface.setDefaultCacheFolder(urlToLocalPath(cacheFolderDialog.selectedFolder.toString()));
            processorInterface.rememberFolder("cache", urlToLocalPath(cacheFolderDialog.selectedFolder.toString()));
        }
    }

    MessageDialog {
        id: clearCacheDialog
        title: qsTr("Clear Cache")
        property string folder: ""
        buttons: MessageDialog.Yes | MessageDialog.No
        onButtonClicked: function (button, role) {
            if (button === MessageDialog.Yes) {
                processorInterface.clearCache(folder);
            }
        }
    }

    // File > New with erased or added boxes not saved yet
    MessageDialog {
        id: newSessionDialog
        title: qsTr("New")
        text: qsTr("The edited boxes are not saved.")
        informativeText: qsTr("Save the changes into the shapefile before clearing?")
        buttons: MessageDialog.Save | MessageDialog.Discard | MessageDialog.Cancel
        onButtonClicked: function (button, role) {
            if (button === MessageDialog.Save) {
                rasterItem.saveEdits();
                resetSession();
            } else if (button === MessageDialog.Discard) {
                resetSession();
            }
        }
    }

    MessageDialog {
        id: modelsDialog
        title: qsTr("Models")
        buttons: MessageDialog.Ok
    }

    // At startup, when models are missing: offer to download them from Hugging Face
    MessageDialog {
        id: downloadModelsDialog
        title: qsTr("Models Missing")
        buttons: MessageDialog.Yes | MessageDialog.No
        onButtonClicked: function (button, role) {
            if (button === MessageDialog.Yes) {
                modelsFromMenu = true;
                processorInterface.downloadModels();
            }
        }
    }

    MessageDialog {
        id: infoDialog
        title: qsTr("Process Completed")
        text: qsTr("The inference process has finished successfully.")
        buttons: MessageDialog.Ok | MessageDialog.Open
        onButtonClicked: function (button, role) {
            switch (button) {
            case MessageDialog.Open:
                processorInterface.openOutputFolder(outputFolderPath);
                break;
            }
        }
    }

    MessageDialog {
        id: errorDialog
        title: qsTr("Process Error")
        text: qsTr("An error occurred during processing.")
        buttons: MessageDialog.Ok
    }

    Connections {
        target: processorInterface
        function onProgressUpdated(info) {
            if (info.progress !== undefined) {
                progressBar.value = info.progress;
            }
            if (info.status !== undefined) {
                statusLabel.text = info.status;
            }
        }
        function onFinished(info) {
            taskRunning = false;
            progressBar.value = 100;
            if (info.status === "error") {
                statusLabel.text = "Error";
                errorDialog.text = info.message || "An unknown error occurred.";
                errorDialog.open();
            } else {
                // Show what actually happened (output path, or skipped because it already exists)
                statusLabel.text = "Finished";
                infoDialog.text = info.message || "The process has finished.";
                infoDialog.open();
                suggestNextInputs(info.task || "", info.output_file || "");
            }
        }
        function onVisualizationReady(imagePath) {
            previewImage.source = "file:///" + imagePath + "?" + new Date().getTime();
            statusLabel.text = "Visualization Ready!";
            progressBar.value = 100;
        }
        function onModelsStatus(level, folder, message) {
            statusLabel.text = level === "ok" ? " Ready. Models: " + folder : " Models " + level + ": " + folder;
            // At startup only problems are shown, with the offer to download the missing models;
            // from the menu the result is always shown
            if (level !== "ok" && !modelsFromMenu) {
                downloadModelsDialog.text = message + "\n\nDownload the missing models now?";
                downloadModelsDialog.open();
            } else if (level !== "ok" || modelsFromMenu) {
                modelsDialog.title = level === "ok" ? qsTr("Models") : (level === "warning" ? qsTr("Models Warning") : qsTr("Models Error"));
                modelsDialog.text = message;
                modelsDialog.open();
            }
            modelsFromMenu = false;
        }
        function onRasterBandsInfo(labels, used, choose) {
            bandLabels = labels;
            chooseBands = choose;
            if (choose) {
                redBandCombo.currentIndex = used[0] - 1;
                greenBandCombo.currentIndex = used[1] - 1;
                blueBandCombo.currentIndex = used[2] - 1;
            }
        }
        function onRasterPreviewReady(imagePath, rasterPath) {
            previewImage.source = "file:///" + imagePath + "?" + new Date().getTime();
            // The raster shown (RGB view of the chosen bands, or the raster itself)
            rasterItem.sourcePath = rasterPath || inputRasterPath;
            if (previewContainer.fitPending) {
                previewContainer.fitToImage();
                previewContainer.fitPending = false;
            }
            rasterItem.shapefilePath = "";
            rasterItem.outlinePath = "";
            statusLabel.text = "Raster preview generated. Ready for inference.";
        }
        function onVisualizationPathsReady(rasterPath, shpPath) {
            rasterItem.sourcePath = rasterPath;
            if (shpPath !== rasterItem.shapefilePath)
                rasterItem.outlinePath = "";  // outlines of other detections no longer match
            rasterItem.shapefilePath = shpPath;
            console.log("Vector paths set:", rasterPath, shpPath);
        }
        function onOutlinesReady(path) {
            rasterItem.outlinePath = path;  // plant_traits: outlines shown with the boxes
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        // Main Layout
        Item {
            Layout.fillHeight: true
            Layout.fillWidth: true

            SplitView {
                anchors.fill: parent
                orientation: Qt.Horizontal

                // PANE 1: Sidebar
                RowLayout {
                    SplitView.preferredWidth: parent.width * 0.25
                    SplitView.minimumWidth: 50
                    spacing: 0

                    // Sidebar (Left Panel)
                    Rectangle {
                        id: sidebarContainer
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true  // nothing is drawn outside the panel when it is narrow
                        color: "transparent"
                        border.color: "lightgray"
                        border.width: 1

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 0

                            TabBar {
                                id: leftTabBar
                                Layout.fillWidth: true
                                TabButton {
                                    text: qsTr("Settings")
                                }
                            }

                            StackLayout {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                currentIndex: leftTabBar.currentIndex

                                Item {
                                    ColumnLayout {
                                        anchors.fill: parent
                                        spacing: 15
                                        anchors.margins: 10

                                        // Input Raster Selection
                                        ColumnLayout {
                                            Layout.fillWidth: true
                                            spacing: 5
                                            Label {
                                                text: rasterOptional ? "Input Raster (Optional, for preview):" : "Input Raster:"
                                                font.bold: true
                                                Layout.fillWidth: true
                                                elide: Text.ElideRight
                                            }
                                            RowLayout {
                                                Layout.fillWidth: true
                                                TextField {
                                                    text: inputRasterPath
                                                    placeholderText: "Select input..."
                                                    Layout.fillWidth: true
                                                    readOnly: true
                                                }
                                                Button {
                                                    text: "..."
                                                    ToolTip.visible: hovered
                                                    ToolTip.text: "Browse for the raster file..."
                                                    onClicked: openInFolder(fileDialog, "raster", inputRasterPath, "")
                                                }
                                                Button {
                                                    text: "^"
                                                    Layout.preferredWidth: 24
                                                    enabled: inputRasterPath !== ""
                                                    ToolTip.visible: hovered
                                                    ToolTip.text: "Open containing folder"
                                                    onClicked: processorInterface.openOutputFolder(inputRasterPath)
                                                }
                                            }
                                            // Bands used as red, green, blue (rasters with more than 3 bands)
                                            RowLayout {
                                                Layout.fillWidth: true
                                                visible: chooseBands && inputRasterPath !== ""
                                                Label {
                                                    text: "Bands R, G, B:"
                                                }
                                                ComboBox {
                                                    id: redBandCombo
                                                    Layout.fillWidth: true
                                                    model: bandLabels
                                                    onActivated: refreshBands()
                                                    ToolTip.visible: hovered
                                                    ToolTip.text: "Band shown and processed as red"
                                                }
                                                ComboBox {
                                                    id: greenBandCombo
                                                    Layout.fillWidth: true
                                                    model: bandLabels
                                                    onActivated: refreshBands()
                                                    ToolTip.visible: hovered
                                                    ToolTip.text: "Band shown and processed as green"
                                                }
                                                ComboBox {
                                                    id: blueBandCombo
                                                    Layout.fillWidth: true
                                                    model: bandLabels
                                                    onActivated: refreshBands()
                                                    ToolTip.visible: hovered
                                                    ToolTip.text: "Band shown and processed as blue"
                                                }
                                            }
                                        }

                                        // Detections Shapefile Selection (numbering and postprocessing tasks)
                                        ColumnLayout {
                                            Layout.fillWidth: true
                                            spacing: 5
                                            visible: vectorTask
                                            Label {
                                                text: "Input Shapefile:"
                                                font.bold: true
                                            }
                                            RowLayout {
                                                Layout.fillWidth: true
                                                TextField {
                                                    text: inputShapefilePath
                                                    placeholderText: "Select detections shapefile..."
                                                    Layout.fillWidth: true
                                                    readOnly: true
                                                }
                                                Button {
                                                    text: "..."
                                                    ToolTip.visible: hovered
                                                    ToolTip.text: "Browse for the shapefile..."
                                                    onClicked: openInFolder(shapefileDialog, "shapefile", inputShapefilePath, outputFolderPath)
                                                }
                                                Button {
                                                    text: "^"
                                                    Layout.preferredWidth: 24
                                                    enabled: inputShapefilePath !== ""
                                                    ToolTip.visible: hovered
                                                    ToolTip.text: "Open containing folder"
                                                    onClicked: processorInterface.openOutputFolder(inputShapefilePath)
                                                }
                                            }
                                        }

                                        // Crop Shapefile Selection (Optional)
                                        ColumnLayout {
                                            Layout.fillWidth: true
                                            spacing: 5
                                            visible: taskCombo.currentText === "tiling_detection_on_crop" || taskCombo.currentText === "export_coco"
                                            Label {
                                                text: "Crop Shapefile (Optional):"
                                                font.bold: true
                                                Layout.fillWidth: true
                                                elide: Text.ElideRight
                                            }
                                            RowLayout {
                                                Layout.fillWidth: true
                                                TextField {
                                                    text: inputCropShapefilePath
                                                    placeholderText: "Select crop shapefile or leave blank to auto-detect..."
                                                    Layout.fillWidth: true
                                                    readOnly: true
                                                }
                                                Button {
                                                    text: "..."
                                                    ToolTip.visible: hovered
                                                    ToolTip.text: "Browse for the crop shapefile..."
                                                    onClicked: openInFolder(cropFileDialog, "shapefile", inputCropShapefilePath, outputFolderPath)
                                                }
                                                Button {
                                                    text: "^"
                                                    Layout.preferredWidth: 24
                                                    enabled: inputCropShapefilePath !== ""
                                                    ToolTip.visible: hovered
                                                    ToolTip.text: "Open containing folder"
                                                    onClicked: processorInterface.openOutputFolder(inputCropShapefilePath)
                                                }
                                                Button {
                                                    text: "X"
                                                    onClicked: inputCropShapefilePath = ""
                                                }
                                            }
                                        }

                                        // Output Folder Selection
                                        ColumnLayout {
                                            Layout.fillWidth: true
                                            spacing: 5
                                            Label {
                                                text: "Output Folder:"
                                                font.bold: true
                                            }
                                            RowLayout {
                                                Layout.fillWidth: true
                                                TextField {
                                                    text: outputFolderPath
                                                    placeholderText: "Select output..."
                                                    Layout.fillWidth: true
                                                    readOnly: true
                                                }
                                                Button {
                                                    text: "..."
                                                    ToolTip.visible: hovered
                                                    ToolTip.text: "Browse for the output folder..."
                                                    onClicked: openInFolder(folderDialog, "output", outputFolderPath, "")
                                                }
                                                Button {
                                                    text: "^"
                                                    Layout.preferredWidth: 24
                                                    enabled: outputFolderPath !== ""
                                                    ToolTip.visible: hovered
                                                    ToolTip.text: "Open output folder"
                                                    onClicked: processorInterface.openOutputFolder(outputFolderPath)
                                                }
                                            }
                                        }

                                        // Task Selection
                                        ColumnLayout {
                                            Layout.fillWidth: true
                                            spacing: 5
                                            Label {
                                                text: "Task:"
                                                font.bold: true
                                            }
                                            ComboBox {
                                                id: taskCombo
                                                Layout.fillWidth: true
                                                model: ["tiling_detection", "crop_detection", "tiling_detection_on_crop", "tiling_detection_only", "plot_numbering", "postprocessing", "plant_traits", "export_coco"]
                                            }
                                        }

                                        // Advanced Options: task specific settings, hidden until opened (only tasks with options)
                                        GroupBox {
                                            id: advancedGroup
                                            Layout.fillWidth: true
                                            clip: true
                                            visible: ["tiling_detection", "tiling_detection_on_crop", "tiling_detection_only", "plot_numbering", "plant_traits"].indexOf(taskCombo.currentText) >= 0
                                            // Collapsible group: Qt has no built in one, so the title is a flat ToolButton with an
                                            // arrow (closed or open), the usual Qt pattern. Same title bar as the "Preview" group.
                                            property bool expanded: false
                                            topPadding: 28 + (expanded ? 8 : 0)  // title bar 28 px; collapsed: only the title bar
                                            background: Rectangle {
                                                width: parent.width
                                                height: parent.height
                                                color: "transparent"
                                                border.color: advancedGroup.palette.mid
                                                radius: 3
                                                Rectangle {  // title bar
                                                    x: 1
                                                    y: 1
                                                    width: parent.width - 2
                                                    height: 26
                                                    radius: 2
                                                    color: Qt.darker(advancedGroup.palette.window, 1.08)
                                                }
                                            }
                                            label: ToolButton {
                                                id: advancedToggle
                                                x: advancedGroup.leftPadding
                                                y: (28 - height) / 2  // centred in the title bar
                                                leftPadding: 2
                                                text: (advancedGroup.expanded ? "▾  " : "▸  ") + "Advanced"
                                                font: advancedGroup.font
                                                onClicked: advancedGroup.expanded = !advancedGroup.expanded
                                                ToolTip.visible: hovered
                                                ToolTip.text: advancedGroup.expanded ? "Hide advanced options" : "Show advanced options"
                                            }
                                            contentHeight: advancedGroup.expanded ? advancedOptions.implicitHeight : 0  // no empty space when collapsed
                                            bottomPadding: advancedGroup.expanded ? padding : 0  // collapsed: the box holds only the title
                                            ColumnLayout {
                                                id: advancedOptions
                                                width: parent.width
                                                spacing: 10
                                                visible: advancedGroup.expanded

                                                // Detection Options (plant detection tasks)
                                                ColumnLayout {
                                                    Layout.fillWidth: true
                                                    spacing: 5
                                                    visible: taskCombo.currentText === "tiling_detection" || taskCombo.currentText === "tiling_detection_on_crop" || taskCombo.currentText === "tiling_detection_only"
                                                    Label {
                                                        text: "Detection:"
                                                        font.bold: true
                                                    }
                                                    RowLayout {
                                                        Layout.fillWidth: true
                                                        Label {
                                                            text: "Confidence:"
                                                        }
                                                        TextField {
                                                            id: confidenceField
                                                            Layout.fillWidth: true
                                                            placeholderText: "0.10 (default)"
                                                            validator: DoubleValidator {
                                                                bottom: 0.01
                                                                top: 0.99
                                                            }
                                                        }
                                                    }
                                                    CheckBox {
                                                        id: resampleCheckbox
                                                        Layout.fillWidth: true  // long text is shortened with ... instead of leaving the group
                                                        text: "Resample fine rasters to the model resolution"
                                                        checked: true
                                                        ToolTip.visible: hovered
                                                        ToolTip.text: "Rasters finer than 1.1 cm per pixel are resampled to 1.5 cm, the resolution the plant model was trained on"
                                                    }
                                                }

                                                // Plant Traits Options (plant_traits task)
                                                ColumnLayout {
                                                    Layout.fillWidth: true
                                                    spacing: 5
                                                    visible: taskCombo.currentText === "plant_traits"
                                                    Label {
                                                        text: "Plant traits:"
                                                        font.bold: true
                                                    }
                                                    RowLayout {
                                                        Layout.fillWidth: true
                                                        Label {
                                                            text: "Index:"
                                                        }
                                                        ComboBox {
                                                            id: traitsIndexCombo
                                                            Layout.fillWidth: true
                                                            // Auto: NDVI when the raster has a near infrared band, otherwise RGBVI
                                                            model: ["Auto", "RGBVI", "ExG", "NDVI", "NDRE", "GNDVI"]
                                                        }
                                                    }
                                                    RowLayout {
                                                        Layout.fillWidth: true
                                                        Label {
                                                            text: "Threshold:"
                                                        }
                                                        TextField {
                                                            id: traitsThresholdField
                                                            Layout.fillWidth: true
                                                            placeholderText: "automatic (Otsu)"
                                                            validator: DoubleValidator {
                                                                bottom: -1.0
                                                                top: 1.0
                                                            }
                                                        }
                                                    }
                                                }

                                                // Numbering Options (plot_numbering task; detection tasks only the empty plots option)
                                                ColumnLayout {
                                                    Layout.fillWidth: true
                                                    spacing: 5
                                                    visible: ["tiling_detection", "tiling_detection_on_crop", "plot_numbering"].indexOf(taskCombo.currentText) >= 0
                                                    Label {
                                                        text: "Numbering:"
                                                        font.bold: true
                                                    }
                                                    CheckBox {
                                                        id: serpentineCheckbox
                                                        Layout.fillWidth: true  // long text is shortened with ... instead of leaving the group
                                                        text: "Serpentine"
                                                        checked: true
                                                        visible: taskCombo.currentText === "plot_numbering"
                                                    }
                                                    RowLayout {
                                                        Layout.fillWidth: true
                                                        visible: taskCombo.currentText === "plot_numbering"
                                                        Label {
                                                            text: "Start corner:"
                                                        }
                                                        ComboBox {
                                                            id: startCornerCombo
                                                            Layout.fillWidth: true
                                                            model: ["NW", "NE", "SW", "SE"]
                                                        }
                                                    }
                                                    CheckBox {
                                                        id: swapAxesCheckbox
                                                        Layout.fillWidth: true  // long text is shortened with ... instead of leaving the group
                                                        text: "Swap rows and columns"
                                                        checked: false
                                                        visible: taskCombo.currentText === "plot_numbering"
                                                    }
                                                    CheckBox {
                                                        id: emptyPlotsCheckbox
                                                        Layout.fillWidth: true  // long text is shortened with ... instead of leaving the group
                                                        text: "Empty plots for missing plants"
                                                        checked: false
                                                        ToolTip.visible: hovered
                                                        ToolTip.text: "Number planting positions: a missing plant gets an empty plot and keeps its number"
                                                    }
                                                }
                                            }
                                        }

                                        // Preview Options: how results are drawn in the preview
                                        GroupBox {
                                            id: previewGroup
                                            title: "Preview"
                                            Layout.fillWidth: true
                                            clip: true
                                            topPadding: 28 + 8  // title bar 28 px
                                            background: Rectangle {
                                                width: parent.width
                                                height: parent.height
                                                color: "transparent"
                                                border.color: previewGroup.palette.mid
                                                radius: 3
                                                Rectangle {  // title bar
                                                    x: 1
                                                    y: 1
                                                    width: parent.width - 2
                                                    height: 26
                                                    radius: 2
                                                    color: Qt.darker(previewGroup.palette.window, 1.08)
                                                }
                                            }
                                            label: Label {
                                                x: previewGroup.leftPadding
                                                y: (28 - height) / 2  // centred in the title bar
                                                text: previewGroup.title
                                            }
                                            ColumnLayout {
                                                width: parent.width
                                                spacing: 5

                                                CheckBox {
                                                    id: useVectorOverlayCheckbox
                                                    Layout.fillWidth: true  // long text is shortened with ... instead of leaving the group
                                                    text: "Use Native Vector Overlay (High Res)"
                                                    checked: true
                                                }

                                                CheckBox {
                                                    id: showRoisCheckbox
                                                    Layout.fillWidth: true  // long text is shortened with ... instead of leaving the group
                                                    text: "Show detections (boxes)"
                                                    checked: true
                                                    enabled: useVectorOverlayCheckbox.checked
                                                }

                                                CheckBox {
                                                    id: showOutlinesCheckbox
                                                    Layout.fillWidth: true  // long text is shortened with ... instead of leaving the group
                                                    text: "Show plant outlines"
                                                    checked: true
                                                    enabled: useVectorOverlayCheckbox.checked && rasterItem.outlinePath !== ""
                                                    ToolTip.visible: hovered
                                                    ToolTip.text: "Canopy outlines from plant_traits (green), drawn under the boxes"
                                                }

                                                CheckBox {
                                                    id: colorByDetectionCheckbox
                                                    Layout.fillWidth: true  // long text is shortened with ... instead of leaving the group
                                                    text: "Color each detection"
                                                    checked: false
                                                    enabled: useVectorOverlayCheckbox.checked && showRoisCheckbox.checked
                                                }
                                            }
                                        }

                                        // Edit Group: erase detections in the shown detections layer (edit mode, as in GIS editors)
                                        GroupBox {
                                            id: editGroup
                                            title: "Edit"
                                            Layout.fillWidth: true
                                            clip: true
                                            enabled: useVectorOverlayCheckbox.checked && showRoisCheckbox.checked && rasterItem.shapefilePath !== ""
                                            topPadding: 28 + 8  // title bar 28 px, same as the Preview group
                                            background: Rectangle {
                                                width: parent.width
                                                height: parent.height
                                                color: "transparent"
                                                border.color: editGroup.palette.mid
                                                radius: 3
                                                Rectangle {  // title bar
                                                    x: 1
                                                    y: 1
                                                    width: parent.width - 2
                                                    height: 26
                                                    radius: 2
                                                    color: Qt.darker(editGroup.palette.window, 1.08)
                                                }
                                            }
                                            label: Label {
                                                x: editGroup.leftPadding
                                                y: (28 - height) / 2  // centred in the title bar
                                                text: editGroup.title
                                            }
                                            ColumnLayout {
                                                width: parent.width
                                                spacing: 5

                                                // Edit tools (one at a time, or none): erase with right button strokes,
                                                // or add a box with a right button drag. The left button always pans.
                                                Button {
                                                    id: eraseToggle
                                                    Layout.fillWidth: true
                                                    checkable: true
                                                    text: checked ? "Erasing boxes (click to stop)" : "Erase boxes"
                                                    ToolTip.visible: hovered
                                                    ToolTip.text: "While on, hold the right mouse button and trace over boxes: every box the stroke touches is erased. The left button still pans"
                                                    onEnabledChanged: if (!enabled) checked = false
                                                    onCheckedChanged: if (checked) addToggle.checked = false
                                                }
                                                Button {
                                                    id: addToggle
                                                    Layout.fillWidth: true
                                                    checkable: true
                                                    text: checked ? "Adding boxes (click to stop)" : "Add boxes"
                                                    ToolTip.visible: hovered
                                                    ToolTip.text: "While on, drag with the right mouse button from one corner of the plant to the other to add a box (Esc cancels). The left button still pans"
                                                    onEnabledChanged: if (!enabled) checked = false
                                                    onCheckedChanged: if (checked) eraseToggle.checked = false
                                                }
                                                RowLayout {
                                                    Layout.fillWidth: true
                                                    visible: addToggle.checked
                                                    Label {
                                                        text: "Species:"
                                                    }
                                                    ComboBox {
                                                        id: addClassCombo
                                                        Layout.fillWidth: true
                                                        model: ["urochloa", "megathyrsus"]  // index = class number of the plant model
                                                    }
                                                }
                                                Label {
                                                    Layout.fillWidth: true
                                                    visible: eraseToggle.checked || addToggle.checked || rasterItem.hasEdits
                                                    elide: Text.ElideRight
                                                    text: rasterItem.hasEdits
                                                          ? rasterItem.erasedCount + " erased, " + rasterItem.addedCount + " added, not saved yet"
                                                          : (addToggle.checked ? "Drag with the right mouse button" : "Trace with the right mouse button")
                                                }
                                                RowLayout {
                                                    Layout.fillWidth: true
                                                    Button {
                                                        text: "Undo"
                                                        Layout.fillWidth: true
                                                        enabled: rasterItem.canUndo
                                                        onClicked: rasterItem.undoErase()
                                                        ToolTip.visible: hovered
                                                        ToolTip.text: "Undo the last stroke or added box (Ctrl+Z)"
                                                    }
                                                    Button {
                                                        text: "Save"
                                                        Layout.fillWidth: true
                                                        enabled: rasterItem.hasEdits
                                                        ToolTip.visible: hovered
                                                        ToolTip.text: "Write the erased and added boxes into the shapefile (cannot be undone)"
                                                        onClicked: {
                                                            var saved = rasterItem.saveEdits();
                                                            if (saved !== "") {
                                                                inputShapefilePath = saved;
                                                                statusLabel.text = " Saved: " + saved;
                                                            }
                                                        }
                                                    }
                                                    Button {
                                                        text: "Discard"
                                                        Layout.fillWidth: true
                                                        enabled: rasterItem.hasEdits
                                                        onClicked: rasterItem.discardEdits()
                                                        ToolTip.visible: hovered
                                                        ToolTip.text: "Forget all erased and added boxes since the last save"
                                                    }
                                                }
                                                Shortcut {
                                                    sequence: StandardKey.Undo
                                                    enabled: editGroup.enabled
                                                    onActivated: rasterItem.undoErase()
                                                }
                                                Shortcut {
                                                    sequence: "Esc"
                                                    enabled: addToggle.checked
                                                    onActivated: rasterItem.cancelBox()
                                                }
                                            }
                                        }

                                        // Process Button
                                        ColumnLayout {
                                            Layout.fillWidth: true
                                            spacing: 5

                                            Button {
                                                text: "Run Inference"
                                                Layout.fillWidth: true
                                                font.bold: true
                                                enabled: true//inputRasterPath !== "" && outputFolderPath !== ""
                                                onClicked: {
                                                    // plot_numbering rewrites the shown shapefile: save pending edits into it first
                                                    if (taskCombo.currentText === "plot_numbering" && rasterItem.hasEdits)
                                                        rasterItem.saveEdits();
                                                    taskRunning = true;
                                                    progressBar.value = 0;
                                                    statusLabel.text = "Processing started...";
                                                    var params = {
                                                        "task": taskCombo.currentText,
                                                        "input_file": inputRasterPath,
                                                        "output_folder": outputFolderPath,
                                                        "use_vector_overlay": useVectorOverlayCheckbox.checked,
                                                        "crop_shapefile": inputCropShapefilePath,
                                                        "input_shapefile": inputShapefilePath,
                                                        "serpentine": serpentineCheckbox.checked,
                                                        "start_corner": startCornerCombo.currentText,
                                                        "swap_axes": swapAxesCheckbox.checked,
                                                        "empty_plots": emptyPlotsCheckbox.checked,
                                                        "index": traitsIndexCombo.currentText,
                                                        "threshold": traitsThresholdField.text,
                                                        "confidence": confidenceField.text,
                                                        "bands": bandsText(),
                                                        "resample": resampleCheckbox.checked
                                                    };
                                                    processorInterface.process(params);
                                                }
                                            }
                                            Button {
                                                id: cancelButton
                                                text: "Cancel"
                                                Layout.fillWidth: true
                                                onClicked: processorInterface.cancelProcessing()
                                            }
                                        }

                                        Item {
                                            Layout.fillHeight: true
                                        } // Spacer
                                    }
                                }
                            }
                        }
                    }
                }

                // PANE 2: Main Workspace (Editor Area + Bottom Panel)
                SplitView {
                    SplitView.fillWidth: true
                    orientation: Qt.Vertical

                    // Top Area (Editor Area)
                    Rectangle {
                        SplitView.fillHeight: true
                        SplitView.minimumHeight: 200
                        color: "transparent"
                        border.color: "lightgray"
                        border.width: 1

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 0

                            TabBar {
                                id: rightTabBar
                                Layout.fillWidth: false
                                TabButton {
                                    text: qsTr("Preview")
                                }
                            }

                            StackLayout {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                currentIndex: rightTabBar.currentIndex

                                Rectangle {
                                    id: previewContainer
                                    color: "#e0e0e0"
                                    clip: true

                                    property real currentZoom: 1.0
                                    property bool fitPending: true  // fit the next raster that opens (not on band changes)

                                    // Whole raster visible: zoom to fit the preview and go to the top left
                                    function fitToImage() {
                                        currentZoom = rasterItem.rasterWidth > 0 ? Math.min(width / rasterItem.rasterWidth, height / rasterItem.rasterHeight) : 1.0;  // image preview: zoom 1 already fits (PreserveAspectFit)
                                        flickable.contentX = 0;
                                        flickable.contentY = 0;
                                    }

                                    BusyIndicator {
                                        id: loadingIndicator
                                        anchors.centerIn: parent
                                        running: false
                                        visible: running
                                        z: 10
                                    }

                                    Timer {
                                        id: rasterLoadTimer
                                        interval: 50
                                        repeat: false
                                        onTriggered: {
                                            processorInterface.previewRaster(inputRasterPath, outputFolderPath, "");  // new raster: guess the bands
                                            loadingIndicator.running = false;
                                        }
                                    }

                                    RasterPreviewItem {
                                        id: rasterItem
                                        anchors.fill: parent
                                        visible: useVectorOverlayCheckbox.checked
                                        contentX: flickable.contentX
                                        contentY: flickable.contentY
                                        zoomScale: previewContainer.currentZoom
                                        showVectorOverlay: showRoisCheckbox.checked
                                        showOutlines: showOutlinesCheckbox.checked
                                        colorByDetection: colorByDetectionCheckbox.checked
                                    }

                                    Flickable {
                                        id: flickable
                                        anchors.fill: parent
                                        contentWidth: useVectorOverlayCheckbox.checked ? (rasterItem.rasterWidth * previewContainer.currentZoom) : (previewImage.width * previewContainer.currentZoom)
                                        contentHeight: useVectorOverlayCheckbox.checked ? (rasterItem.rasterHeight * previewContainer.currentZoom) : (previewImage.height * previewContainer.currentZoom)
                                        clip: true
                                        boundsBehavior: Flickable.StopAtBounds
                                        maximumFlickVelocity: 0
                                        flickDeceleration: 100000

                                        Image {
                                            id: previewImage
                                            visible: !useVectorOverlayCheckbox.checked
                                            width: previewContainer.width
                                            height: previewContainer.height
                                            scale: previewContainer.currentZoom
                                            fillMode: Image.PreserveAspectFit
                                            cache: false
                                            transformOrigin: Item.TopLeft
                                        }

                                        Item {
                                            visible: useVectorOverlayCheckbox.checked
                                            width: rasterItem.rasterWidth * previewContainer.currentZoom
                                            height: rasterItem.rasterHeight * previewContainer.currentZoom
                                        }
                                    }

                                    MouseArea {
                                        anchors.fill: parent
                                        acceptedButtons: Qt.NoButton
                                        onWheel: wheel => {
                                            var zoomFactor = wheel.angleDelta.y > 0 ? 1.1 : 0.9;
                                            var newScale = previewContainer.currentZoom * zoomFactor;
                                            if (newScale >= 0.001 && newScale <= 100.0) {
                                                var oldScale = previewContainer.currentZoom;
                                                var mouseXRel = (wheel.x + flickable.contentX) / oldScale;
                                                var mouseYRel = (wheel.y + flickable.contentY) / oldScale;

                                                previewContainer.currentZoom = newScale;

                                                flickable.contentX = (mouseXRel * newScale) - wheel.x;
                                                flickable.contentY = (mouseYRel * newScale) - wheel.y;
                                            }
                                        }
                                    }

                                    // Erase tool: right button strokes erase the boxes they touch (left button still pans)
                                    MouseArea {
                                        anchors.fill: parent
                                        enabled: eraseToggle.checked && editGroup.enabled
                                        acceptedButtons: Qt.RightButton
                                        onPressed: mouse => rasterItem.beginErase(mouse.x, mouse.y)
                                        onPositionChanged: mouse => rasterItem.eraseTo(mouse.x, mouse.y)
                                        onReleased: rasterItem.endErase()
                                        onCanceled: rasterItem.endErase()
                                    }

                                    // Add tool: a right button drag draws a new box with the chosen species
                                    MouseArea {
                                        anchors.fill: parent
                                        enabled: addToggle.checked && editGroup.enabled
                                        acceptedButtons: Qt.RightButton
                                        onPressed: mouse => rasterItem.beginBox(mouse.x, mouse.y)
                                        onPositionChanged: mouse => rasterItem.dragBoxTo(mouse.x, mouse.y)
                                        onReleased: rasterItem.endBox(addClassCombo.currentIndex)
                                        onCanceled: rasterItem.cancelBox()
                                    }

                                    Text {
                                        text: "Visual Preview\n(Drag & Drop Raster Image Here)"
                                        visible: previewImage.source == "" && rasterItem.sourcePath == ""
                                        anchors.centerIn: parent
                                        color: "gray"
                                        font.pixelSize: 18
                                        horizontalAlignment: Text.AlignHCenter
                                    }

                                    DropArea {
                                        anchors.fill: parent
                                        onEntered: drag => {
                                            drag.accept(Qt.LinkAction);
                                        }
                                        onDropped: drop => {
                                            if (drop.hasUrls) {
                                                inputRasterPath = urlToLocalPath(drop.urls[0].toString());
                                                processorInterface.rememberFolder("raster", inputRasterPath);
                                                previewContainer.fitPending = true;  // fit once the raster is loaded
                                                loadingIndicator.running = true;
                                                rasterLoadTimer.start();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    footer: ToolBar {
        height: 25
        RowLayout {
            anchors.fill: parent
            anchors.margins: 2
            spacing: 10

            Label {
                id: statusLabel
                text: " Ready"
                verticalAlignment: Image.AlignVCenter
                font.pixelSize: 12
            }
            ProgressBar {
                id: progressBar
                visible: value > 0 && value < 100
                Layout.rightMargin: 10
                Layout.leftMargin: 10
                Layout.fillWidth: true
                Layout.preferredHeight: 15
                value: 0
                from: 0
                to: 100
            }
        }
    }
}
