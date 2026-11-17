import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

// File > Preferences: models folder, plant and crop models, processing device, what to do when an
// output already exists, and the default cache folder (values in interface/preferences.py, context
// property "preferences"). Changes are applied with OK or Apply; Cancel discards them.
ApplicationWindow {
    id: dlg
    title: qsTr("Preferences")
    width: 680
    height: 500
    minimumWidth: 580
    minimumHeight: 460
    modality: Qt.ApplicationModal
    flags: Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint

    property var defaults: ({})
    property string modelsFolder: ""
    property string plantModel: ""
    property string cropModel: ""
    property bool useGpu: true
    property string whenExists: "ask"
    property string cacheFolder: ""
    property var plantCheck: ({})
    property var cropCheck: ({})
    property string cacheSize: ""
    property string errorText: ""
    readonly property var whenValues: ["ask", "skip", "overwrite"]

    function openDialog() {
        var v = preferences.load();
        defaults = v;
        modelsFolder = v.models_folder;
        plantModel = v.plant_model;
        cropModel = v.crop_model;
        useGpu = v.use_gpu;
        whenExists = v.when_output_exists;
        cacheFolder = v.cache_folder;
        errorText = "";
        tabs.currentIndex = 0;
        checkModels();
        cacheSize = processorInterface.cacheSizeText(defaults.default_cache);
        show();
        raise();
        requestActivate();
    }
    function checkModels() {
        plantCheck = preferences.checkModel(plantModel, "plant_model", modelsFolder);
        cropCheck = preferences.checkModel(cropModel, "crop_model", modelsFolder);
    }
    function apply() {
        errorText = preferences.save({
            "models_folder": modelsFolder,
            "plant_model": plantModel,
            "crop_model": cropModel,
            "use_gpu": useGpu,
            "when_output_exists": whenExists,
            "cache_folder": cacheFolder
        });
        return errorText === "";
    }
    function restoreDefaults() {
        if (tabs.currentIndex === 0) {
            modelsFolder = defaults.default_models_folder;
            plantModel = "";
            cropModel = "";
            useGpu = true;
            checkModels();
        } else {
            whenExists = "ask";
            cacheFolder = "";
        }
    }
    function levelColor(level) {
        return level === "error" ? "#b00020" : (level === "warning" ? "#a05a00" : "#2e7d32");
    }

    FolderDialog {
        id: modelsFolderPicker
        title: qsTr("Choose the folder that contains the models (.onnx)")
        onAccepted: {
            modelsFolder = preferences.localPath(selectedFolder.toString());
            checkModels();
        }
    }
    FileDialog {
        id: modelPicker
        property string key: "plant_model"
        title: key === "plant_model" ? qsTr("Choose the plant detection model") : qsTr("Choose the crop field model")
        nameFilters: ["ONNX models (*.onnx)", "All files (*)"]
        onAccepted: {
            var path = preferences.localPath(selectedFile.toString());
            if (key === "plant_model")
                plantModel = path;
            else
                cropModel = path;
            processorInterface.rememberFolder("models", path);
            checkModels();
        }
    }
    FolderDialog {
        id: cachePicker
        title: qsTr("Choose a folder for the default cache (a .cache_forages_rois folder is made inside)")
        onAccepted: cacheFolder = preferences.localPath(selectedFolder.toString())
    }
    MessageDialog {
        id: clearCacheQuestion
        title: qsTr("Clear Cache")
        text: qsTr("Delete the default cache (%1)?\n%2\n\nPreviews and RGB views are made again when needed.").arg(cacheSize).arg(defaults.default_cache || "")
        buttons: MessageDialog.Yes | MessageDialog.No
        onButtonClicked: function (button, role) {
            if (button === MessageDialog.Yes) {
                processorInterface.clearCache(defaults.default_cache);
                cacheSize = processorInterface.cacheSizeText(defaults.default_cache);
            }
        }
    }

    function checkText(check) {
        if (!check.text)
            return "";
        return (check.level === "ok" ? qsTr("OK. ") : (check.level === "warning" ? qsTr("Warning. ") : qsTr("Cannot be used. "))) + check.text;
    }
    function browseModel(key, path) {
        modelPicker.key = key;
        modelPicker.currentFolder = processorInterface.startFolderUrl("models", path, modelsFolder);
        modelPicker.open();
    }

    // One model row: file (empty = default), Browse, Default, and the result of the check.
    // Self contained (inline components cannot see the ids of this file): it only emits signals.
    component ModelRow: ColumnLayout {
        property string label: ""
        property string path: ""
        property string defaultName: ""
        property string statusText: ""
        property color statusColor: "black"
        signal browse()
        signal reset()
        Layout.fillWidth: true
        spacing: 2
        RowLayout {
            Layout.fillWidth: true
            Label { text: label; Layout.preferredWidth: 170 }
            TextField {
                Layout.fillWidth: true
                readOnly: true
                text: path
                placeholderText: qsTr("Default (%1)").arg(defaultName)
            }
            Button {
                text: qsTr("Browse...")
                onClicked: browse()
            }
            Button {
                text: qsTr("Default")
                enabled: path !== ""
                onClicked: reset()
            }
        }
        Label {
            Layout.fillWidth: true
            Layout.leftMargin: 176
            wrapMode: Text.Wrap
            font.pointSize: 8
            color: statusColor
            text: statusText
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 8

        TabBar {
            id: tabs
            Layout.fillWidth: true
            TabButton { text: qsTr("Models") }
            TabButton { text: qsTr("Processing") }
        }

        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: tabs.currentIndex

            // ---- Models ----
            ScrollView {
                clip: true
                ColumnLayout {
                    width: dlg.width - 40
                    spacing: 10

                    RowLayout {
                        Layout.fillWidth: true
                        Label { text: qsTr("Models folder:"); Layout.preferredWidth: 170 }
                        TextField {
                            Layout.fillWidth: true
                            text: modelsFolder
                            onEditingFinished: {
                                modelsFolder = text;
                                checkModels();
                            }
                        }
                        Button {
                            text: qsTr("Browse...")
                            onClicked: {
                                modelsFolderPicker.currentFolder = processorInterface.startFolderUrl("models", modelsFolder, "");
                                modelsFolderPicker.open();
                            }
                        }
                        Button {
                            text: qsTr("Default")
                            enabled: modelsFolder !== defaults.default_models_folder
                            onClicked: {
                                modelsFolder = defaults.default_models_folder;
                                checkModels();
                            }
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Layout.leftMargin: 176
                        Button {
                            text: qsTr("Download Models")
                            onClicked: processorInterface.downloadModels()
                        }
                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.Wrap
                            font.pointSize: 8
                            text: qsTr("Downloads the missing default models from Hugging Face into the models folder in use (about 100 MB).")
                        }
                    }

                    ModelRow {
                        label: qsTr("Plant detection model:")
                        path: plantModel
                        defaultName: defaults.plant_default || ""
                        statusText: checkText(plantCheck)
                        statusColor: levelColor(plantCheck.level)
                        onBrowse: browseModel("plant_model", plantModel)
                        onReset: { plantModel = ""; checkModels(); }
                    }
                    ModelRow {
                        label: qsTr("Crop field model:")
                        path: cropModel
                        defaultName: defaults.crop_default || ""
                        statusText: checkText(cropCheck)
                        statusColor: levelColor(cropCheck.level)
                        onBrowse: browseModel("crop_model", cropModel)
                        onReset: { cropModel = ""; checkModels(); }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        Label { text: qsTr("Processing device:"); Layout.preferredWidth: 170 }
                        ComboBox {
                            Layout.fillWidth: true
                            model: [qsTr("GPU (CUDA) when available, otherwise CPU"), qsTr("CPU only")]
                            currentIndex: useGpu ? 0 : 1
                            onActivated: useGpu = currentIndex === 0
                        }
                    }
                    Label {
                        Layout.fillWidth: true
                        Layout.leftMargin: 176
                        wrapMode: Text.Wrap
                        font.pointSize: 8
                        text: (defaults.cuda_available ? qsTr("This build can use an NVIDIA GPU (CUDA 12).") : qsTr("This build has no GPU support: the CPU is used."))
                              + (defaults.last_device ? qsTr(" The last model loaded ran on the %1.").arg(defaults.last_device) : "")
                    }
                }
            }

            // ---- Processing ----
            ScrollView {
                clip: true
                ColumnLayout {
                    width: dlg.width - 40
                    spacing: 10

                    RowLayout {
                        Layout.fillWidth: true
                        Label { text: qsTr("When the output already exists:"); Layout.preferredWidth: 200 }
                        ComboBox {
                            Layout.fillWidth: true
                            model: [qsTr("Ask"), qsTr("Skip (keep the existing output)"), qsTr("Overwrite")]
                            currentIndex: Math.max(0, whenValues.indexOf(whenExists))
                            onActivated: whenExists = whenValues[currentIndex]
                        }
                    }
                    Label {
                        Layout.fillWidth: true
                        Layout.leftMargin: 206
                        wrapMode: Text.Wrap
                        font.pointSize: 8
                        text: qsTr("Overwrite deletes the existing output, including the edits saved in it, and runs the task again. plot_numbering always writes into its input shapefile.")
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        Label { text: qsTr("Default cache folder:"); Layout.preferredWidth: 200 }
                        TextField {
                            Layout.fillWidth: true
                            readOnly: true
                            text: cacheFolder
                            placeholderText: qsTr("Default (%1)").arg(defaults.default_cache || "")
                        }
                        Button {
                            text: qsTr("Browse...")
                            onClicked: {
                                cachePicker.currentFolder = processorInterface.startFolderUrl("cache", cacheFolder, "");
                                cachePicker.open();
                            }
                        }
                        Button {
                            text: qsTr("Default")
                            enabled: cacheFolder !== ""
                            onClicked: cacheFolder = ""
                        }
                    }
                    Label {
                        Layout.fillWidth: true
                        Layout.leftMargin: 206
                        wrapMode: Text.Wrap
                        font.pointSize: 8
                        text: qsTr("Used without an output folder (previews, RGB views). With an output folder, the cache is its .cache_forages_rois folder.")
                    }
                    Button {
                        Layout.leftMargin: 206
                        text: qsTr("Clear Default Cache (%1)").arg(cacheSize)
                        onClicked: clearCacheQuestion.open()
                    }
                }
            }
        }

        Label {
            Layout.fillWidth: true
            visible: errorText !== ""
            wrapMode: Text.Wrap
            color: "#b00020"
            text: errorText
        }

        RowLayout {
            Layout.fillWidth: true
            Button {
                text: qsTr("Restore Defaults")
                onClicked: restoreDefaults()
            }
            Item { Layout.fillWidth: true }
            Button {
                text: qsTr("OK")
                highlighted: true
                onClicked: if (apply()) dlg.close()
            }
            Button {
                text: qsTr("Cancel")
                onClicked: dlg.close()
            }
            Button {
                text: qsTr("Apply")
                onClicked: apply()
            }
        }
    }
}
