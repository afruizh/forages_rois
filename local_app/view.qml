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
            //MenuSeparator { }
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

    Window  {
        id: aboutWindow
        title: qsTr("About")
        width: 630
        height: 250
        modality: Qt.ApplicationModal
        flags: Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint

        ColumnLayout {
            anchors.fill:parent
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
                            anchors.fill:parent
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
                            anchors.fill:parent
                            anchors.margins: 10

                            RowLayout {
                                anchors.fill:parent
                                Image {
                                    Layout.fillWidth:true
                                    Layout.fillHeight:true
                                    source: "../res/tf.png"
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
            inputRasterPath = urlToLocalPath(fileDialog.currentFile.toString())
            loadingIndicator.running = true
            rasterLoadTimer.start()
        }
    }

    FolderDialog {
        id: folderDialog
        title: qsTr("Please choose an output folder")
        onAccepted: {
            outputFolderPath = urlToLocalPath(folderDialog.currentFolder.toString())
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
                processorInterface.openOutputFolder(outputFolderPath)
                break;
            }
        }
    }

    Connections {
        target: processorInterface
        function onProgressUpdated(info) {
            if (info.progress !== undefined) {
                progressBar.value = info.progress
            }
            if (info.status !== undefined) {
                statusLabel.text = info.status
            }
        }
        function onVisualizationReady(imagePath) {
            previewImage.source = "file:///" + imagePath + "?" + new Date().getTime()
            statusLabel.text = "Visualization Ready!"
            progressBar.value = 100
            infoDialog.open()
        }
        function onRasterPreviewReady(imagePath) {
            previewImage.source = "file:///" + imagePath + "?" + new Date().getTime()
            rasterItem.sourcePath = inputRasterPath
            rasterItem.shapefilePath = ""
            statusLabel.text = "Raster preview generated. Ready for inference."
        }
        function onVisualizationPathsReady(rasterPath, shpPath) {
            rasterItem.sourcePath = rasterPath
            rasterItem.shapefilePath = shpPath
            console.log("Vector paths set:", rasterPath, shpPath)
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
                        color: "transparent"
                        border.color: "lightgray"
                        border.width: 1

                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 0

                            TabBar {
                                id: leftTabBar
                                Layout.fillWidth: true
                                TabButton { text: qsTr("Settings") }
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
                                            Label { text: "Input Raster:"; font.bold: true }
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
                                                    onClicked: fileDialog.open()
                                                }
                                            }
                                        }

                                        // Output Folder Selection
                                        ColumnLayout {
                                            Layout.fillWidth: true
                                            spacing: 5
                                            Label { text: "Output Folder:"; font.bold: true }
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
                                                    onClicked: folderDialog.open()
                                                }
                                            }
                                        }

                                        // Task Selection
                                        ColumnLayout {
                                            Layout.fillWidth: true
                                            spacing: 5
                                            Label { text: "Task:"; font.bold: true }
                                            ComboBox {
                                                id: taskCombo
                                                Layout.fillWidth: true
                                                model: ["tiling_detection", "plot_numbering", "postprocessing"]
                                            }
                                        }

                                        // Process Button
                                        ColumnLayout {
                                            Layout.fillWidth: true
                                            spacing: 5
                                            
                                            CheckBox {
                                                id: useVectorOverlayCheckbox
                                                text: "Use Native Vector Overlay (High Res)"
                                                checked: true
                                            }

                                            Button {
                                                text: "Run Inference"
                                                Layout.fillWidth: true
                                                font.bold: true
                                                enabled: true//inputRasterPath !== "" && outputFolderPath !== ""
                                                onClicked: {
                                                    progressBar.value = 0
                                                    statusLabel.text = "Processing started..."
                                                    var params = {
                                                        "task": taskCombo.currentText,
                                                        "input_file": inputRasterPath,
                                                        "output_folder": outputFolderPath,
                                                        "use_vector_overlay": useVectorOverlayCheckbox.checked
                                                    }
                                                    processorInterface.process(params)
                                                }
                                            }
                                            Button {
                                                id: cancelButton
                                                text: "Cancel"
                                                Layout.fillWidth: true
                                                onClicked: processorInterface.cancelProcessing()
                                            }
                                        }

                                        Item { Layout.fillHeight: true } // Spacer
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
                                TabButton { text: qsTr("Preview") }
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
                                            processorInterface.previewRaster(inputRasterPath)
                                            loadingIndicator.running = false
                                        }
                                    }

                                    RasterPreviewItem {
                                        id: rasterItem
                                        anchors.fill: parent
                                        visible: useVectorOverlayCheckbox.checked
                                        contentX: flickable.contentX
                                        contentY: flickable.contentY
                                        zoomScale: previewContainer.currentZoom
                                        showVectorOverlay: true
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
                                        onWheel: (wheel) => {
                                            var zoomFactor = wheel.angleDelta.y > 0 ? 1.1 : 0.9
                                            var newScale = previewContainer.currentZoom * zoomFactor
                                            if (newScale >= 0.001 && newScale <= 100.0) {
                                                var oldScale = previewContainer.currentZoom
                                                var mouseXRel = (wheel.x + flickable.contentX) / oldScale
                                                var mouseYRel = (wheel.y + flickable.contentY) / oldScale
                                                
                                                previewContainer.currentZoom = newScale
                                                
                                                flickable.contentX = (mouseXRel * newScale) - wheel.x
                                                flickable.contentY = (mouseYRel * newScale) - wheel.y
                                            }
                                        }
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
                                        onEntered: (drag) => {
                                            drag.accept(Qt.LinkAction)
                                        }
                                        onDropped: (drop) => {
                                            if (drop.hasUrls) {
                                                inputRasterPath = urlToLocalPath(drop.urls[0].toString())
                                                loadingIndicator.running = true
                                                rasterLoadTimer.start()
                                                previewContainer.currentZoom = 1.0 // Reset zoom
                                                flickable.contentX = 0
                                                flickable.contentY = 0
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
                visible: value > 0 && value < 100
                Layout.rightMargin: 10
                Layout.leftMargin: 10
                id: progressBar
                Layout.fillWidth: true
                Layout.preferredHeight: 15
                value: 0
                from: 0
                to: 100
            }
        }
    }
}
