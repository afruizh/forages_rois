# -*- coding: utf-8 -*-

"""
/***************************************************************************
 ForagesROIs
                                 A QGIS plugin
 Detection, classification and grid generation for forages
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2025-05-14
        copyright            : (C) 2025 by Andres Felipe Ruiz-Hurtado, Tropical Forages Program CIAT
        email                : a.f.ruiz@cgiar.org
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

__author__ = 'Andres Felipe Ruiz-Hurtado, Tropical Forages Program CIAT'
__date__ = '2025-05-14'
__copyright__ = '(C) 2025 by Andres Felipe Ruiz-Hurtado, Tropical Forages Program CIAT'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsFeatureSink,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink)

from qgis.core import QgsProcessingParameterRasterLayer, QgsProcessingParameterFileDestination
import subprocess
import tempfile
from qgis.core import QgsVectorLayer, QgsProject


class ROIsDetectionAlgorithm(QgsProcessingAlgorithm):
    """
    This is an example algorithm that takes a vector layer and
    creates a new identical one.

    It is meant to be used as an example of how to create your own
    algorithms and explain methods and variables used to do it. An
    algorithm like this will be available in all elements, and there
    is not need for additional work.

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    OUTPUT = 'OUTPUT'
    INPUT = 'INPUT'

    def initAlgorithm(self, config):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        # Change input to raster layer selector
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                self.tr('Input raster layer')
            )
        )

        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT,
                self.tr('Output shapefile'),
                'ESRI Shapefile (*.shp)'
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """
        import os
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if raster_layer is None:
            raise Exception('Invalid raster layer')
            
        raster_path = raster_layer.source()
        output_path = self.parameterAsFileOutput(parameters, self.OUTPUT, context)

        if not os.path.exists(output_path):

            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            cwd_path = os.path.join(plugin_dir, 'ForagesROIs')
            exe_path = os.path.join(plugin_dir, 'ForagesROIs', 'ForagesROIs.exe')

            # Prepare environment with cwd_path added to PATH
            env = os.environ.copy()
            env["PATH"] = cwd_path# + os.pathsep + env.get("PATH", "")
            #proj_lib_path = r"C:\Program Files\QGIS 3.34.2\share\proj"  # Adjust if your QGIS is in a different folder
            #env["PROJ_LIB"] = cwd_path

            # Add cwd_path to the Python process PATH if not already present
            if cwd_path not in os.environ["PATH"]:
                os.environ["PATH"] = cwd_path# + os.pathsep + os.environ.get("PATH", "")
            #os.environ["PROJ_LIB"] = cwd_path

            #cmd = [exe_path, '--cli', '--input', os.path.normpath(raster_path), '--output', os.path.normpath(output_path)]
            cmd = ['ForagesROIs.exe', '--cli','--task','tiling_detection_only', '--input', os.path.normpath(raster_path), '--output', os.path.normpath(output_path)]
            feedback.pushInfo(f"Process working directory (cwd): {cwd_path}")
            feedback.pushInfo(f"Process PATH: {env['PATH']}")
            #feedback.pushInfo(f"PROJ_LIB: {env['PROJ_LIB']}")
            feedback.pushInfo(f'Running: {" ".join(cmd)}')
            feedback.setProgress(10)  # Set progress to 10% before running the subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd_path, env=env, encoding='latin1')
            feedback.setProgress(80)  # Set progress to 80% after subprocess completes
            
            feedback.pushInfo(result.stdout)
            
            # Always show stderr if present
            if result.stderr:
                feedback.reportError(result.stderr)
                feedback.pushInfo(f"STDERR: {result.stderr}")

            if result.returncode != 0:
                raise Exception(f'Error running ForagesROIs.exe: {result.stderr}')

            

        feedback.pushInfo("Loading output shapefile into QGIS...")

        # Load the output shapefile into QGIS
        if os.path.exists(output_path):
            if context.willLoadLayerOnCompletion(output_path):
                style_path = os.path.join(os.path.dirname(__file__), "ForagesROIs_style.qml")
                def apply_style(layer):
                    if os.path.exists(style_path):
                        layer.loadNamedStyle(style_path)
                        layer.triggerRepaint()
                    else:
                        feedback.pushInfo(f"Style file not found: {style_path}")
                context.addLayerToLoadOnCompletion(
                    output_path,
                    {
                        'layerName': os.path.basename(output_path),
                        'provider': 'ogr',
                        'postProcessor': apply_style
                    }
                )
                feedback.pushInfo(f"Layer scheduled to load: {output_path}")
            else:
                feedback.pushInfo("Context will not load layer on completion.")

                # Manually load the layer into the project
                layer = QgsVectorLayer(output_path, os.path.basename(output_path), "ogr")
                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
                    feedback.pushInfo(f"Layer loaded manually: {output_path}")
                    # Optionally apply style
                    style_path = os.path.join(os.path.dirname(__file__), "ForagesROIs_style.qml")
                    if os.path.exists(style_path):
                        layer.loadNamedStyle(style_path)
                        layer.triggerRepaint()
                else:
                    feedback.reportError(f"Failed to load layer manually: {output_path}")
        else:
            feedback.reportError(f"Output file does not exist: {output_path}")

        return {self.OUTPUT: output_path}

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'ROIsDetection'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr(self.name())

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr(self.groupId())

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'ForagesROIs'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return ROIsDetectionAlgorithm()
