#from rootprocessor import RootSegmentor

import os
from custom_processor import ForagesROIsDetector

# QGIS style written next to each output shapefile (same name, .qml) so QGIS opens it styled:
# light blue borders only, and grid_id labels in white bold with a dark halo (0 = not numbered)
QGIS_STYLE = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.40.0-Bratislava" styleCategories="Symbology|Labeling" labelsEnabled="1">
  <renderer-v2 type="singleSymbol" forceraster="0" enableorderby="0" symbollevels="0" referencescale="-1">
    <symbols>
      <symbol type="fill" name="0" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleFill" pass="0" locked="0" enabled="1">
          <Option type="Map">
            <Option type="QString" name="style" value="no"/>
            <Option type="QString" name="outline_color" value="90,190,255,255,rgb:0.353,0.745,1,1"/>
            <Option type="QString" name="outline_style" value="solid"/>
            <Option type="QString" name="outline_width" value="0.8"/>
            <Option type="QString" name="outline_width_unit" value="MM"/>
            <Option type="QString" name="joinstyle" value="round"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="simple">
    <settings calloutType="simple">
      <text-style fieldName="CASE WHEN &quot;grid_id&quot; &gt; 0 THEN &quot;grid_id&quot; END" isExpression="1" fontFamily="Open Sans" fontSize="10" fontSizeUnit="Point" fontWeight="75" forcedBold="1" namedStyle="Bold" textColor="255,255,255,255,rgb:1,1,1,1" textOpacity="1">
        <text-buffer bufferDraw="1" bufferSize="0.8" bufferSizeUnits="MM" bufferColor="40,40,40,220,rgb:0.157,0.157,0.157,0.863" bufferOpacity="1" bufferNoFill="1" bufferJoinStyle="128"/>
      </text-style>
      <placement placement="1" quadOffset="4" dist="0" offsetType="0" layerType="PolygonGeometry" centroidInside="1" centroidWhole="0" overlapHandling="AllowOverlapIfRequired" priority="5"/>
      <rendering scaleVisibility="0" obstacle="0" drawLabels="1" fontLimitPixelSize="0" upsidedownLabels="0" limitNumLabels="0"/>
    </settings>
  </labeling>
</qgis>
"""


class Processor():

    def __init__(self, params, progress_callback = None, interruption_check = None):
        self.params = params
        self.progress_callback = progress_callback
        self.interruption_check = interruption_check

    def run(self):

        results = self.params


        task = self.params.get("task")
        output_filepath = None  # set by each task, checked at the end
        skipped = False         # True when the output already existed and nothing was run

        #***************
        if task == "detection":

            input_file = self.params.get("input_file")
            output_folder = self.params.get("output_folder")

            self.forages_rois_detector = ForagesROIsDetector()
            self.forages_rois_detector.inference(input_file, output_folder)



            results.update({"status": "completed", "message": "Task completed succesfully."})

        elif task == "tiling_detection":

            import os
            input_file = self.params.get("input_file")
            output_folder = self.params.get("output_folder")
            
            if not output_folder.lower().endswith(".shp"):
                basename = os.path.splitext(os.path.basename(input_file))[0]
                output_filepath = os.path.join(output_folder, basename + ".shp")
            else:
                output_filepath = output_folder

            self.forages_rois_detector = ForagesROIsDetector()
            
            if os.path.exists(output_filepath):
                print(f"Output already exists, skipping inference: {output_filepath}")
                skipped = True
            else:
                self.forages_rois_detector.tile_inference(input_file, output_filepath, progress_callback=self.progress_callback, interruption_check=self.interruption_check)

            results.update({"status": "completed", "message": "Task completed succesfully."})

        elif task == "crop_detection":

            import os
            input_file = self.params.get("input_file")
            output_folder = self.params.get("output_folder")
            
            if not output_folder.lower().endswith(".shp"):
                basename = os.path.splitext(os.path.basename(input_file))[0]
                output_filepath = os.path.join(output_folder, basename + "_crops.shp")
            else:
                output_filepath = output_folder

            self.forages_rois_detector = ForagesROIsDetector()
            
            if os.path.exists(output_filepath):
                print(f"Output already exists, skipping inference: {output_filepath}")
                skipped = True
            else:
                self.forages_rois_detector.crop_inference(input_file, output_filepath, progress_callback=self.progress_callback, interruption_check=self.interruption_check)

            results.update({"status": "completed", "message": "Task completed succesfully."})

        elif task == "tiling_detection_on_crop":

            import os
            input_file = self.params.get("input_file")
            output_folder = self.params.get("output_folder")
            crop_shapefile = self.params.get("crop_shapefile")
            
            basename = os.path.splitext(os.path.basename(input_file))[0]
            if not crop_shapefile:
                if not output_folder.lower().endswith(".shp"):
                    crop_shapefile = os.path.join(output_folder, basename + "_crops.shp")
                else:
                    crop_shapefile = os.path.join(os.path.dirname(output_folder), basename + "_crops.shp")

            if not output_folder.lower().endswith(".shp"):
                output_filepath = os.path.join(output_folder, basename + "_on_crop.shp")
            else:
                output_filepath = output_folder

            if not os.path.exists(crop_shapefile):
                error_msg = f"Error: Crop shapefile not found at {crop_shapefile}"
                print(error_msg)
                if self.progress_callback:
                    self.progress_callback({"status": error_msg, "progress": 0})
                results.update({"status": "error", "message": error_msg})
                return results

            self.forages_rois_detector = ForagesROIsDetector()
            
            if os.path.exists(output_filepath):
                print(f"Output already exists, skipping inference: {output_filepath}")
                skipped = True
            else:
                self.forages_rois_detector.tile_inference_on_crop(input_file, output_filepath, crop_shapefile, progress_callback=self.progress_callback, interruption_check=self.interruption_check)

            results.update({"status": "completed", "message": "Task completed succesfully."})

        elif task == "plot_numbering":

            import os
            # GUI sends the detections shapefile separately; CLI passes it as --input
            input_file = self.params.get("input_shapefile") or self.params.get("input_file")
            output_folder = self.params.get("output_folder")
            
            if not output_folder.lower().endswith(".shp"):
                basename = os.path.splitext(os.path.basename(input_file))[0]
                output_filepath = os.path.join(output_folder, basename + "_numbering.shp")
            else:
                output_filepath = output_folder

            align = self.params.get("align", False)
            serpentine = self.params.get("serpentine", False)

            self.forages_rois_detector = ForagesROIsDetector()
            
            if os.path.exists(output_filepath):
                print(f"Output already exists, skipping inference: {output_filepath}")
                skipped = True
            else:
                self.forages_rois_detector.plot_numbering(input_file, output_filepath
                                                          , align_to_grid=align
                                                          , serpentine=serpentine
                                                          , start_corner=self.params.get("start_corner") or "NW"
                                                          , swap_axes=self.params.get("swap_axes", False))

            results.update({"status": "completed", "message": "Task completed succesfully."})

        elif task == "postprocessing":

            import os
            # GUI sends the detections shapefile separately; CLI passes it as --input
            input_file = self.params.get("input_shapefile") or self.params.get("input_file")
            output_folder = self.params.get("output_folder")
            
            if not output_folder.lower().endswith(".shp"):
                basename = os.path.splitext(os.path.basename(input_file))[0]
                output_filepath = os.path.join(output_folder, basename + "_postprocess.shp")
            else:
                output_filepath = output_folder

            #if key exists in params, use it, otherwise set default value
            align = self.params.get("align", False)
            serpentine = self.params.get("serpentine", False)

            self.forages_rois_detector = ForagesROIsDetector()
            
            if os.path.exists(output_filepath):
                print(f"Output already exists, skipping inference: {output_filepath}")
                skipped = True
            else:
                self.forages_rois_detector.plot_numbering(input_file, output_filepath
                                                          , only_postprocess=True
                                                          , align_to_grid=align
                                                          , serpentine=serpentine)

            results.update({"status": "completed", "message": "Task completed succesfully."})

        elif task == "tiling_detection_only":

            import os
            input_file = self.params.get("input_file")
            output_folder = self.params.get("output_folder")
            
            if not output_folder.lower().endswith(".shp"):
                basename = os.path.splitext(os.path.basename(input_file))[0]
                output_filepath = os.path.join(output_folder, basename + ".shp")
            else:
                output_filepath = output_folder

            self.forages_rois_detector = ForagesROIsDetector()
            
            if os.path.exists(output_filepath):
                print(f"Output already exists, skipping inference: {output_filepath}")
                skipped = True
            else:
                self.forages_rois_detector.tile_inference(input_file, output_filepath, only=True, progress_callback=self.progress_callback, interruption_check=self.interruption_check)

            results.update({"status": "completed", "message": "Task completed succesfully."})


            

        # if task == "batch_segmentation":

        #     # Perform batch segmentation
        #     input_folder = self.params.get("input_folder")
        #     output_folder = self.params.get("output_folder")

        #     # Initialize the BackgroundRemover and perform batch processing
        #     self.background_remover = RootSegmentor()
        #     self.background_remover.batch_processing(input_folder
        #                                             , output_folder
        #                                             , progress_callback = self.progress_callback
        #                                             , interruption_check = self.interruption_check)
        
        # # elif task == "single_segmentation":
        # #     # Perform batch segmentation
        # #     input_file = self.params.get("input_file")
        # #     output_folder = self.params.get("output_folder")

        # #     # Initialize the BackgroundRemover and perform batch processing
        # #     self.background_remover = BackgroundRemover()
        # #     self.background_remover.inference_file_save(input_file
        # #                                             , output_folder
        # #                                             , progress_callback = self.progress_callback
        # #                                             , interruption_check = self.interruption_check)            


        else:
            results.update({"status": "error", "message": "Invalid task specified."})





        #****************
        # Report what actually happened: an existing output means nothing was run,
        # and a missing output means the task failed even if no exception was raised
        if results.get("status") == "completed" and output_filepath:
            if skipped:
                results.update({"output_file": output_filepath,
                                "message": f"Output already exists, nothing was run:\n{output_filepath}\n\nDelete it or choose another output folder to run again."})
            elif not os.path.exists(output_filepath):
                results.update({"status": "error",
                                "message": f"No output was created:\n{output_filepath}\n\nCheck that the input is valid for the task '{task}'."})
            else:
                results.update({"output_file": output_filepath,
                                "message": f"Task completed successfully.\n\nOutput:\n{output_filepath}"})
                # Save the QGIS style next to the output (same name, .qml) so QGIS opens it styled
                if output_filepath.lower().endswith(".shp"):
                    with open(os.path.splitext(output_filepath)[0] + ".qml", "w", encoding="utf-8") as f:
                        f.write(QGIS_STYLE)

        print(results)

        return results

    
