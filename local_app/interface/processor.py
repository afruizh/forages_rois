#from rootprocessor import RootSegmentor

import os
from custom_processor import ForagesROIsDetector, PLANT_CONFIDENCE, raster_pixel_size_m

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
          <data_defined_properties>
            <Option type="Map">
              <Option type="QString" name="name" value=""/>
              <Option type="Map" name="properties">
                <Option type="Map" name="outlineColor">
                  <Option type="bool" name="active" value="true"/>
                  <Option type="QString" name="expression" value="CASE WHEN &quot;status&quot; = 'empty' THEN '200,200,200,255' ELSE '90,190,255,255' END"/>
                  <Option type="int" name="type" value="3"/>
                </Option>
                <Option type="Map" name="outlineStyle">
                  <Option type="bool" name="active" value="true"/>
                  <Option type="QString" name="expression" value="CASE WHEN &quot;status&quot; = 'empty' THEN 'dash' ELSE 'solid' END"/>
                  <Option type="int" name="type" value="3"/>
                </Option>
              </Option>
              <Option type="QString" name="type" value="collection"/>
            </Option>
          </data_defined_properties>
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

    def confidence(self):
        # Plant detection threshold; empty means the default PLANT_CONFIDENCE
        value = self.params.get("confidence")
        return float(value) if value not in (None, "") else PLANT_CONFIDENCE

    def empty_plots(self):
        # Numbering with empty plots for missing plants, off by default; GUI sends empty_plots, CLI --empty_plots
        return bool(self.params.get("empty_plots", False))

    def use_rgb_view(self):
        # Rasters that are not 8 bit RGB (multispectral, 16 bit, reflectance): the tasks work on an
        # RGB view with the chosen bands (GUI Bands lists, CLI --bands; empty: guessed), see rgb_bands
        # For plant detection, rasters finer than the model's training resolution are also resampled
        # to it (MODEL_RESOLUTION; GUI Advanced > Detection, CLI --no_resample to turn it off)
        from processing.rgb_bands import rgb_raster, parse_bands
        from utils.cachemanager import cache_folder
        from custom_processor import MODEL_FILENAME, MODEL_RESOLUTION, raster_pixel_size_m
        task = self.params.get("task")
        if task in ("plot_numbering", "postprocessing", "plant_traits"):
            return  # plant_traits reads the original bands (spectral_indices), not the RGB view
        resolution = MODEL_RESOLUTION.get(MODEL_FILENAME)
        # export_coco also, so the exported tiles have the resolution the detection uses
        resample = (task in ("tiling_detection", "tiling_detection_on_crop", "tiling_detection_only", "export_coco") and resolution
                    and self.params.get("resample", True) and not self.params.get("no_resample", False))
        self.params = dict(self.params)
        for key in ("input_file", "raster"):
            path = self.params.get(key)
            if path and not path.lower().endswith(".shp") and os.path.isfile(path):
                pixel = raster_pixel_size_m(path) if resample else None
                fine = bool(pixel and pixel < resolution["finest"])
                self.params[key] = rgb_raster(path, parse_bands(self.params.get("bands")),
                                              cache_folder(self.params.get("output_folder")),
                                              pixel_m=pixel if fine else None, target_m=resolution["target"] if fine else None)

    def run(self):

        self.use_rgb_view()
        results = self.params


        task = self.params.get("task")
        output_filepath = None  # set by each task, checked at the end
        skipped = False         # True when the output already existed and nothing was run

        #***************
        if task == "detection":

            input_file = self.params.get("input_file")
            output_folder = self.params.get("output_folder")

            self.forages_rois_detector = ForagesROIsDetector()
            self.forages_rois_detector.inference(input_file, output_folder, conf_threshold=self.confidence())



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
                self.forages_rois_detector.tile_inference(input_file, output_filepath, progress_callback=self.progress_callback, interruption_check=self.interruption_check,
                                                          conf_threshold=self.confidence(), empty_plots=self.empty_plots())

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
                self.forages_rois_detector.tile_inference_on_crop(input_file, output_filepath, crop_shapefile, progress_callback=self.progress_callback, interruption_check=self.interruption_check,
                                                                  conf_threshold=self.confidence(), empty_plots=self.empty_plots())

            results.update({"status": "completed", "message": "Task completed succesfully."})

        elif task == "plot_numbering":

            import os
            # GUI sends the detections shapefile separately; CLI passes it as --input
            input_file = self.params.get("input_shapefile") or self.params.get("input_file")
            output_folder = self.params.get("output_folder")
            
            # The numbering is written into the input shapefile itself, like the edits; an output
            # given as a .shp path (CLI) writes a new shapefile instead
            if output_folder and output_folder.lower().endswith(".shp"):
                output_filepath = output_folder
            else:
                output_filepath = input_file

            align = self.params.get("align", False)
            serpentine = self.params.get("serpentine", False)

            self.forages_rois_detector = ForagesROIsDetector()
            self.forages_rois_detector.plot_numbering(input_file, output_filepath
                                                      , align_to_grid=align
                                                      , serpentine=serpentine
                                                      , start_corner=self.params.get("start_corner") or "NW"
                                                      , swap_axes=self.params.get("swap_axes", False)
                                                      , empty_plots=self.empty_plots())

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

        elif task == "plant_traits":

            import os
            # Needs both the raster and the plant ROIs: GUI sends them as input_file and
            # input_shapefile; the CLI passes the ROIs as --input and the raster as --raster
            rois_file = self.params.get("input_shapefile") or self.params.get("input_file")
            raster_file = self.params.get("raster") or self.params.get("input_file")
            output_folder = self.params.get("output_folder")

            if not output_folder.lower().endswith(".shp"):
                basename = os.path.splitext(os.path.basename(rois_file))[0]
                output_filepath = os.path.join(output_folder, basename + "_traits.shp")
            else:
                output_filepath = output_folder

            if not raster_file or raster_file.lower().endswith(".shp") or not os.path.exists(raster_file):
                results.update({"status": "error", "message": "Plant traits need the orthomosaic as Input Raster."})
                return results
            if not rois_file or not rois_file.lower().endswith(".shp") or not os.path.exists(rois_file):
                results.update({"status": "error", "message": "Plant traits need the plant ROIs shapefile as Input Shapefile."})
                return results

            threshold = self.params.get("threshold")
            threshold = float(threshold) if threshold not in (None, "") else None  # empty: automatic (Otsu)

            self.forages_rois_detector = ForagesROIsDetector()

            if os.path.exists(output_filepath):
                print(f"Output already exists, skipping inference: {output_filepath}")
                skipped = True
            else:
                from processing.rgb_bands import parse_bands
                from processing.spectral_indices import find_bands
                # Bands of the original raster: R, G, B as for the RGB view (GUI Bands lists, CLI
                # --bands), near infrared and red edge from --nir_band and --rededge_band or the band names
                bands = find_bands(raster_file, parse_bands(self.params.get("bands")),
                                   nir=self.params.get("nir_band"), rededge=self.params.get("rededge_band"))
                index = self.params.get("index")
                self.forages_rois_detector.plant_traits(raster_file, rois_file, output_filepath,
                                                        index=None if index in (None, "", "Auto") else index,
                                                        threshold=threshold,
                                                        progress_callback=self.progress_callback,
                                                        bands=bands)

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
                self.forages_rois_detector.tile_inference(input_file, output_filepath, only=True, progress_callback=self.progress_callback, interruption_check=self.interruption_check,
                                                          conf_threshold=self.confidence())

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


        elif task == "export_coco":

            import os
            # Reviewed plant boxes as a COCO dataset of RGB tiles (see coco_export). Inputs as for
            # plant_traits: GUI input_file and input_shapefile; CLI --raster and --input
            rois_file = self.params.get("input_shapefile") or self.params.get("input_file")
            raster_file = self.params.get("raster") or self.params.get("input_file")
            output_folder = self.params.get("output_folder")
            if output_folder.lower().endswith(".shp"):
                output_folder = os.path.dirname(output_folder)
            if not raster_file or raster_file.lower().endswith(".shp") or not os.path.exists(raster_file):
                results.update({"status": "error", "message": "The COCO export needs the orthomosaic as Input Raster."})
                return results
            if not rois_file or not rois_file.lower().endswith(".shp") or not os.path.exists(rois_file):
                results.update({"status": "error", "message": "The COCO export needs the plant boxes shapefile as Input Shapefile."})
                return results

            dataset_dir = os.path.join(output_folder, os.path.splitext(os.path.basename(rois_file))[0] + "_coco")
            output_filepath = os.path.join(dataset_dir, "annotations", "instances_default.json")
            # Crop fields: the chosen ones, else <raster>_crops.shp of the output folder if it exists
            crop_shapefile = self.params.get("crop_shapefile") or os.path.join(
                output_folder, os.path.splitext(os.path.basename(raster_file))[0] + "_crops.shp")

            if os.path.exists(output_filepath):
                print(f"Output already exists, skipping export: {output_filepath}")
                skipped = True
            else:
                from processing.coco_export import export_coco
                export_coco(raster_file, rois_file, dataset_dir,
                            crop_shp=crop_shapefile if os.path.exists(crop_shapefile) else None,
                            progress_callback=self.progress_callback, interruption_check=self.interruption_check)

            results.update({"status": "completed", "message": "Task completed succesfully."})

        else:
            results.update({"status": "error", "message": "Invalid task specified."})





        #****************
        # Report what actually happened: an existing output means nothing was run,
        # and a missing output means the task failed even if no exception was raised
        results["skipped"] = skipped
        if results.get("status") == "completed" and output_filepath:
            if skipped:
                results.update({"output_file": output_filepath,
                                "message": f"Output already exists, nothing was run:\n{output_filepath}\n\nDelete it or choose another output folder to run again."})
            elif not os.path.exists(output_filepath) and task in ("tiling_detection", "tiling_detection_on_crop", "tiling_detection_only"):
                # Detection tasks write no output when no plant is detected, usually a too coarse raster
                pixel = raster_pixel_size_m(input_file)
                pixel_text = f"about {pixel * 100:.0f} cm" if pixel else "unknown"
                results.update({"status": "error",
                                "message": f"No plants were detected, so no output was created:\n{output_filepath}\n\n"
                                           f"Pixel size of the raster: {pixel_text}. The plant model was trained on "
                                           f"orthomosaics of 1 to 2 cm per pixel and also works at about 3 cm; with much larger pixels plants are too small to detect. "
                                           f"Use the full resolution orthomosaic (1 to 3 cm per pixel) or lower the confidence threshold in Advanced."})
            elif not os.path.exists(output_filepath):
                results.update({"status": "error",
                                "message": f"No output was created:\n{output_filepath}\n\nCheck that the input is valid for the task '{task}'."})
            else:
                results.update({"output_file": output_filepath,
                                "message": f"Task completed successfully.\n\nOutput:\n{output_filepath}"})
                # Save the QGIS style next to the output (same name, .qml) so QGIS opens it styled
                if output_filepath.lower().endswith(".shp"):
                    style = QGIS_STYLE
                    if task == "plant_traits":
                        # Plant outlines in green, as in the preview, so they differ from the boxes
                        style = style.replace("90,190,255,255,rgb:0.353,0.745,1,1", "60,220,90,255,rgb:0.235,0.863,0.353,1")
                        style = style.replace("'90,190,255,255'", "'60,220,90,255'")
                    with open(os.path.splitext(output_filepath)[0] + ".qml", "w", encoding="utf-8") as f:
                        f.write(style)

        print(results)

        return results

    
