#from rootprocessor import RootSegmentor

from custom_processor import ForagesROIsDetector


class Processor():

    def __init__(self, params, progress_callback = None, interruption_check = None):
        self.params = params
        self.progress_callback = progress_callback
        self.interruption_check = interruption_check

    def run(self):

        results = self.params


        task = self.params.get("task")

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
            else:
                self.forages_rois_detector.tile_inference(input_file, output_filepath, progress_callback=self.progress_callback, interruption_check=self.interruption_check)

            results.update({"status": "completed", "message": "Task completed succesfully."})

        elif task == "plot_numbering":

            import os
            input_file = self.params.get("input_file")
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
            else:
                self.forages_rois_detector.plot_numbering(input_file, output_filepath
                                                          , align_to_grid=align
                                                          , serpentine=serpentine)

            results.update({"status": "completed", "message": "Task completed succesfully."})

        elif task == "postprocessing":

            import os
            input_file = self.params.get("input_file")
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
        print(results)

        return results

    
