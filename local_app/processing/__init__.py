"""Processing steps used by custom_processor and the Processor, without Qt.

rgb_bands         8 bit RGB view of a raster (band choice, stretch, resampling) as a GDAL VRT
spectral_indices  vegetation indices of the original bands (RGBVI, ExG, NDVI, NDRE, GNDVI)
coco_export       reviewed plant boxes as a COCO dataset of RGB tiles
yolo_seg_util     decoding of the crop field segmentation model output
qgis2coco         earlier COCO converter (from TreeEyed), kept for reference
"""
