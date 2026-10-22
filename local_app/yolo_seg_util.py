import cv2
import numpy as np

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def process_mask(protos, masks_in, bboxes, shape):
    """
    Process masks.
    protos: (32, 160, 160)
    masks_in: (N, 32)
    bboxes: (N, 4) in scaled image coordinates (xyxy)
    shape: (input_size, input_size) usually (640, 640)
    """
    if len(masks_in) == 0:
        return np.zeros((0, shape[0], shape[1]), dtype=np.uint8)
        
    c, mh, mw = protos.shape
    masks = (masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)  # (N, 160, 160)
    masks = 1 / (1 + np.exp(-masks))  # sigmoid
    
    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= mw / shape[1]
    downsampled_bboxes[:, 2] *= mw / shape[1]
    downsampled_bboxes[:, 1] *= mh / shape[0]
    downsampled_bboxes[:, 3] *= mh / shape[0]
    
    masks_out = []
    for i in range(len(masks)):
        mask = masks[i]
        x1, y1, x2, y2 = downsampled_bboxes[i].astype(int)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(mw, x2), min(mh, y2)
        
        cropped_mask = np.zeros_like(mask)
        if x1 < x2 and y1 < y2:
            cropped_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            
        mask_resized = cv2.resize(cropped_mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        masks_out.append(mask_resized > 0.5)
        
    return np.array(masks_out)

def postprocess_yolov8_seg_output(outputs, conf_threshold=0.3, nms_threshold=0.5, input_size=1024, orig_shape=(1024, 1024)):
    output = outputs[0]
    output = np.squeeze(output, axis=0)
    if output.shape[0] < output.shape[1]:
        output = output.transpose(1, 0)
        
    protos = np.squeeze(outputs[1], axis=0)  # (32, 160, 160)
    num_masks = protos.shape[0]
    
    # Check if this is an End2End ONNX export with NMS already applied
    is_end2end = output.shape[1] == 4 + 2 + num_masks
    
    if is_end2end:
        boxes_xyxy = output[:, :4]
        confidences = output[:, 4]
        class_ids = output[:, 5].astype(np.int32)
        mask_coeffs = output[:, 6:]
        
        mask = confidences > conf_threshold
        final_boxes = boxes_xyxy[mask]
        final_classes = class_ids[mask]
        final_scores = confidences[mask]
        final_mask_coeffs = mask_coeffs[mask]
    else:
        num_classes = output.shape[1] - 4 - num_masks
        
        boxes_xywh = output[:, :4]
        scores = output[:, 4:4+num_classes]
        mask_coeffs = output[:, 4+num_classes:]
        
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        mask = confidences > conf_threshold
        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        mask_coeffs = mask_coeffs[mask]
        
        if len(boxes_xywh) == 0:
            return [[np.zeros((0, 5))], [np.zeros((0,), dtype=np.int32)], []]
            
        boxes_xyxy = xywh2xyxy(boxes_xywh)
        
        final_boxes_list = []
        final_scores_list = []
        final_classes_list = []
        final_mask_coeffs_list = []
        
        for cls in np.unique(class_ids):
            cls_mask = class_ids == cls
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = confidences[cls_mask]
            cls_mask_coeffs = mask_coeffs[cls_mask]
            
            boxes_nms = cls_boxes.copy()
            boxes_nms[:, 2] -= boxes_nms[:, 0]
            boxes_nms[:, 3] -= boxes_nms[:, 1]
            
            indices = cv2.dnn.NMSBoxes(
                bboxes=boxes_nms.tolist(),
                scores=cls_scores.tolist(),
                score_threshold=conf_threshold,
                nms_threshold=nms_threshold
            )
            
            if len(indices) > 0:
                indices = np.array(indices).flatten()
                final_boxes_list.append(cls_boxes[indices])
                final_scores_list.append(cls_scores[indices])
                final_classes_list.append(np.full(len(indices), cls, dtype=np.int32))
                final_mask_coeffs_list.append(cls_mask_coeffs[indices])
                
        if not final_boxes_list:
            return [[np.zeros((0, 5))], [np.zeros((0,), dtype=np.int32)], []]
            
        final_boxes = np.concatenate(final_boxes_list, axis=0)
        final_scores = np.concatenate(final_scores_list, axis=0)
        final_classes = np.concatenate(final_classes_list, axis=0)
        final_mask_coeffs = np.concatenate(final_mask_coeffs_list, axis=0)
        
    if len(final_boxes) == 0:
        return [[np.zeros((0, 5))], [np.zeros((0,), dtype=np.int32)], []]
    
    masks = process_mask(protos, final_mask_coeffs, final_boxes, shape=(input_size, input_size))
    
    gain = input_size / max(orig_shape)
    pad_x = (input_size - orig_shape[1] * gain) / 2
    pad_y = (input_size - orig_shape[0] * gain) / 2
    
    final_boxes[:, [0, 2]] -= pad_x
    final_boxes[:, [1, 3]] -= pad_y
    final_boxes /= gain
    final_boxes = np.clip(final_boxes, 0, [orig_shape[1], orig_shape[0], orig_shape[1], orig_shape[0]])
    
    polygons = []
    for i in range(len(masks)):
        mask = masks[i]
        py1, py2 = int(pad_y), int(input_size - pad_y)
        px1, px2 = int(pad_x), int(input_size - pad_x)
        if py2 > py1 and px2 > px1:
            mask = mask[py1:py2, px1:px2]
        mask = cv2.resize(mask.astype(np.uint8), (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Morphological close to bridge small gaps or thin lines in the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > 5:
                c_list = contour.reshape(-1, 2).tolist()
                if len(c_list) > 2:
                    valid_contours.append(c_list)
                    
        if len(valid_contours) > 0:
            # We must flatten it into a single list of coordinates, separated by the first point 
            # to form a valid multipolygon, or just use the largest contour for simplicity.
            # But the user's codebase only supports single polygons. 
            # If the mask is fragmented into multiple disconnected blobs, they will be returned.
            # But wait, save_shapefile_bb only expects ONE list. So we will just pick the largest.
            contour = max(contours, key=cv2.contourArea)
            contour = contour.reshape(-1, 2).tolist()
            if len(contour) > 2:
                polygons.append(contour)
            else:
                polygons.append(None)
        else:
            polygons.append(None)
            
    final_boxes = np.concatenate([final_boxes, final_scores[:, None]], axis=1)
    return [[final_boxes], [final_classes], [polygons]]
