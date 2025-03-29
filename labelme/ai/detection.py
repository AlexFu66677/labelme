import onnxruntime as ort
import numpy as np
import cv2

def nms(pred, conf_thres, iou_thres):
    conf = pred[..., 4] > conf_thres
    box = pred[conf == True]
    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))
    total_cls = list(set(cls))
    output_box = []
    for i in range(len(total_cls)):
        clss = total_cls[i]
        cls_box = []
        for j in range(len(cls)):
            if cls[j] == clss:
                box[j][5] = clss
                cls_box.append(box[j][:6])
        cls_box = np.array(cls_box)
        box_conf = cls_box[..., 4]
        box_conf_sort = np.argsort(box_conf)
        max_conf_box = cls_box[box_conf_sort[len(box_conf) - 1]]
        output_box.append(max_conf_box)
        cls_box = np.delete(cls_box, 0, 0)
        while len(cls_box) > 0:
            max_conf_box = output_box[len(output_box) - 1]
            del_index = []
            for j in range(len(cls_box)):
                current_box = cls_box[j]
                interArea = getInter(max_conf_box, current_box)
                iou = getIou(max_conf_box, current_box, interArea)
                if iou > iou_thres:
                    del_index.append(j)
            cls_box = np.delete(cls_box, del_index, 0)
            if len(cls_box) > 0:
                output_box.append(cls_box[0])
                cls_box = np.delete(cls_box, 0, 0)
    return output_box


def obb_nms(boxes, iou_thres):
    remove_flags = [False] * len(boxes)
    keep_boxes = []
    for i, ibox in enumerate(boxes):
        if remove_flags[i]:
            continue
        keep_boxes.append(ibox)
        for j in range(i + 1, len(boxes)):
            if remove_flags[j]:
                continue
            jbox = boxes[j]
            if (ibox[6] != jbox[6]):
                continue
            if probIou(ibox, jbox) > iou_thres:
                remove_flags[j] = True
    return keep_boxes


def xywhr2xyxyxyxy(center):
    cos, sin = (np.cos, np.sin)
    ctr = center[..., :2]
    w, h, angle = (center[..., i: i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1)
    vec2 = np.concatenate(vec2, axis=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2

    # return [pt1, pt2, pt3, pt4]
    return np.stack([pt1, pt2, pt3, pt4], axis=-2)


def getIou(box1, box2, inter_area):
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - inter_area
    iou = inter_area / union
    return iou


def covariance_matrix(obb):
    # Extract elements
    w, h, r = obb[2:5]
    a = (w ** 2) / 12
    b = (h ** 2) / 12

    # Calculate cosine and sine using NumPy
    cos_r = np.cos(r)
    sin_r = np.sin(r)

    # Calculate covariance matrix elements
    a_val = a * cos_r ** 2 + b * sin_r ** 2
    b_val = a * sin_r ** 2 + b * cos_r ** 2
    c_val = (a - b) * sin_r * cos_r

    return a_val, b_val, c_val


def probIou(obb1, obb2, eps=1e-7):
    a1, b1, c1 = covariance_matrix(obb1)
    a2, b2, c2 = covariance_matrix(obb2)
    x1, y1 = obb1[:2]
    x2, y2 = obb2[:2]
    # Calculate terms for Bhattacharyya distance
    t1 = ((a1 + a2) * ((y1 - y2) ** 2) + (b1 + b2) * ((x1 - x2) ** 2)) / \
         ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)

    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / \
         ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)
    t3 = np.log(((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2) / \
                (4 * np.sqrt(a1 * b1 - c1 ** 2) * np.sqrt(a2 * b2 - c2 ** 2) + eps) + eps)
    # Bhattacharyya distance calculation
    bd = 0.25 * t1 + 0.5 * t2 + 0.5 * t3
    hd = np.sqrt(1.0 - np.exp(-np.clip(bd, eps, 100.0)) + eps)

    # Extract the x, y coordinates of both
    return 1 - hd


def getInter(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, \
                                         box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[0] - box2[2] / 2, box2[1] - box1[3] / 2, \
                                         box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    if box1_x1 > box2_x2 or box1_x2 < box2_x1:
        return 0
    if box1_y1 > box2_y2 or box1_y2 < box2_y1:
        return 0
    x_list = [box1_x1, box1_x2, box2_x1, box2_x2]
    x_list = np.sort(x_list)
    x_inter = x_list[2] - x_list[1]
    y_list = [box1_y1, box1_y2, box2_y1, box2_y2]
    y_list = np.sort(y_list)
    y_inter = y_list[2] - y_list[1]
    inter = x_inter * y_inter
    return inter


def resize_with_padding(image, target_size, fill_color=114):
    original_size = image.shape[:2]  # 原始尺寸 (height, width)
    ratio = min(target_size[1] / original_size[1], target_size[0] / original_size[0])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))  # new_size (height, width)
    resized_image = cv2.resize(image, (new_size[1], new_size[0]))
    new_image = np.full((target_size[0], target_size[1], 3), fill_color, dtype=np.uint8)
    new_image[0:new_size[0], 0:new_size[1]] = resized_image
    return new_image, ratio


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1[0][0], box1[0][1], box1[1][0], box1[1][1]
    x3, y3, x4, y4 = box2[0][0], box2[0][1], box2[1][0], box2[1][1]

    # 计算交集的坐标
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    # 计算交集面积
    if x_right > x_left and y_bottom > y_top:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
    else:
        intersection_area = 0

    # 计算两个框的面积
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    # 计算 IoU
    iou = intersection_area / (box1_area + box2_area - intersection_area) if (
                                                                                     box1_area + box2_area - intersection_area) > 0 else 0
    return iou

def optical_flow(pre_image, cur_image):
    feature_params = dict(maxCorners=500, qualityLevel=0.1, minDistance=10, blockSize=10)
    prev_pts = cv2.goodFeaturesToTrack(pre_image, mask=None, **feature_params)
    lk_params = dict(winSize=(20, 20), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    cur_pts, status, err = cv2.calcOpticalFlowPyrLK(pre_image, cur_image, prev_pts, None, **lk_params)
    good_prev_pts = prev_pts[status == 1]
    good_cur_pts = cur_pts[status == 1]
    # 计算单应性矩阵
    H, mask = cv2.findHomography(good_prev_pts, good_cur_pts, cv2.RANSAC, 5.0)
    return H

def optical_flow_Homography(pre_image, cur_image):
    feature_params = dict(maxCorners=500, qualityLevel=0.1, minDistance=10, blockSize=10)
    prev_pts = cv2.goodFeaturesToTrack(pre_image, mask=None, **feature_params)
    lk_params = dict(winSize=(20, 20), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    cur_pts, status, err = cv2.calcOpticalFlowPyrLK(pre_image, cur_image, prev_pts, None, **lk_params)
    good_prev_pts = prev_pts[status == 1]
    good_cur_pts = cur_pts[status == 1]
    # 计算单应性矩阵
    H, mask = cv2.findHomography(good_prev_pts, good_cur_pts, cv2.RANSAC, 5.0)
    return H

def SIFT_Homography(pre_image, cur_image):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(pre_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(cur_image, None)
    # 创建FLANN匹配器
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 使用KNN匹配特征点
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    # 应用比率测试来选择良好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # 使用RANSAC算法计算单应性矩阵H
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

#图像预处理
def preprocess(image, input_tensors):
    for input_tensor in input_tensors:
        input_info = {
            "name": input_tensor.name,
            "type": input_tensor.type,
            "shape": input_tensor.shape,
        }
    img = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    image_height, image_width = img.shape[:2]
    img, ratio = resize_with_padding(img, (input_info["shape"][2], input_info["shape"][3]))
    img = img / 255
    img = img.astype(np.float32)
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    return blob,ratio, image_height, image_width

def detect_postprocess(pred, object_conf_thres,ratio,image_width,image_height):
    polygon = []
    pred = np.squeeze(pred)
    pred = np.transpose(pred, (1, 0))
    pred_class = pred[..., 4:]
    pred_conf = np.max(pred_class, axis=-1)
    pred = np.insert(pred, 4, pred_conf, axis=-1)
    result = nms(pred, object_conf_thres, 0.45)

    for detection in result:
        x_center, y_center, w, h, score, class_id = detection
        detect = [int((x_center - w / 2) / ratio), int((y_center - h / 2) / ratio),
                  int((x_center + w / 2) / ratio), int((y_center + h / 2) / ratio)]

        points0 = [detect[0], detect[1]]
        points1 = [detect[2], detect[3]]
        points0[0] = max(0, min(points0[0], image_width - 1))
        points0[1] = max(0, min(points0[1], image_height - 1))
        points1[0] = max(0, min(points1[0], image_width - 1))
        points1[1] = max(0, min(points1[1], image_height - 1))
        polygon.append([class_id, points0, points1])
    return polygon

def obb_detect_postprocess(pred, object_conf_thres,ratio,image_width,image_height):
    polygon = []
    pred = np.transpose(pred, (0, 2, 1))
    conf_thres = object_conf_thres
    iou_thres = 0.45
    boxes = []
    for item in pred[0]:
        cx, cy, w, h = item[:4]
        angle = item[-1]
        label = item[4:-1].argmax()
        confidence = item[4 + label]
        if confidence < conf_thres:
            continue
        boxes.append([cx, cy, w, h, angle, confidence, label])
    boxes = np.array(boxes)
    boxes = sorted(boxes.tolist(), key=lambda x: x[5], reverse=True)
    boxes = obb_nms(np.array(boxes), iou_thres)
    confs = [box[5] for box in boxes]
    classes = [int(box[6]) for box in boxes]
    if len(boxes) != 0:
        xyxy_boxes = xywhr2xyxyxyxy(np.array(boxes)[..., :5])
    else:
        xyxy_boxes = []
    for i, box in enumerate(xyxy_boxes):
        box = box / ratio
        box[:, 0] = np.clip(box[:, 0], 0, image_width - 1)  # 限制 x 坐标范围
        box[:, 1] = np.clip(box[:, 1], 0, image_height - 1)  # 限制 y 坐标范围
        polygon.append([classes[i], box[0], box[1], box[2], box[3], boxes[i][4]])
    return polygon

def obb_detect_postprocess_adjust(pred, object_conf_thres,ratio,image_width,image_height,img_r,img,image_show):
    polygon = []
    pred = np.transpose(pred, (0, 2, 1))
    conf_thres = object_conf_thres
    iou_thres = 0.45
    boxes = []
    for item in pred[0]:
        cx, cy, w, h = item[:4]
        angle = item[-1]
        label = item[4:-1].argmax()
        confidence = item[4 + label]
        if confidence < conf_thres:
            continue
        boxes.append([cx, cy, w, h, angle, confidence, label])
    boxes = np.array(boxes)
    boxes = sorted(boxes.tolist(), key=lambda x: x[5], reverse=True)
    boxes = obb_nms(np.array(boxes), iou_thres)
    confs = [box[5] for box in boxes]
    classes = [int(box[6]) for box in boxes]
    # if len(boxes) != 0:
    #     xyxy_boxes = xywhr2xyxyxyxy(np.array(boxes)[..., :5])
    # else:
    #     xyxy_boxes = []
    xyxy_boxes = []
    if len(boxes) != 0:
        for i, box in enumerate(boxes):
            cx, cy, w, h = box[0], box[1], box[2] + 2, box[3] + 2
            if min(box[2], box[3]) / max(box[2], box[3]) > 0.8:
                radius = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
                # 创建一个圆形掩码
                mask = np.zeros_like(img, dtype=np.uint8)
                cv2.circle(mask, (int(cx), int(cy)), int(radius), (255, 255, 255), -1)
                # 提取圆形区域
                circular_region = cv2.bitwise_and(img_r, mask)
                # 灰度转换和边缘检测
                gray = cv2.cvtColor(circular_region, cv2.COLOR_BGR2GRAY)
                lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
                lines = lsd.detect(gray)[0]  # 输入灰度图像
                # 筛选与长或宽最接近的直线
                best_line = None
                min_diff = float('inf')
                if lines is not None:
                    for idx, line in enumerate(lines):
                        x1, y1, x2, y2 = map(int, line[0])
                        cv2.line(img_r, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 绿色线，粗细2
                        center_dist = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / np.sqrt(
                            (y2 - y1) ** 2 + (x2 - x1) ** 2)
                        # expected_dist = (box[2] + box[3]) / 4
                        if abs(center_dist - box[2] / 2) > 5 and abs(center_dist - box[3] / 2) > 5:
                            continue
                        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        diff = abs(box[2] + box[3] - 2 * length)

                        if diff < min_diff:
                            min_diff = diff
                            best_line = (x1, y1, x2, y2)
                    if best_line is not None:
                        x1, y1, x2, y2 = best_line
                        cv2.line(img_r, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 绿色线，粗细2
                        dx, dy = x2 - x1, y2 - y1
                        A = dx ** 2 + dy ** 2
                        B = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
                        C = (x1 - cx) ** 2 + (y1 - cy) ** 2 - radius ** 2
                        D = B ** 2 - 4 * A * C
                        t1 = (-B + np.sqrt(D)) / (2 * A)
                        t2 = (-B - np.sqrt(D)) / (2 * A)
                        x1, y1, x2, y2 = x1 + t1 * dx, y1 + t1 * dy, x1 + t2 * dx, y1 + t2 * dy
                        sym_x1, sym_y1 = 2 * cx - x1, 2 * cy - y1
                        sym_x2, sym_y2 = 2 * cx - x2, 2 * cy - y2
                        points = [np.array((x1, y1)), np.array((x2, y2)), np.array((sym_x1, sym_y1)),
                                  np.array((sym_x2, sym_y2))]
                        xyxy_boxes.append(points)
                    else:
                        xyxy_boxes.append(xywhr2xyxyxyxy(np.array(box)[:5]))
                else:
                    xyxy_boxes.append(xywhr2xyxyxyxy(np.array(box)[:5]))
            else:
                xyxy_boxes.append(xywhr2xyxyxyxy(np.array(box)[:5]))
    for i, box in enumerate(xyxy_boxes):
        box = np.array(box) / ratio
        box[:, 0] = np.clip(box[:, 0], 0, image_width - 1)  # 限制 x 坐标范围
        box[:, 1] = np.clip(box[:, 1], 0, image_height - 1)  # 限制 y 坐标范围
        polygon.append([classes[i], box[0], box[1], box[2], box[3], boxes[i][4]])
    # 绘制多边形
    for poly in polygon:
        cls, p1, p2, p3, p4, angle = poly
        points = np.array([p1, p2, p3, p4], dtype=np.int32).reshape((-1, 1, 2))
        color = (255, 0, 0)  # 默认绿色
        cv2.polylines(image_show, [points], isClosed=True, color=color, thickness=2)
        for point in points:
            x, y = point[0]  # 提取坐标
            text = f"({int(x)}, {int(y)})"
            font_scale = 0.4  # 字体大小
            thickness = 1  # 字体粗细
            text_color = (255, 0, 0)  # 白色字体
            cv2.putText(image_show, text, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        text_color, thickness)
    # 保存标注后的图像
    image_show = cv2.cvtColor(image_show, cv2.COLOR_RGBA2BGR)
    cv2.imwrite("out.jpg", image_show)

def obb_detect_preprocess_adjust(image, input_tensors):
    for input_tensor in input_tensors:  # 因为可能有多个输入，所以为列表
        input_info = {
            "name": input_tensor.name,
            "type": input_tensor.type,
            "shape": input_tensor.shape,
        }
    img = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    image_show = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    image_height, image_width = img.shape[:2]
    img, ratio = resize_with_padding(img, (input_info["shape"][2], input_info["shape"][3]))
    img_r = img.copy()
    img = img / 255
    img = img.astype(np.float32)
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    return blob,ratio, image_height, image_width, img_r, img, image_show