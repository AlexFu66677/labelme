import cv2
import numpy as np


def _create_csrt_tracker():
    """Create a CSRT tracker across OpenCV/contrib API variants."""
    factory = getattr(cv2, "TrackerCSRT_create", None)
    if callable(factory):
        return factory()

    tracker_class = getattr(cv2, "TrackerCSRT", None)
    if tracker_class is not None and callable(getattr(tracker_class, "create", None)):
        return tracker_class.create()

    legacy = getattr(cv2, "legacy", None)
    if legacy is not None:
        factory = getattr(legacy, "TrackerCSRT_create", None)
        if callable(factory):
            return factory()
        tracker_class = getattr(legacy, "TrackerCSRT", None)
        if tracker_class is not None and callable(
            getattr(tracker_class, "create", None)
        ):
            return tracker_class.create()

    raise RuntimeError(
        "当前 OpenCV 环境未提供 CSRT，请安装与当前 Python 环境匹配的 "
        "opencv-contrib-python。"
    )


def track_boxes_csrt(previous_image, current_image, previous_boxes):
    """Track labeled boxes from one frame to the next with independent CSRTs.

    Args:
        previous_image: Previous BGR image as a numpy array.
        current_image: Current BGR image as a numpy array.
        previous_boxes: Items shaped as [label, [x1, y1], [x2, y2]].

    Returns:
        Successfully tracked boxes in the same format.
    """
    if previous_image is None or current_image is None:
        raise ValueError("上一帧和当前帧图像不能为空")

    image_height, image_width = current_image.shape[:2]
    tracked_boxes = []
    for box in previous_boxes:
        label, point1, point2 = box[:3]
        x1, x2 = sorted((float(point1[0]), float(point2[0])))
        y1, y2 = sorted((float(point1[1]), float(point2[1])))
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue

        tracker = _create_csrt_tracker()
        initial_box = tuple(
            int(round(value)) for value in (x1, y1, width, height)
        )
        if initial_box[2] <= 0 or initial_box[3] <= 0:
            continue
        initialized = tracker.init(previous_image, initial_box)
        # Some OpenCV Python bindings return None on successful initialization.
        if initialized is False:
            continue
        success, tracked = tracker.update(current_image)
        if not success:
            continue

        tracked_x, tracked_y, tracked_width, tracked_height = tracked
        new_x1 = max(0.0, min(float(tracked_x), image_width - 1.0))
        new_y1 = max(0.0, min(float(tracked_y), image_height - 1.0))
        new_x2 = max(
            0.0,
            min(float(tracked_x + tracked_width), image_width - 1.0),
        )
        new_y2 = max(
            0.0,
            min(float(tracked_y + tracked_height), image_height - 1.0),
        )
        if new_x1 < new_x2 and new_y1 < new_y2:
            tracked_boxes.append(
                [label, [new_x1, new_y1], [new_x2, new_y2]]
            )

    return tracked_boxes


def _box_iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _nms(candidates, iou_threshold):
    kept = []
    for candidate in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(_box_iou(candidate[0], item[0]) <= iou_threshold for item in kept):
            kept.append(candidate)
    return kept


def track_boxes_template_matching(
    previous_image,
    current_image,
    previous_boxes,
    score_threshold=0.8,
    nms_threshold=0.3,
):
    """Match previous-frame box crops in the current frame and apply NMS."""
    if previous_image is None or current_image is None:
        raise ValueError("上一帧和当前帧图像不能为空")
    previous_height, previous_width = previous_image.shape[:2]
    current_height, current_width = current_image.shape[:2]
    candidates_by_label = {}

    for box in previous_boxes:
        label, point1, point2 = box[:3]
        x1, x2 = sorted((int(round(point1[0])), int(round(point2[0]))))
        y1, y2 = sorted((int(round(point1[1])), int(round(point2[1]))))
        x1, x2 = max(0, x1), min(previous_width, x2)
        y1, y2 = max(0, y1), min(previous_height, y2)
        if x1 >= x2 or y1 >= y2:
            continue
        template = previous_image[y1:y2, x1:x2]
        template_height, template_width = template.shape[:2]
        if template_height > current_height or template_width > current_width:
            continue

        # CCOEFF is unreliable for a constant-color template. SQDIFF remains
        # meaningful there; convert it to a larger-is-better score.
        if float(np.std(template)) < 1e-6:
            response = cv2.matchTemplate(
                current_image, template, cv2.TM_SQDIFF_NORMED
            )
            response = 1.0 - response
        else:
            response = cv2.matchTemplate(
                current_image, template, cv2.TM_CCOEFF_NORMED
            )
        response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)

        # Keep local maxima only. Without this step every neighboring pixel
        # around one good match would become an almost identical box.
        local_max = cv2.dilate(response, np.ones((3, 3), dtype=np.uint8))
        ys, xs = np.where((response >= score_threshold) & (response == local_max))
        label_candidates = candidates_by_label.setdefault(str(label), [])
        for match_y, match_x in zip(ys.tolist(), xs.tolist()):
            matched_box = (
                float(match_x),
                float(match_y),
                float(match_x + template_width),
                float(match_y + template_height),
            )
            label_candidates.append(
                (matched_box, float(response[match_y, match_x]))
            )

    results = []
    for label, candidates in candidates_by_label.items():
        for box, _score in _nms(candidates, nms_threshold):
            results.append([label, [box[0], box[1]], [box[2], box[3]]])
    return results
