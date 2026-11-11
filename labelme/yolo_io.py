import os


def load_yolo_shapes(filename, image_width, image_height, labels=None):
    """Load YOLO detection annotations as labelme rectangle dictionaries."""
    labels = list(labels or [])
    shapes = []
    with open(filename, "r", encoding="utf-8") as stream:
        for raw_line in stream:
            fields = raw_line.strip().split()
            if not fields:
                continue
            if len(fields) != 5:
                continue
            try:
                class_id = int(fields[0])
                center_x, center_y, width, height = map(float, fields[1:])
            except ValueError:
                continue
            if class_id < 0 or width < 0 or height < 0:
                continue
            x1 = (center_x - width / 2.0) * image_width
            y1 = (center_y - height / 2.0) * image_height
            x2 = (center_x + width / 2.0) * image_width
            y2 = (center_y + height / 2.0) * image_height
            label = labels[class_id] if class_id < len(labels) else str(class_id)
            shapes.append({
                "label": label, "points": [[x1, y1], [x2, y2]],
                "group_id": None, "description": "", "shape_type": "rectangle",
                "flags": {}, "mask": None,
                "other_data": {"yolo_class_id": class_id},
            })
    return shapes


def save_yolo_shapes(filename, shapes, image_width, image_height, labels=None):
    """Save labelme rectangles in YOLO detection format."""
    labels = list(labels or [])
    label_ids = {label: index for index, label in enumerate(labels)}
    lines = []
    for shape in shapes:
        if shape.get("shape_type") != "rectangle" or len(shape.get("points", [])) < 2:
            continue
        class_id = shape.get("other_data", {}).get("yolo_class_id")
        if class_id is None:
            class_id = label_ids.get(shape.get("label"))
        if class_id is None:
            try:
                class_id = int(shape.get("label"))
            except (TypeError, ValueError):
                raise ValueError(
                    "Label {!r} is not present in the configured labels".format(shape.get("label"))
                )
        (x1, y1), (x2, y2) = shape["points"][:2]
        clipped_x = [min(float(image_width), max(0.0, value)) for value in (x1, x2)]
        clipped_y = [min(float(image_height), max(0.0, value)) for value in (y1, y2)]
        left, right = sorted(clipped_x)
        top, bottom = sorted(clipped_y)
        values = (
            (left + right) / 2.0 / image_width,
            (top + bottom) / 2.0 / image_height,
            (right - left) / image_width,
            (bottom - top) / image_height,
        )
        lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(class_id, *values))
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
        if lines:
            stream.write("\n")
