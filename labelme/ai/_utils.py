import imgviz
import cv2
import numpy as np

from labelme.logger import logger


def compute_polygon_from_mask(mask):
    mask = np.asarray(mask)
    binary = np.ascontiguousarray((mask > 0).astype(np.uint8))
    contours, _hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) == 0:
        logger.warning("No contour found, so returning empty polygon.")
        return np.empty((0, 2), dtype=np.float32)

    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    POLYGON_APPROX_TOLERANCE = 0.004
    polygon = cv2.approxPolyDP(
        contour, perimeter * POLYGON_APPROX_TOLERANCE, True
    )
    polygon = polygon.reshape(-1, 2).astype(np.float32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, mask.shape[1] - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, mask.shape[0] - 1)

    if 0:
        import PIL.Image

        image_pil = PIL.Image.fromarray(imgviz.gray2rgb(imgviz.bool2ubyte(mask)))
        imgviz.draw.line_(image_pil, yx=polygon, fill=(0, 255, 0))
        for point in polygon:
            imgviz.draw.circle_(image_pil, center=point, diameter=10, fill=(0, 255, 0))
        imgviz.io.imsave("contour.jpg", np.asarray(image_pil))

    return polygon


def compute_polygons_from_mask(mask):
    """Convert every connected foreground region in a mask to a polygon."""
    mask = np.asarray(mask)
    binary = np.ascontiguousarray((mask > 0).astype(np.uint8))
    component_count, components, stats, _centroids = (
        cv2.connectedComponentsWithStats(binary, connectivity=8)
    )
    polygons = []
    for component_id in range(1, component_count):
        component = np.ascontiguousarray(
            (components == component_id).astype(np.uint8)
        )
        polygon = compute_polygon_from_mask(component)
        if len(polygon) >= 3:
            polygons.append(polygon)
            continue

        # Very small or thin components may collapse during polygon
        # approximation. Preserve them with a minimal rectangular polygon.
        x, y, width, height = stats[component_id, :4]
        x0, x1 = float(x), float(x + width - 1)
        y0, y1 = float(y), float(y + height - 1)
        if x0 == x1:
            x1 = min(float(mask.shape[1] - 1), x1 + 1.0)
            x0 = max(0.0, x0 - 1.0) if x0 == x1 else x0
        if y0 == y1:
            y1 = min(float(mask.shape[0] - 1), y1 + 1.0)
            y0 = max(0.0, y0 - 1.0) if y0 == y1 else y0
        polygons.append(
            np.asarray(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                dtype=np.float32,
            )
        )
    return polygons
