import importlib.util
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "labelme_yolo_io", Path(__file__).parents[1] / "labelme" / "yolo_io.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
load_yolo_shapes = _MODULE.load_yolo_shapes
save_yolo_shapes = _MODULE.save_yolo_shapes


def test_yolo_round_trip(tmpdir):
    path = Path(str(tmpdir.join("image.txt")))
    path.write_text("1 0.500000 0.500000 0.500000 0.250000\n", encoding="utf-8")
    shapes = load_yolo_shapes(str(path), 200, 100, ["cat", "dog"])
    assert shapes[0]["label"] == "dog"
    assert shapes[0]["points"] == [[50.0, 37.5], [150.0, 62.5]]
    output = Path(str(tmpdir.join("saved.txt")))
    save_yolo_shapes(str(output), shapes, 200, 100, ["cat", "dog"])
    assert output.read_text(encoding="utf-8") == "1 0.500000 0.500000 0.500000 0.250000\n"


def test_save_rejects_unknown_label(tmpdir):
    shape = {"label": "unknown", "points": [[0, 0], [10, 10]],
             "shape_type": "rectangle", "other_data": {}}
    with pytest.raises(ValueError):
        save_yolo_shapes(str(tmpdir.join("a.txt")), [shape], 100, 100, ["cat"])
