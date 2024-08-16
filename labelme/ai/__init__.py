import gdown
import os.path as osp
import sys
from .grounddino import GroundingDINO

if getattr(sys, 'frozen', False):
    here = osp.dirname(sys.executable)
else:
    here = osp.dirname(osp.abspath(__file__))
from .efficient_sam import EfficientSam
from .segment_anything_model import SegmentAnythingModel
class EfficientSamVitT(EfficientSam):
    name = "EfficientSam (speed)"

    def __init__(self,device):
        super().__init__(
            encoder_path="./ai/seg_model/efficient_sam_vitt_encoder.onnx",
            decoder_path="./ai/seg_model/efficient_sam_vitt_decoder.onnx",
            device=device,
        )

class EfficientSamVitS(EfficientSam):
    name = "EfficientSam (accuracy)"

    def __init__(self,device):
        super().__init__(
            encoder_path="./ai/seg_model/efficient_sam_vits_encoder.onnx",
            decoder_path="./ai/seg_model/efficient_sam_vits_decoder.onnx",
            device=device,
        )

class SegmentAnythingModelVitB(SegmentAnythingModel):
    name = "SegmentAnything (speed)"

    def __init__(self,device):
        super().__init__(
            encoder_path="./ai/seg_model/sam_vit_b_01ec64.quantized.encoder.onnx",
            decoder_path="./ai/seg_model/sam_vit_b_01ec64.quantized.decoder.onnx",
            device=device,
        )


class SegmentAnythingModelVitL(SegmentAnythingModel):
    name = "SegmentAnything (balanced)"

    def __init__(self,device):
        super().__init__(
            encoder_path="./ai/seg_model/sam_vit_l_0b3195.quantized.encoder.onnx",
            decoder_path="./ai/seg_model/sam_vit_l_0b3195.quantized.decoder.onnx",  # NOQA
            device=device,
        )


class SegmentAnythingModelVitH(SegmentAnythingModel):
    name = "SegmentAnything (accuracy)"

    def __init__(self, device):
        super().__init__(
            encoder_path="./ai/seg_model/sam_vit_h_4b8939.quantized.encoder.onnx",  # NOQA
            decoder_path="./ai/seg_model/sam_vit_h_4b8939.quantized.decoder.onnx",  # NOQA
            device=device,
        )

class GroundDINO(SegmentAnythingModel):
    name = "GroundingDINO (accuracy)"
    config_path="",
    model_path="./ai/seg_model/GroundingDINO_SwinT_OGC.onnx"
class GroundDINO_INIT(SegmentAnythingModel):
    name = "NULL"
    config_path="",
    model_path="./ai/seg_model/GroundingDINO_SwinT_OGC.onnx"
MODELS = [
    SegmentAnythingModelVitB,
    SegmentAnythingModelVitL,
    SegmentAnythingModelVitH,
    EfficientSamVitT,
    EfficientSamVitS,
]
Text2LabelMODELS = [
    GroundDINO_INIT,
    GroundDINO,

]