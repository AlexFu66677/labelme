from ultralytics.models.sam import SAM3SemanticPredictor

overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="D:/code/sam3.pt",
    half=True,
    iou=1.0,
    save=True,
    imgsz=1008,
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Set image once for multiple queries
predictor.set_image(r"C:\Users\fjl\Desktop\380c\vis\vis\seg/MOES_380C_0.jpg")

# Query with multiple text prompts
results = predictor(text=["road"])