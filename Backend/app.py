from pathlib import Path
import io
import json

import torch
import timm
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "label_mapping.json", "r", encoding="utf-8") as f:
    id_to_label = json.load(f)

id_to_label = {int(k): v for k, v in id_to_label.items()}

eval_metrics = {"overall": {}, "classes": {}}
metrics_path = BASE_DIR / "evaluation_metrics.json"
if metrics_path.exists():
    with open(metrics_path, "r", encoding="utf-8") as f:
        eval_metrics = json.load(f)

model = timm.create_model(
    "efficientnet_b2",
    pretrained=False,
    num_classes=len(id_to_label)
)

model_file = BASE_DIR / "skin_model_best.pth"
if not model_file.exists():
    model_file = BASE_DIR / "skin_model.pth"

state_dict = torch.load(
    model_file,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "metrics_loaded": metrics_path.exists()
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0)
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to read the uploaded image.")

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits[0], dim=0)
        confidence, prediction = probabilities.max(dim=0)
        top3_conf, top3_idx = torch.topk(probabilities, k=min(3, len(id_to_label)))

    class_id = prediction.item()
    class_metrics = eval_metrics.get("classes", {}).get(str(class_id), {})
    overall_metrics = eval_metrics.get("overall", {})

    top_predictions = [
        {
            "label": id_to_label[idx.item()],
            "confidence": round(conf.item() * 100, 2)
        }
        for conf, idx in zip(top3_conf, top3_idx)
    ]

    return {
        "prediction": id_to_label[class_id],
        "class_id": class_id,
        "confidence": round(confidence.item() * 100, 2),
        "top_predictions": top_predictions,
        "overall_metrics": overall_metrics,
        "class_metrics": class_metrics
    }