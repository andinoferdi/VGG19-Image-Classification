import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Tuple
from config import *
from models import load_model


def get_prediction_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])


def predict_image(image_path: str, model_path: str = MODEL_SAVE_PATH) -> Dict:
    model = load_model(model_path)
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    transform = get_prediction_transform()
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item()
    
    all_probs = probabilities[0].cpu().numpy()
    class_confidences = {CLASS_NAMES[i]: float(all_probs[i]) for i in range(len(CLASS_NAMES))}
    
    top_3_indices = torch.topk(probabilities[0], k=min(3, NUM_CLASSES)).indices.cpu().numpy()
    top_3_predictions = [
        {
            'class': CLASS_NAMES[idx],
            'confidence': float(all_probs[idx])
        }
        for idx in top_3_indices
    ]
    
    result = {
        'predicted_class': predicted_class,
        'confidence': confidence_score,
        'all_confidences': class_confidences,
        'top_3': top_3_predictions
    }
    
    return result

