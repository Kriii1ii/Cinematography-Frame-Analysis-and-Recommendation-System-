import os
import logging
import torch
import joblib
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Global model state (loaded once at startup)
_clip_model = None
_clip_processor = None
_classifiers = {}
_label_encoders = {}
_device = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_KEY_MAP = {
    "scale": ("Scale_final_model.joblib", "Scale_label_encoder.joblib"),
    "angle": ("Angle_vertical_final_model.joblib", "Angle_vertical_label_encoder.joblib"),
    "composition": ("Composition_final_model.joblib", "Composition_label_encoder.joblib"),
    "lighting_quality": ("Lighting_quality_final_model.joblib", "Lighting_quality_label_encoder.joblib"),
    "lighting_intensity": ("Lighting_intensity_final_model.joblib", "Lighting_intensity_label_encoder.joblib")
}

def load_cinematography_models(models_dir: str = None) -> None:
    """
    Loads CLIP model and the 5 CineTechBench .joblib classifiers + label encoders into memory once.
    """
    global _clip_model, _clip_processor, _classifiers, _label_encoders
    
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        
    logger.info("Initializing CLIP model: openai/clip-vit-base-patch32...")
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)
    _clip_model.eval()
    logger.info("CLIP model loaded successfully.")

    logger.info(f"Loading CineTechBench classifier models from {models_dir}...")
    for attr, (model_file, encoder_file) in TARGET_KEY_MAP.items():
        m_path = os.path.join(models_dir, model_file)
        e_path = os.path.join(models_dir, encoder_file)
        
        if not os.path.exists(m_path) or not os.path.exists(e_path):
            raise FileNotFoundError(f"Required model or encoder file missing: {m_path} / {e_path}")
            
        _classifiers[attr] = joblib.load(m_path)
        _label_encoders[attr] = joblib.load(e_path)
        logger.info(f"Loaded classifier & encoder for '{attr}'")

def predict_cinematography(image_path: str) -> Dict[str, str]:
    """
    Given an image path, extracts 512-dim CLIP embedding and runs prediction through all 5 CineTechBench classifiers.
    Returns decoded human-readable string predictions.
    """
    global _clip_model, _clip_processor, _classifiers, _label_encoders
    
    if _clip_model is None or not _classifiers:
        # Auto-load if not previously loaded
        load_cinematography_models()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Representative frame image not found at {image_path}")

    # 1. Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = _clip_processor(images=image, return_tensors="pt").to(_device)

    # 2. Extract 512-dimensional CLIP embedding
    with torch.no_grad():
        out = _clip_model.get_image_features(**inputs)
        if isinstance(out, torch.Tensor):
            image_features = out
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            image_features = out.pooler_output
        elif hasattr(out, "image_embeds") and out.image_embeds is not None:
            image_features = out.image_embeds
        else:
            image_features = out[0]

        # Normalize CLIP embedding
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        embedding = image_features.cpu().numpy()

    # Ensure shape is (1, 512)
    if len(embedding.shape) == 1:
        embedding = embedding.reshape(1, -1)

    # 3. Predict each attribute with primary label and top-three probability rankings
    predictions = {}
    top_predictions = {}

    for attr in TARGET_KEY_MAP.keys():
        model = _classifiers[attr]
        encoder = _label_encoders[attr]
        
        pred_idx = model.predict(embedding)
        label = encoder.inverse_transform(pred_idx)[0]
        primary_label = str(label)
        predictions[attr] = primary_label

        # Top-3 predictions using predict_proba()
        if hasattr(model, "predict_proba"):
            try:
                # Calculate estimated class probabilities on the same 512-dim CLIP embedding
                probs = model.predict_proba(embedding)[0]

                # Map each probability to its corresponding class label using model.classes_
                # and encoder.inverse_transform, verifying the exact mapping
                ranked_candidates = []
                for class_index, prob in zip(model.classes_, probs):
                    class_label = str(encoder.inverse_transform([class_index])[0])
                    p_float = float(prob)
                    ranked_candidates.append({
                        "label": class_label,
                        "probability": round(p_float, 4),
                        "percentage": round(p_float * 100.0, 1),
                        "score_type": "estimated_confidence"
                    })

                # Sort predictions by estimated probability in descending order
                ranked_candidates.sort(key=lambda x: x["probability"], reverse=True)

                # Preserve current primary prediction as highest-ranked prediction
                if ranked_candidates and ranked_candidates[0]["label"] != primary_label:
                    primary_pos = next(
                        (i for i, item in enumerate(ranked_candidates) if item["label"] == primary_label),
                        None
                    )
                    if primary_pos is not None:
                        primary_item = ranked_candidates.pop(primary_pos)
                        ranked_candidates.insert(0, primary_item)

                # Return top three, or fewer if classifier has fewer than three classes
                top_k = ranked_candidates[:min(3, len(ranked_candidates))]
                top_predictions[attr] = top_k
            except Exception as e:
                logger.error(f"Error computing probabilities for {attr}: {e}")
                top_predictions[attr] = [{
                    "label": primary_label,
                    "probability": 1.0,
                    "percentage": 100.0,
                    "score_type": "fallback"
                }]
        else:
            # Model does not support predict_proba() - report limitation without fabricating scores
            top_predictions[attr] = [{
                "label": primary_label,
                "probability": None,
                "percentage": None,
                "unsupported": True,
                "message": "Classifier does not support predict_proba()"
            }]

    predictions["top_predictions"] = top_predictions
    return predictions
