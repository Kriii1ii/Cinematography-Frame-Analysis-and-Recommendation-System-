import os
import sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.cinematography_model import load_cinematography_models, predict_cinematography
from backend.services.scoring import evaluate_intent_alignment
from backend.services.rulebook import GENRE_TO_EMOTIONS, compute_alignment_score

def test_full_backend_pipeline():
    print("==================================================")
    print("RUNNING BACKEND TEST SUITE FOR CINETECHBENCH EVALUATION")
    print("==================================================")

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    print("[1/4] Loading models, encoders, and CLIP...")
    load_cinematography_models(models_dir)
    print(" -> Models loaded successfully.")

    test_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_sample.jpg")
    img = Image.new("RGB", (320, 240), color=(120, 60, 40))
    img.save(test_img_path)
    
    try:
        print("[2/4] Running CLIP embedding & 5 classifier predictions on test image...")
        preds = predict_cinematography(test_img_path)
        print(" -> Predictions dictionary:", preds)
        
        required_attrs = ["scale", "angle", "composition", "lighting_quality", "lighting_intensity"]
        for attr in required_attrs:
            assert attr in preds, f"Missing prediction for {attr}"
            print(f"    - {attr.capitalize()}: {preds[attr]}")

        print("[3/4] Testing GENRE_TO_EMOTIONS & compute_alignment_score...")
        genre = "Psychological Thriller"
        emotion = "Anxiety"
        
        assert genre in GENRE_TO_EMOTIONS, f"Genre '{genre}' missing from GENRE_TO_EMOTIONS"
        assert emotion in GENRE_TO_EMOTIONS[genre], f"Emotion '{emotion}' missing for genre '{genre}'"

        preds["movement"] = "handheld"
        
        evaluation = evaluate_intent_alignment(preds, genre, emotion)
        print(" -> Intent Alignment Score:", evaluation["score"])
        print(" -> Matches count:", len(evaluation["matches"]))
        print(" -> Mismatches count:", len(evaluation["mismatches"]))
        print(" -> Recommendations:", evaluation["recommendations"])

        assert 0 <= evaluation["score"] <= 100, f"Score out of range: {evaluation['score']}"

        print("==================================================")
        print("SUCCESS: ALL BACKEND TESTS PASSED CLEANLY!")
        print("==================================================")

    finally:
        if os.path.exists(test_img_path):
            os.remove(test_img_path)

if __name__ == "__main__":
    test_full_backend_pipeline()
