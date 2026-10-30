import os
import shutil
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.services.video_processing import extract_representative_frame
from backend.services.cinematography_model import predict_cinematography
from backend.services.optical_flow import analyze_camera_movement
from backend.services.scoring import evaluate_intent_alignment
from backend.services.rulebook import RULEBOOK, GENRE_TO_EMOTIONS
from backend.services.llm_recommendations import generate_llm_recommendations

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/api/options")
@router.get("/options")
async def get_genre_emotion_options():
    """
    Returns GENRE_TO_EMOTIONS mapping of valid genre and emotion dropdown choices.
    """
    return GENRE_TO_EMOTIONS

@router.post("/api/analyze")
@router.post("/analyze")
async def analyze_video_scene(
    video: UploadFile = File(...),
    genre: str = Form(...),
    emotion: str = Form(...)
):
    """
    Analyzes uploaded 5-10 second video clip:
    1. Validates that (genre, emotion) is a valid key pair in RULEBOOK and GENRE_TO_EMOTIONS.
    2. Extracts midpoint representative frame.
    3. Runs CLIP + 5 CineTechBench .joblib classifiers (with top-3 predict_proba scores).
    4. Runs optical-flow movement detection.
    5. Calculates Intent Alignment Score using existing rulebook scoring logic.
    6. Replaces rulebook template recommendations with configurable LLM recommendation engine.
    """
    # Validate genre and emotion pair
    if (genre, emotion) not in RULEBOOK or genre not in GENRE_TO_EMOTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid (genre, emotion) pair: ('{genre}', '{emotion}'). Not found in RULEBOOK."
        )
    if emotion not in GENRE_TO_EMOTIONS[genre]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid emotion '{emotion}' for genre '{genre}'. Valid emotions for {genre}: {GENRE_TO_EMOTIONS[genre]}"
        )

    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file uploaded.")

    temp_dir = tempfile.mkdtemp(prefix="cine_analysis_")
    temp_video_path = os.path.join(temp_dir, video.filename)
    temp_frame_path = os.path.join(temp_dir, "representative_frame.jpg")

    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # 1. Extract representative midpoint frame
        try:
            extracted_frame = extract_representative_frame(temp_video_path, temp_frame_path)
        except Exception as e:
            logger.error(f"Frame extraction error: {e}")
            raise HTTPException(status_code=422, detail=f"Failed to process video file: {str(e)}")

        # 2. Run CLIP + 5 CineTechBench Classifiers (calculates top-3 probability predictions)
        try:
            cinematography_preds = predict_cinematography(extracted_frame)
            top_predictions = cinematography_preds.pop("top_predictions", {})
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Cinematography model inference failed: {str(e)}")

        # 3. Evaluate Intent Alignment across the 5 CineTechBench visual classifiers (excluding camera movement)
        evaluation = evaluate_intent_alignment(
            detected_cinematography=cinematography_preds,
            genre=genre,
            emotion=emotion,
            include_movement=False
        )

        # Attach top-3 predictions to evaluation response
        evaluation["top_predictions"] = top_predictions
        evaluation["cinematography"]["top_predictions"] = top_predictions

        # 5. Generate LLM recommendations (replaces knowledge-base recommendations)
        try:
            llm_recommendations = generate_llm_recommendations(
                genre=genre,
                emotion=emotion,
                detected_cinematography=cinematography_preds,
                top_predictions=top_predictions,
                alignment_score=evaluation.get("score"),
                matches=evaluation.get("matches", []),
                mismatches=evaluation.get("mismatches", []),
                movement_result=None
            )
            evaluation["llm_recommendations"] = llm_recommendations

            # Format recommendations array from LLM suggestions for backward-compatible clients
            if llm_recommendations.get("suggestions"):
                evaluation["recommendations"] = [
                    f"[{s.get('priority', 'Medium')}] {s.get('change')}: {s.get('reason')}"
                    for s in llm_recommendations.get("suggestions", [])
                ]
            elif llm_recommendations.get("overall_explanation"):
                evaluation["recommendations"] = [llm_recommendations["overall_explanation"]]
        except Exception as e:
            logger.warning(f"LLM recommendation integration error: {e}")

        # 6. Attach Base64 Data URL of the exact extracted representative frame
        try:
            import base64
            with open(temp_frame_path, "rb") as img_f:
                b64_bytes = base64.b64encode(img_f.read()).decode("utf-8")
                evaluation["frame_data_url"] = f"data:image/jpeg;base64,{b64_bytes}"
        except Exception as e:
            logger.warning(f"Failed to encode frame data URL: {e}")

        return evaluation

    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")
