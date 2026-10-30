from typing import Dict, Any
from backend.services.rulebook import compute_alignment_score, GENRE_TO_EMOTIONS

def evaluate_intent_alignment(
    detected_cinematography: Dict[str, str],
    genre: str,
    emotion: str,
    include_movement: bool = False
) -> Dict[str, Any]:
    """
    Adapter function mapping detected predictions dictionary to compute_alignment_score.
    When include_movement is False, scoring evaluates purely across the 5 image-based classifiers.
    """
    # Key mapping to Title case expected by compute_alignment_score
    mapped_detected = {
        "Scale": detected_cinematography.get("scale", detected_cinematography.get("Scale", "medium")),
        "Angle": detected_cinematography.get("angle", detected_cinematography.get("Angle", "eye-level")),
        "Composition": detected_cinematography.get("composition", detected_cinematography.get("Composition", "Rule of Thirds")),
        "Lighting_quality": detected_cinematography.get("lighting_quality", detected_cinematography.get("Lighting_quality", "Soft Light")),
        "Lighting_intensity": detected_cinematography.get("lighting_intensity", detected_cinematography.get("Lighting_intensity", "High Key")),
    }
    if include_movement and ("movement" in detected_cinematography or "Movement" in detected_cinematography):
        mapped_detected["Movement"] = detected_cinematography.get("movement", detected_cinematography.get("Movement", "locked"))

    raw_result = compute_alignment_score(mapped_detected, genre, emotion, include_movement=include_movement)
    
    matches = []
    mismatches = []
    
    for item in raw_result.get("breakdown", []):
        entry = {
            "attribute": item["attribute"].lower(),
            "detected": item["detected"],
            "expected": " or ".join(item["expected"]),
            "explanation": f"Detected '{item['detected']}' for {item['attribute']} (Matches expected: {item['match']})."
        }
        if item["match"]:
            matches.append(entry)
        else:
            mismatches.append(entry)

    return {
        "score": int(round(raw_result["score"])),
        "genre": genre,
        "emotion": emotion,
        "cinematography": detected_cinematography,
        "matches": matches,
        "mismatches": mismatches,
        "recommendations": raw_result["recommendations"],
        "breakdown": raw_result["breakdown"]
    }
