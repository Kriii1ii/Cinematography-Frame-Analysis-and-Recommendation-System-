import os
import json
import logging
import httpx
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20.0"))

def get_llm_config() -> Dict[str, Any]:
    """
    Resolves LLM provider, API key, model, and base URL from environment variables.
    Supported providers: 'gemini', 'openai', 'anthropic', 'openrouter', 'custom'.
    """
    provider = os.getenv("LLM_PROVIDER", "").lower().strip()
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "")
    base_url = os.getenv("LLM_BASE_URL", "")

    # Auto-detect provider from available environment variables if not specified
    if not provider:
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            provider = "unconfigured"

    # Resolve provider-specific keys and default models
    if provider == "gemini":
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        model = model or "gemini-3.5-flash-lite"
    elif provider == "openai":
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        model = model or "gpt-4o-mini"
        base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    elif provider == "anthropic":
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        model = model or "claude-3-5-haiku-20241022"
    elif provider == "openrouter":
        api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        model = model or "google/gemini-2.5-flash"
        base_url = base_url or "https://openrouter.ai/api/v1"

    return {
        "provider": provider,
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
        "timeout": DEFAULT_TIMEOUT
    }

def detect_uncertain_features(top_predictions: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """
    Identifies features with close alternative predictions (e.g. delta <= 12% between rank 1 and rank 2).
    """
    uncertainties = []
    for attr, ranked in top_predictions.items():
        if isinstance(ranked, list) and len(ranked) >= 2:
            p1 = ranked[0].get("probability")
            p2 = ranked[1].get("probability")
            if p1 is not None and p2 is not None:
                delta = abs(p1 - p2)
                if delta <= 0.12 or p1 < 0.40:
                    lbl1 = ranked[0].get("label")
                    lbl2 = ranked[1].get("label")
                    pct1 = ranked[0].get("percentage", round(p1 * 100, 1))
                    pct2 = ranked[1].get("percentage", round(p2 * 100, 1))
                    uncertainties.append(
                        f"Classifier confidence for '{attr.replace('_', ' ').capitalize()}' is contested between "
                        f"{lbl1} ({pct1}%) and {lbl2} ({pct2}%)."
                    )
    return uncertainties

def build_fallback_recommendations(
    genre: str,
    emotion: str,
    detected_cinematography: Dict[str, Any],
    top_predictions: Dict[str, Any],
    matches: List[Dict[str, Any]],
    mismatches: List[Dict[str, Any]],
    error_note: str = ""
) -> Dict[str, Any]:
    """
    Constructs a valid, structured fallback recommendation object when the LLM
    provider is unconfigured, times out, or fails.
    """
    uncertainties = detect_uncertain_features(top_predictions)

    strengths = []
    if matches:
        for m in matches:
            attr_name = m.get("attribute", "").replace("_", " ").title()
            det = m.get("detected")
            strengths.append(f"{attr_name} ({det}) supports standard genre conventions for {genre}.")
    else:
        strengths.append(f"Primary detected visual dimensions establish a coherent baseline for {genre}.")

    suggestions = []
    if mismatches:
        for idx, mm in enumerate(mismatches):
            attr_name = mm.get("attribute", "").replace("_", " ").title()
            det = mm.get("detected")
            exp = mm.get("expected")
            priority = "High" if idx == 0 else ("Medium" if idx == 1 else "Low")
            suggestions.append({
                "change": f"Consider testing {exp} instead of {det} for {attr_name}",
                "reason": f"In {genre} scenes aiming for {emotion}, {exp} typically enhances emotional resonance.",
                "priority": priority
            })
    else:
        suggestions.append({
            "change": "Refine camera blocking or lighting nuances while preserving primary framing",
            "reason": f"Current composition and lighting already align with {genre} expectations for {emotion}.",
            "priority": "Low"
        })

    return {
        "status": "fallback",
        "provider": "rulebook_fallback",
        "error_message": error_note or "LLM provider is not configured or unavailable.",
        "overall_explanation": (
            f"The analyzed scene exhibits visual traits that reflect the aesthetic baseline of {genre} "
            f"with intended {emotion}. (Fallback editorial guidance active)."
        ),
        "strengths": strengths,
        "suggestions": suggestions,
        "uncertainty_notes": uncertainties or [
            "Classifier predictions are relatively distinct across primary feature dimensions."
        ],
        "creative_options": [
            f"Experiment with contrast ratios and character eye-lines to emphasize {emotion.lower()}."
        ],
        "disclaimer": "Generated via rule-based fallback. Configure an LLM provider (e.g. GEMINI_API_KEY or OPENAI_API_KEY) for dynamic AI consultation."
    }

def generate_llm_recommendations(
    genre: str,
    emotion: str,
    detected_cinematography: Dict[str, Any],
    top_predictions: Dict[str, Any],
    alignment_score: Optional[int] = None,
    matches: Optional[List[Dict[str, Any]]] = None,
    mismatches: Optional[List[Dict[str, Any]]] = None,
    movement_result: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Calls a configured LLM provider to generate structured cinematography recommendations.
    Never raises an uncaught exception; returns structured fallback JSON on any failure.
    """
    matches = matches or []
    mismatches = mismatches or []
    cfg = get_llm_config()

    if not cfg["api_key"] or cfg["provider"] == "unconfigured":
        logger.info("LLM recommendation service: No API key configured. Using structured fallback.")
        return build_fallback_recommendations(
            genre, emotion, detected_cinematography, top_predictions, matches, mismatches,
            error_note="No LLM API key configured (set GEMINI_API_KEY or OPENAI_API_KEY)."
        )

    # 1. Prepare structured input for the LLM
    uncertainties = detect_uncertain_features(top_predictions)
    movement_val = movement_result.get("movement") if isinstance(movement_result, dict) else (movement_result or detected_cinematography.get("movement", "locked"))

    structured_input = {
        "genre": genre,
        "intended_emotion": emotion,
        "primary_classifier_predictions": {
            "scale": detected_cinematography.get("scale"),
            "angle": detected_cinematography.get("angle"),
            "composition": detected_cinematography.get("composition"),
            "lighting_quality": detected_cinematography.get("lighting_quality"),
            "lighting_intensity": detected_cinematography.get("lighting_intensity"),
            "movement": movement_val
        },
        "alternative_predictions_with_estimated_scores": top_predictions,
        "intent_alignment_score": alignment_score,
        "existing_matches": [f"{m.get('attribute')}: {m.get('detected')}" for m in matches],
        "existing_mismatches": [f"{m.get('attribute')}: detected {m.get('detected')}, rulebook expected {m.get('expected')}" for m in mismatches],
        "contested_or_uncertain_features": uncertainties
    }

    system_prompt = (
        "You are an expert film cinematography consultant and directorial advisor. "
        "Your task is to analyze the provided structured visual attributes detected in a film frame "
        "against the director's intended genre and emotion, and produce actionable, professional creative recommendations.\n\n"
        "CRITICAL CONSTRAINTS:\n"
        "1. Do NOT invent new classifier predictions or modify the detected classifier outputs.\n"
        "2. Do NOT fabricate numerical scores or claim suggestions are 'objectively correct'.\n"
        "3. Emphasize creative options, lighting nuance, camera movement, and framing that support the intended mood.\n"
        "4. You must output strictly valid JSON conforming to the requested schema with no extra conversational text or markdown ticks outside the JSON.\n\n"
        "REQUIRED JSON OUTPUT SCHEMA:\n"
        "{\n"
        '  "overall_explanation": "Concise 1-2 sentence assessment of how detected cinematography relates to the intended genre and emotion.",\n'
        '  "strengths": ["List of existing cinematography strengths worth preserving..."],\n'
        '  "suggestions": [\n'
        '    {\n'
        '      "change": "Specific, practical cinematography change to consider (e.g. shift from Hard to Soft Light)",\n'
        '      "reason": "Artistic and technical reason for this change in the context of the genre and emotion.",\n'
        '      "priority": "High" | "Medium" | "Low"\n'
        '    }\n'
        '  ],\n'
        '  "uncertainty_notes": ["Note on features where classifier predictions are close/contested, or note that predictions are clear."],\n'
        '  "creative_options": ["1-2 additional stylistic or lighting alternatives for the director to consider."]\n'
        "}"
    )

    user_prompt = f"Analyze this scene and provide recommendations in the required JSON format:\n\n{json.dumps(structured_input, indent=2)}"

    try:
        response_text = ""
        provider = cfg["provider"]

        if provider == "gemini":
            response_text = _call_gemini_api(cfg["api_key"], cfg["model"], system_prompt, user_prompt, cfg["timeout"])
        elif provider in ["openai", "openrouter", "custom"]:
            response_text = _call_openai_compatible_api(
                cfg["api_key"], cfg["model"], cfg["base_url"], system_prompt, user_prompt, cfg["timeout"]
            )
        elif provider == "anthropic":
            response_text = _call_anthropic_api(cfg["api_key"], cfg["model"], system_prompt, user_prompt, cfg["timeout"])
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        # Clean potential markdown wrapping
        clean_json = response_text.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        elif clean_json.startswith("```"):
            clean_json = clean_json[3:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
        clean_json = clean_json.strip()

        parsed = json.loads(clean_json)

        # Validate required fields
        if not isinstance(parsed, dict) or "overall_explanation" not in parsed or "suggestions" not in parsed:
            raise ValueError("Response JSON missing required keys ('overall_explanation' or 'suggestions')")

        parsed["status"] = "success"
        parsed["provider"] = f"{provider} ({cfg['model']})"
        parsed["disclaimer"] = "AI-generated creative cinematography recommendations for editorial guidance only."
        return parsed

    except httpx.TimeoutException as te:
        logger.warning(f"LLM recommendation call timed out after {cfg['timeout']}s: {te}")
        return build_fallback_recommendations(
            genre, emotion, detected_cinematography, top_predictions, matches, mismatches,
            error_note=f"LLM request timed out ({cfg['timeout']}s)."
        )
    except httpx.HTTPStatusError as hse:
        logger.warning(f"LLM provider HTTP error {hse.response.status_code}: {hse}")
        return build_fallback_recommendations(
            genre, emotion, detected_cinematography, top_predictions, matches, mismatches,
            error_note=f"LLM provider returned status {hse.response.status_code} ({hse.response.text[:100]})."
        )
    except Exception as e:
        logger.warning(f"LLM recommendation generation failed: {e}. Using fallback.")
        return build_fallback_recommendations(
            genre, emotion, detected_cinematography, top_predictions, matches, mismatches,
            error_note=f"LLM generation failed: {str(e)}"
        )


def _call_gemini_api(api_key: str, model: str, system_prompt: str, user_prompt: str, timeout: float) -> str:
    """Calls Google Gemini REST API with automatic fallback across flash models on 503/429 spikes."""
    candidate_models = [model]
    for fallback_m in ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash"]:
        if fallback_m not in candidate_models:
            candidate_models.append(fallback_m)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.3
        }
    }

    last_error = None
    with httpx.Client(timeout=timeout) as client:
        for m in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
            try:
                resp = client.post(url, json=payload)
                if resp.status_code in [503, 429]:
                    last_error = httpx.HTTPStatusError(f"{resp.status_code} on {m}", request=resp.request, response=resp)
                    continue
                resp.raise_for_status()
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise ValueError(f"No response candidate returned by Gemini API ({m})")
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise ValueError(f"Empty response parts from Gemini API ({m})")
                text_content = "".join([p.get("text", "") for p in parts if "text" in p]).strip()
                if not text_content:
                    raise ValueError(f"No text part in Gemini API response ({m})")
                return text_content
            except httpx.HTTPStatusError as hse:
                last_error = hse
                if hse.response.status_code in [503, 429]:
                    continue
                raise
            except Exception as e:
                last_error = e
                continue

    if last_error:
        raise last_error
    raise RuntimeError("All Gemini candidate models failed to respond")


def _call_openai_compatible_api(api_key: str, model: str, base_url: str, system_prompt: str, user_prompt: str, timeout: float) -> str:
    """Calls OpenAI-compatible /chat/completions endpoint."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No response choices returned by OpenAI API")
        return choices[0].get("message", {}).get("content", "")


def _call_anthropic_api(api_key: str, model: str, system_prompt: str, user_prompt: str, timeout: float) -> str:
    """Calls Anthropic Messages API."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", [])
        if not content:
            raise ValueError("No content returned by Anthropic API")
        return content[0].get("text", "")
