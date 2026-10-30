# RULEBOOK + SCORING
# Hand-authored genre+emotion -> expected cinematography profile, plus the
# weighted comparison that produces the Intent Alignment Score.
#
# IMPORTANT: "expected" values use the REAL label vocabulary each classifier
# actually outputs, not simplified names:
#   Scale, Angle, Movement       -> trained on AVE   -> lowercase, hyphenated
#                                    (e.g. "close-up", "low-angle", "handheld")
#   Composition, Lighting_*      -> trained on CineTechBench -> Title Case
#                                    (e.g. "Rule of Thirds", "Hard Light")

ATTRS = ["Scale","Angle","Composition","Lighting_quality","Lighting_intensity","Movement"]
def profile(*pairs):   # each pair = ({expected labels}, weight), in ATTRS order
    return [{"attribute": a, "expected": e, "weight": w} for a, (e, w) in zip(ATTRS, pairs)]

RULEBOOK = {
    ("Psychological Thriller", "Anxiety"): [
        {"attribute": "Scale",              "expected": {"close-up"},                     "weight": 20},
        {"attribute": "Angle",              "expected": {"low-angle", "high-angle"},      "weight": 10},
        {"attribute": "Composition",        "expected": {"Rule of Thirds"},               "weight": 15},
        {"attribute": "Lighting_quality",   "expected": {"Hard Light"},                   "weight": 10},
        {"attribute": "Lighting_intensity", "expected": {"Low Key"},                      "weight": 25},
        {"attribute": "Movement",           "expected": {"handheld"},                     "weight": 20},
    ],
    ("Horror", "Fear"): [
        {"attribute": "Scale",              "expected": {"close-up", "extreme-close-up"}, "weight": 15},
        {"attribute": "Angle",              "expected": {"low-angle"},                    "weight": 15},
        {"attribute": "Composition",        "expected": {"Framing"},                      "weight": 15},
        {"attribute": "Lighting_quality",   "expected": {"Hard Light"},                   "weight": 15},
        {"attribute": "Lighting_intensity", "expected": {"Low Key"},                      "weight": 30},
        {"attribute": "Movement",           "expected": {"handheld"},                     "weight": 10},
    ],
    ("Romance", "Warmth"): [
        {"attribute": "Scale",              "expected": {"medium"},                       "weight": 20},
        {"attribute": "Angle",              "expected": {"eye-level"},                    "weight": 15},
        {"attribute": "Composition",        "expected": {"Symmetrical", "Central"},       "weight": 15},
        {"attribute": "Lighting_quality",   "expected": {"Soft Light"},                   "weight": 25},
        {"attribute": "Lighting_intensity", "expected": {"High Key"},                     "weight": 15},
        {"attribute": "Movement",           "expected": {"locked", "pan"},                "weight": 10},
    ],
    ("Action", "Excitement"): [
        {"attribute": "Scale",              "expected": {"wide", "medium"},               "weight": 15},
        {"attribute": "Angle",              "expected": {"low-angle"},                    "weight": 15},
        {"attribute": "Composition",        "expected": {"Rule of Thirds"},               "weight": 15},
        {"attribute": "Lighting_quality",   "expected": {"Hard Light"},                   "weight": 10},
        {"attribute": "Lighting_intensity", "expected": {"High Key"},                     "weight": 10},
        {"attribute": "Movement",           "expected": {"handheld", "zoom"},             "weight": 35},
    ],
    ("Drama", "Sadness"): [
        {"attribute": "Scale",              "expected": {"close-up"},                     "weight": 25},
        {"attribute": "Angle",              "expected": {"eye-level"},                    "weight": 10},
        {"attribute": "Composition",        "expected": {"Central"},                      "weight": 15},
        {"attribute": "Lighting_quality",   "expected": {"Soft Light"},                   "weight": 15},
        {"attribute": "Lighting_intensity", "expected": {"Low Key"},                      "weight": 20},
        {"attribute": "Movement",           "expected": {"locked"},                       "weight": 15},
    ],
    ("Comedy", "Joy"): [
        {"attribute": "Scale",              "expected": {"medium"},                       "weight": 15},
        {"attribute": "Angle",              "expected": {"eye-level"},                    "weight": 15},
        {"attribute": "Composition",        "expected": {"Rule of Thirds"},               "weight": 15},
        {"attribute": "Lighting_quality",   "expected": {"Soft Light"},                   "weight": 15},
        {"attribute": "Lighting_intensity", "expected": {"High Key"},                     "weight": 25},
        {"attribute": "Movement",           "expected": {"locked", "pan"},                "weight": 15},
    ],
}

RULEBOOK.update({
  ("Romantic Comedy","Happiness"):       profile(({"medium"},15),({"eye-level"},15),({"Rule of Thirds"},15),({"Soft Light"},20),({"High Key"},25),({"locked","pan"},10)),
  ("Romantic Comedy","Vulnerability"):   profile(({"close-up"},25),({"eye-level","high-angle"},15),({"Central","Framing"},10),({"Soft Light"},25),({"Low Key"},10),({"locked"},15)),
  ("Romance","Vulnerability"):           profile(({"close-up"},25),({"eye-level","high-angle"},10),({"Central"},15),({"Soft Light"},25),({"Low Key"},15),({"locked"},10)),
  ("Psychological Thriller","Paranoia"): profile(({"close-up","extreme-close-up"},15),({"high-angle"},15),({"Framing"},15),({"Hard Light"},15),({"Low Key"},20),({"handheld"},20)),
  ("Horror","Dread"):                    profile(({"wide","extreme-wide"},20),({"high-angle"},10),({"Symmetrical","Framing"},15),({"Hard Light"},10),({"Low Key"},30),({"locked"},15)),
  ("Action","Tension"):                  profile(({"close-up","medium"},15),({"low-angle"},10),({"Rule of Thirds"},10),({"Hard Light"},15),({"Low Key"},20),({"handheld"},30)),
  ("Drama","Hope"):                      profile(({"medium"},15),({"eye-level","low-angle"},15),({"Rule of Thirds"},15),({"Soft Light"},20),({"High Key"},20),({"tilt","pan"},15)),
  ("Comedy","Awkwardness"):              profile(({"wide","medium"},20),({"eye-level"},15),({"Symmetrical","Central"},20),({"Soft Light"},10),({"High Key"},20),({"locked"},15)),
})

# Startup check: validates every profile in RULEBOOK against vocabulary, attributes, and weights
VALID_VOCABULARY = {
    "Scale": {"close-up", "extreme-close-up", "medium", "wide", "extreme-wide"},
    "Angle": {"eye-level", "low-angle", "high-angle"},
    "Composition": {"Rule of Thirds", "Symmetrical", "Framing", "Central"},
    "Lighting_quality": {"Hard Light", "Soft Light"},
    "Lighting_intensity": {"High Key", "Low Key"},
    "Movement": {"handheld", "locked", "pan", "zoom", "tilt"},
}

def validate_rulebook():
    for (genre, emotion), profile_rows in RULEBOOK.items():
        # (a) exactly the six ATTRS
        attrs = [r.get("attribute") for r in profile_rows]
        if len(profile_rows) != 6 or set(attrs) != set(ATTRS) or len(set(attrs)) != 6:
            raise ValueError(
                f"Profile for ({genre!r}, {emotion!r}) does not have exactly the six ATTRS: {attrs}"
            )
        # (b) weights that don't sum to 100
        total_weight = sum(r.get("weight", 0) for r in profile_rows)
        if total_weight != 100:
            raise ValueError(
                f"Profile for ({genre!r}, {emotion!r}) weights sum to {total_weight}, expected 100"
            )
        # (c) uses a label outside this vocabulary
        for r in profile_rows:
            attr = r.get("attribute")
            if attr not in VALID_VOCABULARY:
                raise ValueError(
                    f"Profile for ({genre!r}, {emotion!r}) has unknown attribute {attr!r}"
                )
            expected_labels = r.get("expected", set())
            if isinstance(expected_labels, str):
                expected_labels = {expected_labels}
            invalid_labels = set(expected_labels) - VALID_VOCABULARY[attr]
            if invalid_labels:
                raise ValueError(
                    f"Profile for ({genre!r}, {emotion!r}) attribute {attr!r} uses labels {invalid_labels} outside vocabulary: {VALID_VOCABULARY[attr]}"
                )

validate_rulebook()

# Short hint per attribute, used to flesh out a recommendation sentence
RECOMMENDATION_HINTS = {
    "Scale": "shot size shapes how intimate or distant a moment feels",
    "Angle": "camera height changes the sense of power or vulnerability",
    "Composition": "framing guides where the eye goes and how balanced the shot feels",
    "Lighting_quality": "hard vs soft light changes how harsh or gentle the mood reads",
    "Lighting_intensity": "key lighting sets the baseline brightness and mood",
    "Movement": "camera movement adds or removes a sense of stability",
}

# ---------------------------------------------------------------
# Built FROM the rulebook, not hand-maintained separately — so the
# dropdowns can never offer a combination that isn't actually in RULEBOOK.
# ---------------------------------------------------------------
GENRE_TO_EMOTIONS = {}
for _genre, _emotion in RULEBOOK:
    GENRE_TO_EMOTIONS.setdefault(_genre, []).append(_emotion)


def normalize_label(attr: str, raw_label: str) -> str:
    """
    Normalizes classifier raw predictions to match rulebook vocabulary:
      Scale, Angle, Movement -> lowercase, hyphenated (e.g. "close-up", "low-angle", "handheld")
      Composition, Lighting_* -> Title Case (e.g. "Rule of Thirds", "Hard Light")
    """
    if not raw_label:
        return "unknown"

    clean = raw_label.strip()
    attr_lower = attr.lower()

    if attr_lower in ["scale", "angle", "movement"]:
        clean_norm = clean.lower().replace(" ", "-").replace("shot", "").strip("-")
        if "close" in clean_norm and "up" in clean_norm:
            return "close-up"
        if "low" in clean_norm:
            return "low-angle"
        if "high" in clean_norm:
            return "high-angle"
        if "eye" in clean_norm:
            return "eye-level"
        return clean_norm
    else:
        # Title case for Composition and Lighting
        return clean.title()


def compute_alignment_score(detected: dict, genre: str, emotion: str, include_movement: bool = False) -> dict:
    """
    detected: {"Scale": "close-up", "Angle": "low-angle", ...}
    Returns {"score": 0-100, "breakdown": [...], "recommendations": [...]}.
    Raises KeyError if (genre, emotion) isn't a real dropdown combination.
    When include_movement is False, skips Movement attribute so scoring normalizes purely over the 5 trained visual classifiers.
    """
    profile = RULEBOOK[(genre, emotion)]

    rows, earned, total = [], 0.0, 0.0
    for item in profile:
        attr, expected, weight = item["attribute"], item["expected"], item["weight"]
        if not include_movement and attr.lower() == "movement":
            continue

        # Handle dict or string form
        pred = detected.get(attr, detected.get(attr.lower(), "unknown"))
        raw_label = pred["label"] if isinstance(pred, dict) else str(pred)
        
        label = normalize_label(attr, raw_label)
        
        # Check match against expected set (case-insensitive fallback)
        expected_normalized = {normalize_label(attr, e) for e in expected}
        matched = (label in expected) or (label in expected_normalized) or any(e.lower() in label.lower() for e in expected)
        
        points = weight if matched else 0.0
        earned += points
        total += weight
        rows.append({
            "attribute": attr,
            "detected": raw_label,
            "expected": sorted(expected),
            "match": matched,
            "weight": weight,
            "points": points,
        })

    score = round(earned / total * 100, 1) if total else 0.0

    recommendations = [
        f"Detected {r['detected']!r} for {r['attribute']}, but {genre} + {emotion} "
        f"typically calls for {' or '.join(r['expected'])} — "
        f"{RECOMMENDATION_HINTS.get(r['attribute'], 'worth adjusting this.')}."
        for r in rows if not r["match"]
    ]

    return {"score": score, "breakdown": rows, "recommendations": recommendations}
