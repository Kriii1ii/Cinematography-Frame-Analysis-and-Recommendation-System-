# Cinematography-Frame-Analysis-and-Recommendation-System-

An intelligent multimodal video frame analysis and directorial recommendation engine. The system analyzes uploaded film scenes across key cinematography dimensions, evaluates intent alignment against target genres and emotions, and generates creative directorial recommendations.

---

## 📽️ Features

- **Automated Cinematographic Dimension Classification**:
  - **Shot Scale**: Extreme Long Shot, Long Shot, Medium Shot, Close-Up
  - **Camera Angle**: Bird's Eye View, High Angle, Low Angle
  - **Composition**: Rule of Thirds, Central, Framing, Symmetrical
  - **Lighting Quality**: Hard Light, Soft Light
  - **Lighting Intensity**: High Key, Low Key
  - **Camera Movement**: Optical flow-derived motion classification (Handheld, Tracking, Static, Panning, Tilting)

- **Genre & Emotion Alignment Scoring**:
  - Compares classified visual attributes against an extensive cinematic rulebook across 13 genres and associated emotional profiles.
  - Computes weighted intent alignment scores ($0 - 100$) with highlighted strengths and mismatches.

- **Generative Directorial Recommendations**:
  - Integrates with Google Gemini (or compatible LLM providers) to provide actionable directorial advice, alternative framing solutions, and creative options for scene reshoots.
  - Transparent model uncertainty disclosures when classifier margins are close.

- **Broadsheet Editorial Interface**:
  - Retro-newspaper aesthetic workbench with real-time video upload, freeze-frame preview, and telemetry bars.

---

## 🏗️ Architecture

```
                       Uploaded Video Clip (.mp4)
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
          Midpoint Frame (OpenCV)       Optical Flow Motion
                    │
         OpenAI CLIP (ViT-B/32)
         512-dim Semantic Vector
                    │
       ┌────────────┼────────────┬────────────┬────────────┐
       ▼            ▼            ▼            ▼            ▼
     Scale        Angle     Composition   Lighting-Q   Lighting-I
    (RF-500)     (RF-300)    (Logistic)   (Logistic)   (Logistic)
       │            │            │            │            │
       └────────────┴────────────┼────────────┴────────────┘
                                 ▼
                     Intent Alignment Scoring
                    (Rulebook Knowledge Base)
                                 │
                                 ▼
                  LLM Directorial Consultation
                   (Google Gemini 3.5 Flash)
                                 │
                                 ▼
                    Broadsheet Interactive UI
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Modern Web Browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:Kriii1ii/Cinematography-Frame-Analysis-and-Recommendation-System-.git
   cd Cinematography-Frame-Analysis-and-Recommendation-System-
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn torch torchvision transformers scikit-learn opencv-python Pillow python-multipart python-dotenv httpx
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Add your Google Gemini API key to .env
   ```

5. **Initialize mock database (optional)**:
   ```bash
   cp database.example.json database.json
   ```

6. **Start the application**:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

7. **Open in browser**:
   Navigate to `http://localhost:8000/login.html` or `http://localhost:8000/main.html`.

---

## 🔒 Confidentiality & Security Notice

- Environment configuration files (`.env`) and local user databases (`database.json`) are strictly excluded from version control to protect credentials and personal data.
