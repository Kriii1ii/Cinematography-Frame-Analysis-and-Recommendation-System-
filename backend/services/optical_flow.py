import cv2
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def analyze_camera_movement(video_path: str, max_duration_sec: float = 10.0) -> Dict[str, Any]:
    """
    Computes OpenCV Farneback optical flow over consecutive video frames to estimate camera movement.
    Automatically caps optical flow computation to maximum 10 seconds of video.
    Returns quantitative motion stats and a derived movement label ("Static", "Handheld", "Tracking", "Panning", "Tilting", "Movement analysis pending").
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Could not open video at {video_path} for optical flow analysis.")
        return {
            "movement": "Movement analysis pending",
            "stats": {"mean_motion": 0.0, "max_motion": 0.0, "mean_h_motion": 0.0, "mean_v_motion": 0.0, "frame_pairs": 0}
        }
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or type(fps) != float or np.isnan(fps):
        fps = 30.0

    max_frames = int(max_duration_sec * fps)

    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        cap.release()
        return {
            "movement": "Movement analysis pending",
            "stats": {"mean_motion": 0.0, "max_motion": 0.0, "mean_h_motion": 0.0, "mean_v_motion": 0.0, "frame_pairs": 0}
        }

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Resize frame for efficient optical flow computation
    h, w = prev_gray.shape
    if w > 640:
        scale = 640.0 / w
        prev_gray = cv2.resize(prev_gray, (640, int(h * scale)))

    magnitudes = []
    h_motions = []
    v_motions = []
    frame_pairs = 0
    
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, curr_frame = cap.read()
        if not ret or curr_frame is None:
            break
            
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        if curr_gray.shape != prev_gray.shape:
            curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))
            
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        fx, fy = flow[..., 0], flow[..., 1]
        mag, _ = cv2.cartToPolar(fx, fy)
        
        magnitudes.append(float(np.mean(mag)))
        h_motions.append(float(np.mean(np.abs(fx))))
        v_motions.append(float(np.mean(np.abs(fy))))
        
        frame_pairs += 1
        prev_gray = curr_gray
        frame_count += 1
        
    cap.release()
    
    if frame_pairs == 0 or len(magnitudes) == 0:
        return {
            "movement": "Movement analysis pending",
            "stats": {"mean_motion": 0.0, "max_motion": 0.0, "mean_h_motion": 0.0, "mean_v_motion": 0.0, "frame_pairs": 0}
        }
        
    mean_motion = float(np.mean(magnitudes))
    max_motion = float(np.max(magnitudes))
    mean_h_motion = float(np.mean(h_motions))
    mean_v_motion = float(np.mean(v_motions))
    
    # Derive human readable movement label based on optical flow thresholds
    if mean_motion < 0.4:
        movement = "Static"
    elif mean_h_motion > 2.5 * mean_v_motion and mean_h_motion > 1.2:
        movement = "Panning"
    elif mean_v_motion > 2.5 * mean_h_motion and mean_v_motion > 1.2:
        movement = "Tilting"
    elif mean_motion > 1.8 and abs(mean_h_motion - mean_v_motion) < 1.0:
        movement = "Handheld"
    else:
        movement = "Tracking"
        
    return {
        "movement": movement,
        "stats": {
            "mean_motion": round(mean_motion, 4),
            "max_motion": round(max_motion, 4),
            "mean_h_motion": round(mean_h_motion, 4),
            "mean_v_motion": round(mean_v_motion, 4),
            "frame_pairs": frame_pairs
        }
    }
