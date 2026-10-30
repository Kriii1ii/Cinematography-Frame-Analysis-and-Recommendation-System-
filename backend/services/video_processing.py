import cv2
import os
import logging
import numpy as np
from PIL import Image
from typing import Tuple

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.flv', '.3gp', '.wmv'}
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def extract_representative_frame(input_path: str, output_image_path: str) -> str:
    """
    Extracts the midpoint frame from a 5-10 second video clip using OpenCV.
    If video duration is longer than 10 seconds, caps analysis scope to the first 10 seconds (5s midpoint).
    If input is a static image file, reads and re-saves it cleanly.
    """
    ext = os.path.splitext(input_path)[1].lower()
    
    # Handle static image files
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        img = cv2.imread(input_path)
        if img is not None:
            cv2.imwrite(output_image_path, img)
            return output_image_path
        return input_path

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file at {input_path}. Unsupported codec or format.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or type(fps) != float or np.isnan(fps):
        fps = 30.0

    if total_frames <= 0:
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise ValueError(f"Failed to read any frames from video {input_path}")
        cv2.imwrite(output_image_path, frame)
        return output_image_path

    # Cap analysis duration window to maximum 10 seconds (10 * fps frames)
    max_eval_frames = int(10.0 * fps)
    effective_total_frames = min(total_frames, max_eval_frames)

    # Midpoint frame within 10s evaluation window
    midpoint_frame_index = max(0, effective_total_frames // 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, midpoint_frame_index)
    ret, frame = cap.read()

    if not ret or frame is None:
        # Fallback to frame 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    cap.release()

    if not ret or frame is None:
        raise ValueError(f"Could not extract representative frame from {input_path}")

    cv2.imwrite(output_image_path, frame)
    logger.info(f"Extracted representative midpoint frame (frame {midpoint_frame_index}/{effective_total_frames}) to {output_image_path}")
    return output_image_path
