import cv2
import numpy as np
from typing import Tuple, Optional

def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
   """Resize image while maintaining aspect ratio"""
   h, w = image.shape[:2]
   
   if max(h, w) <= max_size:
       return image
   
   if h > w:
       new_h = max_size
       new_w = int(w * (max_size / h))
   else:
       new_w = max_size
       new_h = int(h * (max_size / w))
   
   return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def crop_face_region(image: np.ndarray, landmarks: np.ndarray, padding: float = 0.2) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
   """Crop face region from image with padding"""
   # Get bounding box of face landmarks
   x_coords = landmarks[:, 0]
   y_coords = landmarks[:, 1]
   
   x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
   y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
   
   # Add padding
   face_width = x_max - x_min
   face_height = y_max - y_min
   
   pad_x = int(face_width * padding)
   pad_y = int(face_height * padding)
   
   # Ensure boundaries are within image
   h, w = image.shape[:2]
   x1 = max(0, x_min - pad_x)
   y1 = max(0, y_min - pad_y)
   x2 = min(w, x_max + pad_x)
   y2 = min(h, y_max + pad_y)
   
   cropped = image[y1:y2, x1:x2]
   bbox = (x1, y1, x2, y2)
   
   return cropped, bbox

def enhance_image_quality(image: np.ndarray) -> np.ndarray:
   """Apply basic image enhancement"""
   # Convert to LAB color space
   lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
   
   # Split channels
   l, a, b = cv2.split(lab)
   
   # Apply CLAHE to L channel
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
   l = clahe.apply(l)
   
   # Merge channels and convert back to BGR
   lab = cv2.merge([l, a, b])
   enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
   
   return enhanced
