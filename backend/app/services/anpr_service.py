import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import torch
from pathlib import Path
import base64

class ANPRService:
    def __init__(self, plate_model_path: str):
        self.plate_model_path = plate_model_path
        print(f"Loading Plate Detection Model from: {plate_model_path}")
        self.plate_model = YOLO(plate_model_path)
        
        print("Loading Vehicle Detection Model (yolov8n.pt)...")
        # This will automatically download yolov8n.pt if not present
        self.vehicle_model = YOLO('yolov8n.pt')
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(['en', 'ne'], gpu=torch.cuda.is_available())
        
        self.processed_ids = set()

    def reset_tracker(self):
        """Reset the tracker state."""
        self.processed_ids = set()

    def calculate_sharpness(self, image):
        """Calculate the sharpness of an image using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def read_plate_text(self, image_crop):
        """Read text from a cropped plate image using multiple preprocessing techniques"""
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        
        # List of images to try OCR on
        images_to_try = [gray]
        
        # 2. Contrast Enhancement
        equalized = cv2.equalizeHist(gray)
        images_to_try.append(equalized)
        
        # 3. Inversion (Negative) - Crucial for Nepal
        inverted = cv2.bitwise_not(gray)
        images_to_try.append(inverted)
        
        # 4. Thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        images_to_try.append(thresh)
        
        final_text = ""
        highest_conf = 0.0
        
        for img in images_to_try:
            results = self.reader.readtext(img)
            
            current_text = ""
            current_conf = 0.0
            count = 0
            
            for (bbox, t, prob) in results:
                if prob > 0.3:
                    current_text += t + " "
                    current_conf += prob
                    count += 1
            
            if count > 0:
                avg_conf = current_conf / count
                if avg_conf > highest_conf and len(current_text.strip()) > 2:
                    highest_conf = avg_conf
                    final_text = current_text
        
        return final_text.strip(), highest_conf

    def process_frame(self, frame, track=False):
        """
        Process a single frame (numpy array).
        If track=True, uses object tracking and only runs OCR on new vehicles crossing the ROI.
        """
        if frame is None:
             raise ValueError("Empty frame")

        height, width = frame.shape[:2]
        
        # 1. Downscale for faster vehicle detection
        # Target size 640px max dimension
        scale = 640 / max(height, width)
        if scale < 1:
            frame_small = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            frame_small = frame
            scale = 1.0

        # 2. Vehicle Detection / Tracking
        # Lower confidence to 0.25 to better detect smaller vehicles like motorcycles
        if track:
            # Use tracking (persist=True keeps ID across frames)
            vehicle_results = self.vehicle_model.track(frame_small, conf=0.25, classes=[2, 3, 5, 7], device=self.device, persist=True, verbose=False, tracker="bytetrack.yaml")
        else:
            # Standard detection
            vehicle_results = self.vehicle_model.predict(frame_small, conf=0.25, classes=[2, 3, 5, 7], device=self.device, verbose=False)
        
        detections = []
        
        # Define ROI (Region of Interest) - e.g., a horizontal band at 50% height
        roi_y_center = int(height * 0.5)
        roi_band = int(height * 0.15) # +/- 15% tolerance
        
        for result in vehicle_results:
            if result.boxes is None:
                continue
                
            boxes = result.boxes.xyxy.cpu().numpy()
            
            # Get IDs if available
            if track and result.boxes.id is not None:
                ids = result.boxes.id.cpu().numpy()
            else:
                ids = [None] * len(boxes)

            for box, track_id in zip(boxes, ids):
                # Map coordinates back to original frame size
                vx1, vy1, vx2, vy2 = box.tolist()
                vx1, vy1, vx2, vy2 = int(vx1/scale), int(vy1/scale), int(vx2/scale), int(vy2/scale)
                
                # Ensure coordinates are within bounds
                vx1, vy1 = max(0, vx1), max(0, vy1)
                vx2, vy2 = min(width, vx2), min(height, vy2)
                
                # Determine if we should run OCR
                should_process_ocr = True
                
                if track and track_id is not None:
                    track_id = int(track_id)
                    cy = (vy1 + vy2) // 2
                    
                    # Check if in ROI
                    in_roi = (roi_y_center - roi_band) < cy < (roi_y_center + roi_band)
                    
                    if track_id in self.processed_ids:
                        # Already processed, skip OCR to save resources
                        should_process_ocr = False
                    elif not in_roi:
                        # Not in ROI yet (or passed it), skip OCR
                        should_process_ocr = False
                    else:
                        # In ROI and not processed -> Run OCR!
                        should_process_ocr = True
                        self.processed_ids.add(track_id)
                
                # If we shouldn't process OCR, just add the vehicle bbox (optional: could return cached text)
                if not should_process_ocr:
                    detections.append({
                        "bbox": [vx1, vy1, vx2, vy2], # Use vehicle bbox
                        "detection_confidence": float(result.boxes.conf[0]) if result.boxes.conf is not None else 0.0,
                        "text": "", # No text
                        "ocr_confidence": 0.0,
                        "sharpness": 0.0,
                        "vehicle_bbox": [vx1, vy1, vx2, vy2],
                        "track_id": track_id
                    })
                    continue

                # Crop vehicle from original high-quality frame
                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                
                if vehicle_crop.size == 0:
                    continue

                # 3. Plate Detection on Vehicle Crop
                # Lower confidence for plates as well, especially for small bike plates
                plate_results = self.plate_model.predict(vehicle_crop, conf=0.2, device=self.device, verbose=False)
                
                plate_found = False
                for plate_result in plate_results:
                    for plate_box in plate_result.boxes:
                        px1, py1, px2, py2 = map(int, plate_box.xyxy[0].tolist())
                        conf = float(plate_box.conf[0].item())
                        
                        # Crop plate from vehicle crop
                        plate_crop = vehicle_crop[py1:py2, px1:px2]
                        
                        text = ""
                        ocr_conf = 0.0
                        sharpness = 0.0
                        plate_image_base64 = None
                        
                        if plate_crop.size > 0:
                            sharpness = self.calculate_sharpness(plate_crop)
                            # Only OCR if it's big enough
                            if (px2 - px1) > 30:
                                text, ocr_conf = self.read_plate_text(plate_crop)
                                # Encode plate image for frontend
                                _, buffer = cv2.imencode('.jpg', plate_crop)
                                plate_image_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Calculate absolute coordinates for the plate on the original frame
                        abs_x1 = vx1 + px1
                        abs_y1 = vy1 + py1
                        abs_x2 = vx1 + px2
                        abs_y2 = vy1 + py2
                        
                        detections.append({
                            "bbox": [abs_x1, abs_y1, abs_x2, abs_y2],
                            "detection_confidence": conf,
                            "text": text,
                            "ocr_confidence": ocr_conf,
                            "sharpness": sharpness,
                            "vehicle_bbox": [vx1, vy1, vx2, vy2],
                            "track_id": track_id,
                            "plate_image": plate_image_base64
                        })
                        plate_found = True
                
                # If no plate found but we decided to process, maybe add a placeholder or just the vehicle
                if not plate_found:
                     detections.append({
                        "bbox": [vx1, vy1, vx2, vy2],
                        "detection_confidence": float(result.boxes.conf[0]) if result.boxes.conf is not None else 0.0,
                        "text": "",
                        "ocr_confidence": 0.0,
                        "sharpness": 0.0,
                        "vehicle_bbox": [vx1, vy1, vx2, vy2],
                        "track_id": track_id
                    })
        
        return detections

    def process_image(self, image_bytes):
        """
        Process a single image.
        Returns: List of detections with bounding boxes, text, confidence, and sharpness score.
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Could not decode image")

        return self.process_frame(frame)

# Singleton instance
anpr_service = None

def get_anpr_service():
    global anpr_service
    if anpr_service is None:
        # Try multiple paths to find the model
        possible_paths = [
            Path("backend/models/best.pt"),
            Path("models/best.pt"),
            Path("../models/best.pt"),
            Path("cindrella/best.pt") # Just in case
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            # If still not found, try absolute path based on current file
            current_dir = Path(__file__).parent
            # backend/app/services/ -> backend/models/
            abs_path = current_dir.parent.parent / "models" / "best.pt"
            if abs_path.exists():
                model_path = abs_path
        
        if model_path is None:
            raise FileNotFoundError("Could not find best.pt model file in any expected location.")

        anpr_service = ANPRService(str(model_path))
    return anpr_service
