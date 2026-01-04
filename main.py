"""
YOLOv8 Number Plate Detection - Inference Script
Performs inference on images, videos, and live webcam feed
"""

import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import argparse
import time


class NumberPlateDetector:
    def __init__(self, model_path='runs/detect/numberplate_detection/weights/best.pt'):
        """Initialize the detector with trained model"""
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Check GPU
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def detect_image(self, image_path, conf_threshold=0.25, save=True):
        """Detect number plates in a single image"""
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            device=self.device,
            save=save,
            project='inference_results',
            name='images'
        )
        
        print(f"\nDetections in {image_path}:")
        print(f"Number of plates detected: {len(results[0].boxes)}")
        
        # Print detection details
        for i, box in enumerate(results[0].boxes):
            conf = box.conf[0].item()
            coords = box.xyxy[0].tolist()
            print(f"  Plate {i+1}: Confidence={conf:.2f}, BBox={[int(c) for c in coords]}")
        
        return results
    
    def detect_video(self, video_path, conf_threshold=0.25, save=True):
        """Detect number plates in a video file"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer if saving
        out = None
        if save:
            output_dir = Path('inference_results/videos')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"detected_{Path(video_path).name}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"Saving output to: {output_path}")
        
        frame_count = 0
        detection_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = self.model.predict(
                source=frame,
                conf=conf_threshold,
                device=self.device,
                verbose=False
            )
            
            # Draw detections
            annotated_frame = results[0].plot()
            detection_count += len(results[0].boxes)
            
            # Display info
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}/{total_frames} | Detections: {len(results[0].boxes)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Show frame
            cv2.imshow('Number Plate Detection', annotated_frame)
            
            # Save frame
            if out:
                out.write(annotated_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopping video processing...")
                break
            
            # Progress update every 30 frames
            if frame_count % 30 == 0:
                print(f"Processed: {frame_count}/{total_frames} frames")
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nVideo processing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Total detections: {detection_count}")
        print(f"Average detections per frame: {detection_count/frame_count:.2f}")
    
    def detect_webcam(self, conf_threshold=0.25, camera_id=0):
        """Real-time detection from webcam"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n=== Webcam Detection Started ===")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("Press 'r' to reset detection count")
        
        frame_count = 0
        detection_count = 0
        fps_display = 0
        save_count = 0
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            frame_count += 1
            
            # Run detection
            results = self.model.predict(
                source=frame,
                conf=conf_threshold,
                device=self.device,
                verbose=False
            )
            
            # Draw detections
            annotated_frame = results[0].plot()
            num_detections = len(results[0].boxes)
            detection_count += num_detections
            
            # Calculate FPS
            fps = 1 / (time.time() - start_time)
            fps_display = fps * 0.9 + fps_display * 0.1  # Smoothing
            
            # Display info
            info_text = [
                f"FPS: {fps_display:.1f}",
                f"Detections: {num_detections}",
                f"Total: {detection_count}",
                f"Confidence: {conf_threshold}"
            ]
            
            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(
                    annotated_frame,
                    text,
                    (10, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Show frame
            cv2.imshow('Number Plate Detection - Live', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nStopping webcam...")
                break
            elif key == ord('s'):
                # Save current frame
                save_dir = Path('inference_results/snapshots')
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"snapshot_{save_count:04d}.jpg"
                cv2.imwrite(str(save_path), annotated_frame)
                print(f"Saved snapshot: {save_path}")
                save_count += 1
            elif key == ord('r'):
                # Reset counters
                detection_count = 0
                frame_count = 0
                print("Counters reset")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n=== Webcam Session Summary ===")
        print(f"Total frames: {frame_count}")
        print(f"Total detections: {detection_count}")
        print(f"Snapshots saved: {save_count}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Number Plate Detection Inference')
    parser.add_argument('--mode', type=str, default='webcam',
                        choices=['image', 'video', 'webcam'],
                        help='Detection mode: image, video, or webcam')
    parser.add_argument('--source', type=str, default='',
                        help='Path to image or video file')
    parser.add_argument('--model', type=str,
                        default='runs/detect/numberplate_detection/weights/best.pt',
                        help='Path to trained model')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (0-1)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID for webcam mode')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = NumberPlateDetector(model_path=args.model)
    
    # Run detection based on mode
    if args.mode == 'image':
        if not args.source:
            print("Error: --source required for image mode")
            print("Example: python main.py --mode image --source path/to/image.jpg")
            return
        
        if not Path(args.source).exists():
            print(f"Error: Image not found: {args.source}")
            return
        
        detector.detect_image(
            args.source,
            conf_threshold=args.conf,
            save=not args.no_save
        )
    
    elif args.mode == 'video':
        if not args.source:
            print("Error: --source required for video mode")
            print("Example: python main.py --mode video --source path/to/video.mp4")
            return
        
        if not Path(args.source).exists():
            print(f"Error: Video not found: {args.source}")
            return
        
        detector.detect_video(
            args.source,
            conf_threshold=args.conf,
            save=not args.no_save
        )
    
    elif args.mode == 'webcam':
        detector.detect_webcam(
            conf_threshold=args.conf,
            camera_id=args.camera
        )
    
    print("\n✓ Detection complete!")


if __name__ == "__main__":
    main()
