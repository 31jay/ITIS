import cv2
import asyncio
import base64
import time
import json
from typing import List, Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from .anpr_service import get_anpr_service

class VideoProcessor:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.is_processing = False
        self.current_source: Optional[str] = None
        self.cap = None
        self.history: List[Dict[str, Any]] = []
        self.anpr_service = None
        self.latest_frame_data: Optional[Dict[str, Any]] = None
        self.processing_task = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total: {len(self.active_connections)}")
        # Send history and latest frame immediately
        if self.history:
             await websocket.send_json({"type": "history", "data": self.history})
        if self.latest_frame_data:
             await websocket.send_json(self.latest_frame_data)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        self.latest_frame_data = message
        to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except (WebSocketDisconnect, RuntimeError):
                to_remove.append(connection)
        
        for conn in to_remove:
            self.disconnect(conn)

    async def start_processing(self, source: str):
        if self.is_processing:
            print("Already processing, stopping current...")
            self.stop_processing()
            # Wait a bit for cleanup
            await asyncio.sleep(0.5)
        
        print(f"Starting processing for source: {source}")
        self.current_source = source
        self.is_processing = True
        self.anpr_service = get_anpr_service()
        
        # Reset tracker state for new video
        if hasattr(self.anpr_service, 'reset_tracker'):
            self.anpr_service.reset_tracker()
        
        # Start the processing loop in a background task
        self.processing_task = asyncio.create_task(self._process_loop())

    def stop_processing(self):
        print("Stopping processing...")
        self.is_processing = False
        if self.cap:
            self.cap.release()
            self.cap = None

    async def _process_loop(self):
        try:
            # Handle integer source for webcam
            if str(self.current_source).isdigit():
                video_source = int(self.current_source)
            else:
                video_source = self.current_source
                import os
                if not os.path.exists(video_source):
                    print(f"Error: Video file does not exist at path: {video_source}")

            self.cap = cv2.VideoCapture(video_source)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open video source {video_source}")
                self.is_processing = False
                return

            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 60: fps = 30
            frame_delay = 1.0 / fps

            while self.is_processing and self.cap.isOpened():
                start_time = time.time()
                ret, frame = self.cap.read()
                
                if not ret:
                    # Loop video if it's a file
                    if isinstance(video_source, str) and not str(video_source).isdigit():
                        print("Video ended, restarting...")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("Stream ended.")
                        break

                # Resize for performance if needed (optional)
                # frame = cv2.resize(frame, (640, 480))

                # Process frame
                try:
                    # Use tracking mode for video processing
                    detections = self.anpr_service.process_frame(frame, track=True)
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    detections = []

                # Calculate FPS and Processing Time
                process_time = time.time() - start_time
                current_fps = 1.0 / process_time if process_time > 0 else 0

                # Encode frame to base64 for frontend
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Prepare data
                timestamp = time.time()
                data = {
                    "type": "frame",
                    "image": frame_base64,
                    "detections": detections,
                    "timestamp": timestamp,
                    "fps": round(current_fps, 1),
                    "processing_time_ms": round(process_time * 1000, 1)
                }
                
                # Update history if detections found
                for det in detections:
                    if det['text'] and len(det['text']) > 2:
                        # Check if we already have this plate recently to avoid spam
                        is_recent = False
                        for h in self.history[-10:]: # Check last 10
                            if h['text'] == det['text'] and (timestamp - h['timestamp']) < 5:
                                is_recent = True
                                break
                        
                        if not is_recent:
                            history_item = {
                                "text": det['text'],
                                "confidence": det['ocr_confidence'],
                                "timestamp": timestamp,
                                "image_crop": det.get('plate_image')
                            }
                            self.history.append(history_item)
                            # Keep history manageable
                            if len(self.history) > 100:
                                self.history.pop(0)
                            
                            # Broadcast new history item separately or let client handle it
                            # For simplicity, we send the full history update or just the new item
                            # Let's send a history update event
                            await self.broadcast({"type": "new_history", "data": history_item})

                await self.broadcast(data)
                
                # Maintain FPS
                process_time = time.time() - start_time
                sleep_time = max(0, frame_delay - process_time)
                await asyncio.sleep(sleep_time)

        except Exception as e:
            print(f"Error in process loop: {e}")
        finally:
            if self.cap:
                self.cap.release()
            self.is_processing = False
            print("Processing loop finished.")

# Global instance
video_processor = VideoProcessor()

def get_video_processor():
    return video_processor
