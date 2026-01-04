from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel
try:
    from app.services.anpr_service import get_anpr_service
    from app.services.video_processor import get_video_processor
except ImportError:
    from backend.app.services.anpr_service import get_anpr_service
    from backend.app.services.video_processor import get_video_processor
import uvicorn
import os

app = FastAPI(title="Smart Traffic Management System API")

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://smart.sanjibkasti.com.np",
    "http://smart.sanjibkasti.com.np"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the video directory to serve files if needed (optional, but good for direct access)
# app.mount("/videos", StaticFiles(directory="frontend/public/video"), name="videos")

@app.on_event("startup")
async def startup_event():
    # Initialize model on startup
    try:
        get_anpr_service()
        print("ANPR Service Initialized")
    except Exception as e:
        print(f"Failed to initialize ANPR Service: {e}")

@app.get("/")
def read_root():
    return {"message": "Smart Traffic Management System API is running"}

@app.get("/videos")
def list_videos():
    """List all video files in the frontend/public/video directory"""
    video_dir = Path("frontend/public/video")
    if not video_dir.exists():
        # Fallback if running from backend directory
        video_dir = Path("../frontend/public/video")
    
    if not video_dir.exists():
        return {"videos": []}
        
    videos = [f.name for f in video_dir.iterdir() if f.is_file()]
    return {"videos": videos}

@app.post("/detect")
async def detect_plate(file: UploadFile = File(...)):
    """
    Upload an image to detect number plates.
    """
    try:
        contents = await file.read()
        service = get_anpr_service()
        detections = service.process_image(contents)
        return {"filename": file.filename, "detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class StreamRequest(BaseModel):
    source: str

@app.post("/stream/start")
async def start_stream(request: StreamRequest):
    processor = get_video_processor()
    
    # Resolve source path if it's a filename
    source = request.source
    if source != "0" and not source.isdigit():
        # Check if it's a file in the video directory
        video_dir = Path("frontend/public/video")
        if not video_dir.exists():
            video_dir = Path("../frontend/public/video")
        
        potential_path = video_dir / source
        if potential_path.exists():
            source = str(potential_path.resolve())
    
    await processor.start_processing(source)
    return {"status": "started", "source": source}

@app.post("/stream/stop")
def stop_stream():
    processor = get_video_processor()
    processor.stop_processing()
    return {"status": "stopped"}

@app.get("/stream/status")
def get_stream_status():
    processor = get_video_processor()
    return {
        "is_streaming": processor.is_processing,
        "source": processor.current_source
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    processor = get_video_processor()
    await processor.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages if any (e.g. client commands)
            data = await websocket.receive_text()
            # We could handle client commands here too
    except WebSocketDisconnect:
        processor.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
