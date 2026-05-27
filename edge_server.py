from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import cv2
import numpy as np
import time
import psutil

app = FastAPI(title="Edge Computing Server")

def process_image_cv(image_bytes: bytes, filter_type: str) -> bytes:
    """Applies a heavy OpenCV filter to simulate a computational task."""
    # Convert bytes to numpy array, then to OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Apply the requested filter
    if filter_type == "Grayscale":
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Convert back to BGR so channels match for encoding
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    elif filter_type == "Blur":
        # A heavy gaussian blur
        processed_img = cv2.GaussianBlur(img, (51, 51), 0)
    elif filter_type == "Edge Detection":
        processed_img = cv2.Canny(img, 100, 200)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    else:
        processed_img = img

    # Simulate extra heavy computation for the server
    time.sleep(0.5) 
    
    # Encode back to bytes (jpg format)
    _, encoded_img = cv2.imencode('.jpg', processed_img)
    return encoded_img.tobytes()




@app.post("/process")
async def process_image_endpoint(file: UploadFile = File(...), filter_type: str = "Grayscale"):
    """
    API Endpoint that receives an image, processes it, and returns the result.
    This acts as our "Edge Node" doing the heavy lifting.
    """
    print(f"[SERVER] Received image for processing. Filter: {filter_type}")
    
    # Read the uploaded file bytes
    image_bytes = await file.read()
    
    # Process the image
    result_bytes = process_image_cv(image_bytes, filter_type)
    
    print("[SERVER] Processing complete. Sending back to client.")
    return Response(content=result_bytes, media_type="image/jpeg")

@app.get("/")
def root():
    return {
        "message": "Edge Computing Server Running",
        "status": "online"
    }

@app.get("/status")
def get_server_status():
    """Returns the current CPU load of the Edge Server."""
    return {"cpu_load": psutil.cpu_percent(interval=0.1)}

# To run this server from the terminal, you will use:
# uvicorn edge_server:app --reload --port 8000