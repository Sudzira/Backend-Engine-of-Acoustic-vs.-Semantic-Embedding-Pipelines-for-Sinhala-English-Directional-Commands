import asyncio
import io
import time
import numpy as np
import torch
import librosa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# Import pipelines
from pipelines.asr_pipeline import ASRPipeline
from pipelines.acoustic_pipeline import AcousticPipeline

app = FastAPI(title="Directional Command Recognition Server")

print("Initializing models... This may take a moment.")
# Initialize pipelines (models are loaded into memory here, only once)
# Paths are relative to the root directory where the server is run from
pipeline_a = ASRPipeline(model_path="models/asr_classifier.pt")
pipeline_b = AcousticPipeline(model_path="models/acoustic_classifier.pt")
print("Models loaded successfully.")

async def process_audio(audio_bytes):
    """Convert raw bytes to 16kHz numpy array."""
    # Assuming incoming bytes are 16-bit PCM (common for WebSockets)
    # If using webm/wav blobs, librosa.load with io.BytesIO is safer
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    return audio

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to WebSocket.")
    try:
        while True:
            # 1. Receive audio bytes from the browser
            audio_bytes = await websocket.receive_bytes()
            
            # 2. Process audio
            audio_data = await process_audio(audio_bytes)
            
            # 3. Run both pipelines concurrently
            start_a = time.time()
            task_a = pipeline_a.predict(audio_data)
            
            start_b = time.time()
            task_b = pipeline_b.predict(audio_data)
            
            result_a, result_b = await asyncio.gather(task_a, task_b)
            
            latency_a = int((time.time() - start_a) * 1000)
            latency_b = int((time.time() - start_b) * 1000)
            
            # 4. Prepare combined response
            response = {
                "pipelineA": {
                    "command": result_a["prediction"],
                    "confidence": result_a["confidence"],
                    "transcription": result_a["transcription"],
                    "latency_ms": latency_a
                },
                "pipelineB": {
                    "command": result_b["prediction"],
                    "confidence": result_b["confidence"],
                    "latency_ms": latency_b
                }
            }
            
            # 5. Send back to frontend
            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error processing audio: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
