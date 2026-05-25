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
pipeline_a = ASRPipeline(model_path="models/asr_classifier.pt")
pipeline_b = AcousticPipeline(model_path="models/acoustic_classifier.pt")
print("Models loaded successfully.")

async def process_audio(audio_bytes):
    """Convert raw bytes to 16kHz numpy array."""
    # Using librosa to decode any browser audio format (wav/webm)
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    return audio

async def run_and_send_a(websocket, audio_data, start_time):
    """Run ASR pipeline and send result immediately."""
    try:
        result = await pipeline_a.predict(audio_data)
        latency = int((time.time() - start_time) * 1000)
        response = {
            "pipeline": "A",
            "command": result["prediction"],
            "confidence": result["confidence"],
            "transcription": result["transcription"],
            "latency_ms": latency
        }
        await websocket.send_json(response)
    except Exception as e:
        print(f"Error in Pipeline A: {e}")

async def run_and_send_b(websocket, audio_data, start_time):
    """Run Acoustic pipeline and send result immediately."""
    try:
        result = await pipeline_b.predict(audio_data)
        latency = int((time.time() - start_time) * 1000)
        response = {
            "pipeline": "B",
            "command": result["prediction"],
            "confidence": result["confidence"],
            "latency_ms": latency
        }
        await websocket.send_json(response)
    except Exception as e:
        print(f"Error in Pipeline B: {e}")

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
            
            # 3. Launch both pipelines as independent concurrent tasks
            # This allows Pipeline B to finish and send its message 
            # while Pipeline A is still transcribing.
            start_time = time.time()
            asyncio.create_task(run_and_send_a(websocket, audio_data, start_time))
            asyncio.create_task(run_and_send_b(websocket, audio_data, start_time))

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Global error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
