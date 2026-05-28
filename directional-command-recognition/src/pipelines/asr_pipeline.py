import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sentence_transformers import SentenceTransformer
import librosa
import numpy as np

# --- Architecture from Research Notebook ---
class CommandClassifier(nn.Module):
    def __init__(self, input_dim=384, num_classes=5):
        super(CommandClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): 
        # BatchNorm1d requires at least 2D input (batch_size, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)

class ASRPipeline:
    def __init__(self, model_path, whisper_path, minilm_path, device="cpu"):
        self.device = device
        self.labels = ["backward", "forward", "left", "right", "stop"]
        
        # Load base models
        self.processor = WhisperProcessor.from_pretrained(whisper_path)
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(whisper_path).to(device)
        self.asr_model.eval()
        
        # Load MiniLM
        self.st_model = SentenceTransformer(minilm_path, device=device)
        
        # Load Classifier (Using the dynamic quantization logic if needed, 
        # but for demo stability we'll stick to full precision unless requested)
        self.classifier = CommandClassifier(input_dim=384, num_classes=5).to(device)
        self.classifier.load_state_dict(torch.load(model_path, map_location=device))
        self.classifier.eval()

    def load_audio_for_inference(self, audio_array, sr=16000):
        # Robust Preprocessing from Notebook
        audio = audio_array - np.mean(audio_array)
        audio, _ = librosa.effects.trim(audio, top_db=25)

        rms = np.sqrt(np.mean(audio**2))
        if rms > 0: audio = audio * (0.05 / rms)

        if len(audio) < (sr * 0.5):
            pad = int(sr * 0.1)
            audio = np.pad(audio, (pad, pad), 'constant')
        return audio.astype(np.float32)

    def transcribe_for_inference(self, audio_array):
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        generated_ids = self.asr_model.generate(
            inputs.input_features.to(self.device),
            attention_mask=inputs.attention_mask.to(self.device),
            language="sinhala",
            task="transcribe",
            num_beams=3,
            max_length=40
        )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def predict(self, audio_data):
        # 1. Preprocess
        audio = self.load_audio_for_inference(audio_data)
        
        # 2. Transcribe
        transcription = self.transcribe_for_inference(audio)
        
        # 3. Extract Semantic Embedding
        embedding = self.st_model.encode(transcription, show_progress_bar=False)
        
        with torch.no_grad():
            input_tensor = torch.tensor(embedding).unsqueeze(0).to(self.device)
            raw_output = self.classifier(input_tensor)
            probabilities = F.softmax(raw_output, dim=1)[0]

            max_prob, predicted_idx = torch.max(probabilities, dim=0)
            
        return {
            "transcription": transcription,
            "prediction": self.labels[predicted_idx.item()],
            "confidence": float(max_prob.item())
        }
