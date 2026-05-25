import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel
import librosa
import numpy as np

class CommandClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=5):
        super(CommandClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(0.20),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.40),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.network(x)

class AcousticPipeline:
    def __init__(self, model_path="models/acoustic_classifier.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = ["backward", "forward", "left", "right", "stop"]
        
        # Load DistilHuBERT
        print("[Acoustic] Loading DistilHuBERT model...")
        self.processor = AutoProcessor.from_pretrained("ntu-spml/distilhubert")
        self.base_model = AutoModel.from_pretrained("ntu-spml/distilhubert").to(self.device)
        
        # Load Classifier
        self.classifier = CommandClassifier().to(self.device)
        try:
            self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[Acoustic] Loaded classifier from {model_path}")
        except Exception as e:
            print(f"[Acoustic] Warning: Could not load classifier: {e}")
        self.classifier.eval()

    def preprocess_audio(self, audio_data):
        # audio_data is expected to be a numpy array at 16kHz
        # Apply normalization/trimming
        audio = audio_data - np.mean(audio_data)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        audio = librosa.util.normalize(audio)
        return audio

    async def predict(self, audio_data):
        # 1. Preprocess
        audio = self.preprocess_audio(audio_data)
        
        # 2. Extract Acoustic Features (DistilHuBERT)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.base_model(**inputs)
            # Use mean pooling over the last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # 3. Classify
        with torch.no_grad():
            logits = self.classifier(embeddings)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
            
        return {
            "prediction": self.labels[idx.item()],
            "confidence": float(conf.item())
        }
