import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, HubertModel
import torchaudio
import librosa
import numpy as np

# --- Architecture from Research Notebook ---
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

    def forward(self, x): return self.network(x)

class AcousticPipeline:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.labels = ["backward", "forward", "left", "right", "stop"]
        
        # Exact model from Sanity Check
        self.processor = Wav2Vec2Processor.from_pretrained("ntu-spml/distilhubert")
        self.hubert = HubertModel.from_pretrained("ntu-spml/distilhubert").to(device)
        self.hubert.eval()
        
        # Load Classifier
        self.classifier = CommandClassifier(input_dim=768, num_classes=5).to(device)
        self.classifier.load_state_dict(torch.load(model_path, map_location=device))
        self.classifier.eval()

    def predict(self, audio_data):
        # A. Audio Loading & Guardrails (Standardized for Demo)
        # audio_data is 16kHz numpy array from server
        
        # Optional: Apply specific trim if used in training pipeline elsewhere
        # Using top_db=20 as previously requested by user for consistency
        y_trimmed, _ = librosa.effects.trim(audio_data, top_db=20)
        y_norm = librosa.util.normalize(y_trimmed)
        
        # B. Feature Extraction
        inputs = self.processor(y_norm, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # DistilHuBERT inference
            features = self.hubert(**inputs)
            # Average pooling over time dimension to get 768-D vector
            embedding = features.last_hidden_state.mean(dim=1)

            # C. MLP Inference
            logits = self.classifier(embedding)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
            
        return {
            "prediction": self.labels[idx.item()],
            "confidence": float(conf.item())
        }
