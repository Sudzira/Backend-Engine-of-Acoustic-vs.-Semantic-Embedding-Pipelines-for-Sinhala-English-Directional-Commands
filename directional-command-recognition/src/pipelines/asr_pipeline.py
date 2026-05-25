import torch
import torch.nn as nn
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sentence_transformers import SentenceTransformer
import librosa
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, dim, p=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(p)
        )
    def forward(self, x):
        return x + self.block(x)

class CommandClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=5, p=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p/2),
        )
        self.res1 = ResidualBlock(256, p)
        self.res2 = ResidualBlock(256, p)
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.head(self.res2(self.res1(self.net(x))))

class ASRPipeline:
    def __init__(self, model_path="models/asr_classifier.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = ["backward", "forward", "left", "right", "stop"]
        
        # Load Base Models
        print("[ASR] Loading Whisper & LaBSE models...")
        self.processor = WhisperProcessor.from_pretrained("Subhaka/whisper-small-Sinhala-Fine_Tune")
        self.asr_model = WhisperForConditionalGeneration.from_pretrained("Subhaka/whisper-small-Sinhala-Fine_Tune").to(self.device)
        self.st_model = SentenceTransformer("sentence-transformers/LaBSE", device=self.device)
        
        # Load Classifier
        self.classifier = CommandClassifier().to(self.device)
        try:
            self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[ASR] Loaded classifier from {model_path}")
        except Exception as e:
            print(f"[ASR] Warning: Could not load classifier: {e}")
        self.classifier.eval()

    def preprocess_audio(self, audio_data):
        # audio_data is expected to be a numpy array at 16kHz
        # Apply trimming and normalization as per training
        audio = audio_data - np.mean(audio_data)
        audio, _ = librosa.effects.trim(audio, top_db=25)
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio *= (0.05 / rms)
        
        # Ensure minimum length for Whisper
        if len(audio) < 16000 * 0.5:
            audio = np.pad(audio, (1600, 1600))
        return audio

    async def predict(self, audio_data):
        # 1. Preprocess
        audio = self.preprocess_audio(audio_data)
        
        # 2. Transcribe (Whisper)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.asr_model.generate(
                input_features, 
                attention_mask=attention_mask,
                language="sinhala", 
                task="transcribe", 
                num_beams=3, 
                max_length=40
            )
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 3. Embed (LaBSE)
        embedding = self.st_model.encode(transcription, show_progress_bar=False)
        embedding_tensor = torch.tensor(embedding).unsqueeze(0).to(self.device)
        
        # 4. Classify
        with torch.no_grad():
            logits = self.classifier(embedding_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
            
        return {
            "transcription": transcription,
            "prediction": self.labels[idx.item()],
            "confidence": float(conf.item())
        }
