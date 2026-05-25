# Directional Command Recognition (Sinhala-English)

A real-time AI dashboard for recognizing directional commands (Forward, Backward, Left, Right, Stop) in code-mixed Sinhala-English speech. This project compares two different architectural approaches:

1.  **ASR-Semantic Pipeline**: Transcribes audio with Whisper-Small (Fine-tuned), generates semantic embeddings with LaBSE, and classifies intent using an MLP with Residual Blocks.
2.  **Acoustic-Only Pipeline**: Processes raw audio features directly using DistilHuBERT and classifies them using a lightweight MLP.

## 🚀 Features
- **Real-time Processing**: WebSocket-based communication for low-latency feedback.
- **Dual Pipeline Comparison**: Compare accuracy and latency between ASR-based and Direct-Acoustic approaches.
- **Visual Dashboard**: Animated robot/object that moves according to detected commands.
- **Sinhala-English Support**: Robust handling of code-mixed commands.

## 📁 Project Structure
```
directional-command-recognition/
├── frontend/
│   └── index.html          # Standalone dashboard
├── models/
│   ├── asr_classifier.pt   # ASR-Semantic MLP weights
│   └── acoustic_classifier.pt # Acoustic MLP weights
├── src/
│   ├── pipelines/
│   │   ├── asr_pipeline.py
│   │   └── acoustic_pipeline.py
│   └── server.py           # FastAPI WebSocket Server
└── requirements.txt
```

## 🛠️ Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/directional-command-recognition.git
    cd directional-command-recognition
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Server**:
    ```bash
    python src/server.py
    ```

4.  **Open the Dashboard**:
    Open `frontend/index.html` in your web browser.

## 🧠 Model Details

### Pipeline A: ASR + Semantic (Whisper + LaBSE)
- **ASR**: `Subhaka/whisper-small-Sinhala-Fine_Tune`
- **Embedding**: `sentence-transformers/LaBSE` (768D)
- **Classifier**: MLP with 2 Residual Blocks (Input: 768, Hidden: 256, Output: 5)

### Pipeline B: Direct Acoustic (DistilHuBERT)
- **Feature Extractor**: `ntu-spml/distilhubert`
- **Classifier**: MLP with GELU activation (Input: 768, Hidden: 256, Output: 5)

## 📊 Evaluation
Detailed training history and evaluation metrics (Confusion Matrices, Learning Curves) can be found in the `Documentation/` and `Trained data/` folders of the main repository.
