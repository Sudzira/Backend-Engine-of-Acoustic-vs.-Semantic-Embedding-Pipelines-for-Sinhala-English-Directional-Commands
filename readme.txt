Congratulations on getting the GitHub MLOps pipeline set up! That is a massive milestone for your project.

Now it is time to bring the code down to your Intel i5 laptop and run the ultimate stress test to prepare for your viva. Here is exactly how to spin up your testing environment locally.

*here we have already done step 1 and step 2 so u can skip these below parts

### Step 1: Clone the Code to Your Laptop

Open **Command Prompt** or **PowerShell** on your Windows machine, navigate to where you want to save the project (e.g., your Documents folder), and run:

```bash
git clone https://github.com/your-username/directional-command-recognition.git
cd directional-command-recognition

```

*(Replace `your-username` with your actual GitHub username).*

### Step 2: Slot in the Model Weights

Find the `best_model.pth` files you downloaded from your Google Drive.
Move them into the `models/` folder inside your newly cloned project directory.

**Crucial:** Rename them so they match exactly what `src/server.py` is looking for:

* Rename your Whisper MLP weight to: `mlp_asr_best.pt`
* Rename your DistilHuBERT MLP weight to: `mlp_acoustic_best.pt`

*here we have already done step 1 and step 2 so u can skip these above parts

### Step 3: Create a Clean Virtual Environment (Mandatory)

Because your laptop has 8GB of RAM, installing global Python packages can cause version conflicts and memory bloat. A virtual environment keeps it clean.
Run this in your terminal:

```bash
python -m venv venv

```

*here naming new environment name it as dashboard_python ont as venv

Activate it (for Windows):

```bash
venv\Scripts\activate

```
*here same as above about the name of the environment 

*(You should see `(venv)` appear at the start of your terminal line).*

### Step 4: Install the Deep Learning Libraries

With your virtual environment active, install the dependencies we listed in your `requirements.txt`:

```bash
pip install -r requirements.txt

```
*here check for correct requirements.txt file in above case

*Note: This will download PyTorch, Faster-Whisper, and Hugging Face Transformers. It may take 5–10 minutes depending on your internet speed.*

### Step 5: Start the AI Server

Make sure you are in the root folder of your project (`directional-command-recognition`), then start the backend:

```bash
uvicorn src.server:app --port 8000

```

**Watch your terminal carefully.** You will see it initializing the models. On your i5 processor, loading Whisper and DistilHuBERT into RAM might take **15 to 30 seconds**. Wait until the terminal explicitly says `"Models loaded successfully"` and `"Uvicorn running on http://127.0.0.1:8000"`.

### Step 6: Launch the Dashboard and Test!

You do not need a web server for the frontend.

1. Open your File Explorer.
2. Navigate to the `frontend/` folder.
3. **Double-click `index.html**` to open it in Chrome, Firefox, or Edge.
4. Click the **"Hold to Speak"** button. Your browser will ask for microphone permissions—click **Allow**.
5. Speak a command (e.g., "Forward" or its Sinhala equivalent).
6. Release the button and watch the dashboard!

### 🚨 Live-Test Troubleshooting for your Hardware:

* **The "RAM Crash":** Keep Windows Task Manager open on the "Performance" tab during your first test. If your RAM hits 99% and the server crashes, let me know. We will need to enforce stricter `INT8` quantization on the HuBERT model.
* **Microphone Silent?** If the server receives audio but outputs weird results, check your Windows sound settings to ensure your default microphone is active and not muted.

Run this test and let me know how the visual "race" between the two robots looks on your screen!