# =============================================================================
# DEEP TEXT ANALYZER - FastAPI APPLICATION
#
# This single script:
# 1. Loads the pre-trained PyTorch model (from your main.py).
# 2. Serves a modern, visually appealing web interface using FastAPI.
# 3. Provides a JSON API endpoint for analysis.
#
# To Run:
# 1. Make sure you have a clean virtual environment:
#    (Delete any old 'venv' folders)
#    python -m venv venv
#    source venv/bin/activate  (On Mac/Linux)
#    .\venv\Scripts\activate   (On Windows)
#
# 2. Install all requirements correctly:
#    pip install -r requirements.txt
#
# 3. Run this app:
#    python app.py
#
# 4. Open your browser and go to:
#    http://127.0.0.1:8000
# =============================================================================

import torch
import torch.nn as nn
import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModel
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import os

# --- CONFIGURATION & SETUP (Copied from main.py) ---
# Ensure your model files are in the same directory as this script
class Config:
    MODEL_PATH = "9350_model.pth"
    TOKENIZER_NAME = "xlm-roberta-base"
    ENCODERS_PATH = "Encoders_9350.pkl"
    MAX_LEN = 128

# --- Define Model Architecture (Must be identical to training script) ---
class GrandUnifiedModel(nn.Module):
    def __init__(self, base_model, num_classes_per_task):
        super(GrandUnifiedModel, self).__init__()
        self.base = base_model
        hidden_size = self.base.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifiers = nn.ModuleDict({task: nn.Linear(hidden_size, num_classes) for task, num_classes in num_classes_per_task.items()})

    def forward(self, input_ids, attention_mask):
        # Use .last_hidden_state and take the [CLS] token's output (index 0)
        # .pooler_output is often not as effective for sentence-level tasks.
        last_hidden_state = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_token_output = last_hidden_state[:, 0, :]
        pooled_output = self.dropout(cls_token_output)
        return {task: classifier(pooled_output) for task, classifier in self.classifiers.items()}

# --- Global variables to hold the loaded assets ---
# We use a dictionary to hold them, which FastAPI likes
model_assets = {
    "device": None,
    "tokenizer": None,
    "label_encoders": None,
    "model": None
}

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Deep Text Analyzer API",
    description="A FastAPI server for the multilingual text analysis model."
)

# --- Pydantic Models for Request/Response ---
class TextInput(BaseModel):
    text: str

# --- Model Loading (at startup) ---
@app.on_event("startup")
def load_all_assets():
    """Loads the tokenizer, encoders, and the trained model into memory on server start."""
    print("Server starting... Loading all assets...")
    
    # Check if files exist
    if not os.path.exists(Config.MODEL_PATH):
        print(f"ERROR: Model file not found at {Config.MODEL_PATH}")
        return
    if not os.path.exists(Config.ENCODERS_PATH):
        print(f"ERROR: Encoders file not found at {Config.ENCODERS_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_assets["device"] = device
    print(f"Using device: {device}")

    try:
        model_assets["tokenizer"] = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)
        
        with open(Config.ENCODERS_PATH, 'rb') as f:
            model_assets["label_encoders"] = pickle.load(f)
        print("✅ Label encoders loaded.")

        num_classes_per_task = {col: len(le.classes_) for col, le in model_assets["label_encoders"].items()}
        base_model = AutoModel.from_pretrained(Config.TOKENIZER_NAME)
        model = GrandUnifiedModel(base_model, num_classes_per_task)

        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        model_assets["model"] = model
        
        print("✅ FINAL Model loaded successfully! Server is ready.")
    
    except Exception as e:
        print(f"FATAL ERROR during model loading: {e}")
        print("The server will run, but /analyze calls will fail.")


# --- Prediction Function (Adapted from main.py) ---
def predict_text(text: str):
    """Takes raw text and returns a dictionary of model predictions."""
    
    # Check if assets are loaded
    if not all(model_assets.values()):
        print("Error: Model assets are not loaded. Was there an error on startup?")
        raise HTTPException(status_code=500, detail="Model assets are not loaded. Check server logs.")

    # Get assets from the global dict
    device = model_assets["device"]
    tokenizer = model_assets["tokenizer"]
    label_encoders = model_assets["label_encoders"]
    model = model_assets["model"]

    encoding = tokenizer(text, add_special_tokens=True, max_length=Config.MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        logits_dict = model(encoding['input_ids'].to(device), encoding['attention_mask'].to(device))
    
    predictions = {
        task: label_encoders[task].inverse_transform([torch.argmax(logits, dim=1).item()])[0]
        for task, logits in logits_dict.items()
    }
    
    # Format the output beautifully
    return {
        "text": text,
        "core_analysis": {
            "sentiment": predictions.get("sentiment_polarity", "N/A"),
            "primary_emotion": predictions.get("emotion", "N/A"),
            "emotional_tone": predictions.get("emotional_tone", "N/A")
        },
        "pragmatics_and_intent": {
            "intent": predictions.get("intent", "N/A")
        },
        "stylistic_analysis": {
            "formality": predictions.get("formality", "N/A"),
            "is_sarcastic": predictions.get("is_sarcastic", "N/A")
        },
        "safety_and_toxicity": {
            "is_toxic": predictions.get("is_toxic", "N/A"),
            "has_hidden_agenda": predictions.get("has_hidden_agenda", "N/A")
        }
    }


# --- API Endpoint for Analysis ---
@app.post("/analyze")
async def analyze_text_endpoint(data: TextInput):
    """
    API endpoint to analyze text.
    Receives: {"text": "some text"}
    Returns: JSON with analysis
    """
    try:
        predictions = predict_text(data.text)
        return predictions
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        # Catch any other unexpected errors during prediction
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

# --- HTML/CSS/JS for the Visually Appealing Frontend ---
# We serve this from the root URL ("/")
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Text Analyzer</title>
    <!-- Load Tailwind CSS from CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Use a modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body {
            font-family: 'Inter', sans-serif;
        }
        /* Simple spinner */
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 24px;
            height: 24px;
            border-radius: 50%;
            border-left-color: #fff;
            animation: spin 1s ease infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100 min-h-screen flex items-center justify-center p-4">

    <div class="bg-white dark:bg-gray-800 shadow-2xl rounded-lg p-8 w-full max-w-2xl">
        
        <h1 class="text-3xl font-bold text-center text-blue-600 dark:text-blue-400 mb-6">
            Deep Text Analyzer
        </h1>
        
        <p class="text-center text-gray-600 dark:text-gray-300 mb-6">
            Enter any text to get a multi-layered analysis from the AI model.
        </p>

        <!-- Input Area -->
        <div class="mb-4">
            <label for="text-input" class="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-2">Your Text:</label>
            <textarea id="text-input" rows="6" class="w-full p-3 rounded-md shadow-sm border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition" placeholder="Type or paste your text here... e.g., 'This is an amazing product!' or 'Qué mal servicio.'"></textarea>
        </div>

        <!-- Analyze Button -->
        <button id="analyze-button" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-md transition duration-300 ease-in-out flex items-center justify-center space-x-2">
            <span id="button-text">Analyze Text</span>
            <div id="loading-spinner" class="spinner hidden"></div>
        </button>

        <!-- Results Area -->
        <div id="results-container" class="mt-8 hidden">
            <h2 class="text-2xl font-semibold mb-4">Analysis Results:</h2>
            <pre id="results-output" class="bg-gray-900 dark:bg-gray-950 text-white text-sm p-4 rounded-md overflow-x-auto"></pre>
        </div>
        
        <!-- Error Area -->
        <div id="error-container" class="mt-6 hidden">
            <div class="bg-red-100 dark:bg-red-900 border border-red-400 dark:border-red-700 text-red-700 dark:text-red-200 px-4 py-3 rounded-md" role="alert">
                <strong class="font-bold">Error:</strong>
                <span id="error-message" class="block sm:inline"></span>
            </div>
        </div>

    </div>

    <script>
        const textInput = document.getElementById('text-input');
        const analyzeButton = document.getElementById('analyze-button');
        const buttonText = document.getElementById('button-text');
        const loadingSpinner = document.getElementById('loading-spinner');
        
        const resultsContainer = document.getElementById('results-container');
        const resultsOutput = document.getElementById('results-output');
        
        const errorContainer = document.getElementById('error-container');
        const errorMessage = document.getElementById('error-message');

        analyzeButton.addEventListener('click', async () => {
            const text = textInput.value.trim();
            if (!text) {
                showError('Please enter some text to analyze.');
                return;
            }

            // --- Set loading state ---
            hideError();
            resultsContainer.classList.add('hidden');
            analyzeButton.disabled = true;
            buttonText.textContent = 'Analyzing...';
            loadingSpinner.classList.remove('hidden');

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();

                if (!response.ok) {
                    // Handle errors from the API (e.g., 500 server error)
                    throw new Error(data.detail || 'An unknown error occurred.');
                }

                // --- Show results ---
                resultsOutput.textContent = JSON.stringify(data, null, 2); // Pretty-print JSON
                resultsContainer.classList.remove('hidden');

            } catch (error) {
                // Handle network errors or errors thrown
                console.error('Analysis error:', error);
                showError(error.message);
            } finally {
                // --- Reset loading state ---
                analyzeButton.disabled = false;
                buttonText.textContent = 'Analyze Text';
                loadingSpinner.classList.add('hidden');
            }
        });

        function showError(message) {
            errorMessage.textContent = message;
            errorContainer.classList.remove('hidden');
        }

        function hideError() {
            errorContainer.classList.add('hidden');
        }
    </script>

</body>
</html>
"""
# --- API Endpoint for the Frontend ---
@app.get("/", response_class=HTMLResponse)
async def get_root_page():
    """Serves the main HTML web page."""
    return HTMLResponse(content=html_content)


# --- Main entry point to run the server ---
if __name__ == "__main__":
    # This block allows you to run the app directly with: python app.py
    # It will start the Uvicorn server on http://127.0.0.1:8000
    print("Starting FastAPI server...")
    print("Find your app at http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)