from flask import Flask, request, jsonify, send_file
import os
import torch
import numpy as np
import torchaudio
import tempfile
from transformers import AutoModelForCausalLM
import time
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure paths for model files
MODEL_PATH = "saheedniyi/YarnGPT"
WAV_TOKENIZER_CONFIG_PATH = os.path.join(os.getcwd(), "wavtokenizer_config.yaml")
WAV_TOKENIZER_MODEL_PATH = os.path.join(os.getcwd(), "wavtokenizer_model.ckpt")

# Initialize global variables for model and tokenizer
audio_tokenizer = None
model = None

def download_model_files():
    """Download necessary model files if they don't exist."""
    logger.info("Checking for model files...")
    
    # Check if files already exist
    if not os.path.exists(WAV_TOKENIZER_CONFIG_PATH):
        logger.info(f"Downloading config file to {WAV_TOKENIZER_CONFIG_PATH}")
        download_file(
            "https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
            WAV_TOKENIZER_CONFIG_PATH
        )
    
    if not os.path.exists(WAV_TOKENIZER_MODEL_PATH):
        logger.info(f"Downloading model file to {WAV_TOKENIZER_MODEL_PATH}")
        download_file(
            "https://huggingface.co/novateur/WavTokenizer-large-speech-75token/resolve/main/wavtokenizer_large_speech_320_24k.ckpt",
            WAV_TOKENIZER_MODEL_PATH
        )
    
    logger.info("Model files ready")

def download_file(url, destination):
    """Download a file from URL to destination path."""
    logger.info(f"Downloading {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Download complete: {destination}")
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise

def load_models():
    """Load YarnGPT models and tokenizer."""
    global audio_tokenizer, model
    
    try:
        logger.info("Loading YarnGPT models...")
        
        # First ensure model files are downloaded
        download_model_files()
        
        # Import here to avoid import errors before installation
        from yarngpt.audiotokenizer import AudioTokenizer
        
        # Create the AudioTokenizer object
        audio_tokenizer = AudioTokenizer(
            MODEL_PATH, 
            WAV_TOKENIZER_MODEL_PATH, 
            WAV_TOKENIZER_CONFIG_PATH
        )
        
        # Load the model weights
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else "auto"
        ).to(audio_tokenizer.device)
        
        logger.info(f"Models loaded successfully! Using device: {audio_tokenizer.device}")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

@app.route("/")
def index():
    """Health check endpoint."""
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        "status": "YarnGPT API is running",
        "model_status": model_status
    })

@app.route("/load_models", methods=["POST"])
def load_models_endpoint():
    """Endpoint to manually trigger model loading."""
    try:
        load_models()
        return jsonify({"status": "Models loaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/tts", methods=["POST"])
def text_to_speech():
    """Generate speech from text using YarnGPT."""
    global audio_tokenizer, model
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    # Check required fields
    if "text" not in data:
        return jsonify({"error": "Text is required"}), 400
    
    text = data["text"]
    speaker = data.get("speaker", "idera")  # Default to idera if not specified
    temperature = float(data.get("temperature", 0.1))
    repetition_penalty = float(data.get("repetition_penalty", 1.1))
    
    # Validate speaker
    valid_speakers = ["idera", "emma", "jude", "osagie", "tayo", 
                     "zainab", "joke", "regina", "remi", "umar", "chinenye"]
    if speaker not in valid_speakers:
        return jsonify({"error": f"Invalid speaker. Choose from: {', '.join(valid_speakers)}"}), 400
    
    # Check if models are loaded
    if audio_tokenizer is None or model is None:
        try:
            load_models()
        except Exception as e:
            return jsonify({"error": f"Failed to load models: {str(e)}"}), 500
    
    try:
        logger.info(f"Processing TTS request: {len(text)} chars with speaker {speaker}")
        
        # Create prompt from input text
        prompt = audio_tokenizer.create_prompt(text, speaker)
        
        # Tokenize the prompt
        input_ids = audio_tokenizer.tokenize_prompt(prompt)
        
        # Generate output from the model
        output = model.generate(
            input_ids=input_ids,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_length=4000,
        )
        
        # Convert the output to "audio codes"
        codes = audio_tokenizer.get_codes(output)
        
        # Convert the codes to audio
        audio = audio_tokenizer.get_audio(codes)
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name
            torchaudio.save(output_path, audio, sample_rate=24000)
        
        logger.info(f"Audio generated successfully, size: {os.path.getsize(output_path)} bytes")
        
        # Return the audio file
        return send_file(
            output_path, 
            mimetype="audio/wav", 
            as_attachment=True,
            download_name=f"yarngpt_{int(time.time())}.wav"
        )
    
    except Exception as e:
        import traceback
        logger.error(f"Error generating audio: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/api/speakers", methods=["GET"])
def get_speakers():
    """Return the list of available speakers."""
    speakers = ["idera", "emma", "jude", "osagie", "tayo", 
               "zainab", "joke", "regina", "remi", "umar", "chinenye"]
    return jsonify({"speakers": speakers})

if __name__ == "__main__":
    # For local development
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Try to load models at startup for local development
    try:
        load_models()
    except Exception as e:
        logger.error(f"Error loading models at startup: {e}")
        logger.info("Models will be loaded on the first request")
    
    app.run(host=host, port=port)