from flask import Flask, request, jsonify, send_file
import os
import torch
import numpy as np
import torchaudio
import tempfile
from transformers import AutoModelForCausalLM
from outetts.wav_tokenizer.decoder import WavTokenizer
from yarngpt.audiotokenizer import AudioTokenizer
import time

app = Flask(__name__)

# Configure paths for model files
MODEL_PATH = "saheedniyi/YarnGPT"
WAV_TOKENIZER_CONFIG_PATH = "wavtokenizer_config.yaml"
WAV_TOKENIZER_MODEL_PATH = "wavtokenizer_model.ckpt"

# Initialize global variables for model and tokenizer
audio_tokenizer = None
model = None

def load_models():
    global audio_tokenizer, model
    print("Loading YarnGPT models...")
    
    # Create the AudioTokenizer object
    audio_tokenizer = AudioTokenizer(
        MODEL_PATH, 
        WAV_TOKENIZER_MODEL_PATH, 
        WAV_TOKENIZER_CONFIG_PATH
    )
    
    # Load the model weights
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype="auto"
    ).to(audio_tokenizer.device)
    
    print("Models loaded successfully!")

@app.route("/")
def index():
    return jsonify({"status": "YarnGPT API is running"})

@app.route("/api/tts", methods=["POST"])
def text_to_speech():
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
    
    try:
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
        
        # Return the audio file
        return send_file(
            output_path, 
            mimetype="audio/wav", 
            as_attachment=True,
            download_name=f"yarngpt_{int(time.time())}.wav"
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/speakers", methods=["GET"])
def get_speakers():
    """Return the list of available speakers"""
    speakers = ["idera", "emma", "jude", "osagie", "tayo", 
               "zainab", "joke", "regina", "remi", "umar", "chinenye"]
    return jsonify({"speakers": speakers})

# Initialize the model on startup
@app.before_first_request
def initialize():
    load_models()

if __name__ == "__main__":
    # For local development
    if not (audio_tokenizer and model):
        load_models()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))