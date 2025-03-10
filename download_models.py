import os
import requests

# Create directory for models
os.makedirs('models', exist_ok=True)

# Download WavTokenizer config and weights
def download_file(url, destination):
    print(f"Downloading {url} to {destination}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Download complete: {destination}")

# Download the required model files
download_file(
    "https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
    "wavtokenizer_config.yaml"
)

download_file(
    "https://huggingface.co/novateur/WavTokenizer-large-speech-75token/resolve/main/wavtokenizer_large_speech_320_24k.ckpt",
    "wavtokenizer_model.ckpt"
)

print("Model files downloaded successfully!")