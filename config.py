"""
Configuration settings for WhatsApp Transcriber
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TEMP_DIR = PROJECT_ROOT / "temp"

# Create directories if they don't exist
for dir_path in [UPLOADS_DIR, OUTPUTS_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# Whisper settings
WHISPER_MODEL = "base"  # tiny, base, small, medium, large, large-v2, large-v3
WHISPER_LANGUAGE = None  # None for auto-detection, or specify like "en", "pt", "es"
WHISPER_TASK = "transcribe"  # "transcribe" or "translate"

# Available Whisper models with descriptions
WHISPER_MODELS = {
    "tiny": {"size": "39 MB", "speed": "~32x", "accuracy": "Basic", "description": "Fastest, least accurate"},
    "base": {"size": "74 MB", "speed": "~16x", "accuracy": "Good", "description": "Default choice, balanced"},
    "small": {"size": "244 MB", "speed": "~6x", "accuracy": "Better", "description": "Good accuracy, slower"},
    "medium": {"size": "769 MB", "speed": "~2x", "accuracy": "Very Good", "description": "High accuracy"},
    "large": {"size": "1550 MB", "speed": "~1x", "accuracy": "Best", "description": "Highest accuracy, slowest"},
    "large-v2": {"size": "1550 MB", "speed": "~1x", "accuracy": "Best", "description": "Latest large model"},
    "large-v3": {"size": "1550 MB", "speed": "~1x", "accuracy": "Best", "description": "Newest, most accurate"}
}

# Supported languages for Whisper
WHISPER_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "pt": "Portuguese", 
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish"
}

# Audio settings
SUPPORTED_AUDIO_FORMATS = [".opus", ".m4a", ".aac", ".mp3", ".wav", ".ogg"]
AUDIO_SAMPLE_RATE = 16000

# WhatsApp chat patterns
AUDIO_PATTERNS = [
    r"<audio omitted>",
    r"<áudio omitido>", 
    r"<audio omitido>",
    r"<Media omitted>",
    r"<media omitted>",
    r"<arquivo de mídia oculto>",
    r"PTT-\d{8}-WA\d+\.opus \(file attached\)",  # New format: PTT-20250329-WA0003.opus (file attached)
    r".\.opus \(file attached\)",  # Any .opus file attached
    r".\.(aac|m4a|mp3|wav|ogg) \(file attached\)",  # Other audio formats
]

# Chat parsing settings - support multiple international formats
DATETIME_FORMATS = [
    # US formats (most common in exports)
    "%m/%d/%y, %H:%M",     # 8/19/24, 17:11
    "%m/%d/%Y, %H:%M",     # 8/19/2024, 17:11
    "%m/%d/%Y, %I:%M %p",  # 8/19/2024, 5:11 PM
    "%m/%d/%y, %I:%M %p",  # 8/19/24, 5:11 PM
    
    # European formats
    "%d/%m/%Y, %H:%M",     # 19/8/2024, 17:11
    "%d/%m/%y, %H:%M",     # 19/8/24, 17:11
    "%d.%m.%Y, %H:%M",     # 19.8.2024, 17:11 (German)
    "%d.%m.%y, %H:%M",     # 19.8.24, 17:11
    
    # ISO and other formats
    "%Y-%m-%d, %H:%M",     # 2024-8-19, 17:11
    "%d-%m-%Y, %H:%M",     # 19-8-2024, 17:11
    
    # With seconds
    "%m/%d/%y, %H:%M:%S",  # 8/19/24, 17:11:30
    "%d/%m/%Y, %H:%M:%S",  # 19/8/2024, 17:11:30
    "%m/%d/%Y, %I:%M:%S %p", # 8/19/2024, 5:11:30 PM
]

# Output settings
OUTPUT_FORMAT = "txt"  # "txt", "json", "csv"
INCLUDE_METADATA = True
PRESERVE_ORIGINAL_FORMAT = True

# Performance settings
MAX_FILE_SIZE_MB = 500
MAX_AUDIO_DURATION_MINUTES = 10
BATCH_SIZE = 5  # Number of audio files to process simultaneously

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "transcriber.log"