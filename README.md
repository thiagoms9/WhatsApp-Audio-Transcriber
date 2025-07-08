# WhatsApp Audio Transcriber

🎵 Convert WhatsApp voice messages to text using AI (OpenAI Whisper)

## Features

- Upload WhatsApp chat exports (ZIP files)
- Automatic audio file matching to chat messages
- AI-powered transcription using Whisper models
- Multiple output formats (TXT, JSON, CSV)
- Real-time processing status
- Multiple language support
- Configurable AI models (speed vs accuracy)

## Live Demo

🌐 **[Try it here](your-railway-url-here)** (will be updated after deployment)

## How to Use

1. **Export your WhatsApp chat:**
   - Open WhatsApp → Select chat
   - Menu (3 dots) → More → Export chat
   - Choose "Include media"
   - Save the ZIP file

2. **Upload and transcribe:**
   - Upload your ZIP file
   - Choose language and AI model
   - Wait for processing
   - Download transcribed results

## Technology Stack

- **Backend**: FastAPI (Python)
- **AI**: OpenAI Whisper
- **Frontend**: HTML/CSS/JavaScript
- **Deployment**: Railway

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python web_app.py

# Open http://localhost:8000
```

## Project Structure

```
├── web_app.py              # Main FastAPI application
├── src/
│   ├── zip_extractor.py    # WhatsApp ZIP extraction
│   ├── chat_parser.py      # Chat message parsing
│   ├── audio_matcher.py    # Audio-to-message matching
│   ├── transcriber.py      # AI transcription
│   ├── output_generator.py # Output file generation
│   └── config.py          # Configuration
├── requirements.txt        # Python dependencies
├── railway.toml           # Railway deployment config
└── README.md              # This file
```

## License

MIT License - Feel free to use and modify!