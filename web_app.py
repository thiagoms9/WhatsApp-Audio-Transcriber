"""
WhatsApp Audio Transcriber Web Interface
A simple, user-friendly web interface for transcribing WhatsApp audio messages
"""

import sys
sys.path.append('src')

import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.zip_extractor import WhatsAppZipExtractor
from src.chat_parser import WhatsAppChatParser
from src.audio_matcher import AudioMatcher
from src.transcriber import AudioTranscriber, WHISPER_AVAILABLE
from src.output_generator import WhatsAppOutputGenerator
from config import OUTPUTS_DIR, UPLOADS_DIR

# Create FastAPI app
app = FastAPI(
    title="WhatsApp Audio Transcriber",
    description="Convert WhatsApp voice messages to text using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Global processing status storage (in production, use a database)
processing_status = {}

# Available models and languages
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
WHISPER_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English", 
    "pt": "Portuguese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese"
}


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main web interface"""
    return get_html_interface()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper_available": WHISPER_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/upload")
async def upload_whatsapp_export(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Form("auto"),
    model: str = Form("base")
):
    """Upload and process WhatsApp export file with custom settings"""
    
    # Validate file
    if not file.filename or not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Please upload a ZIP file")
    
    if not WHISPER_AVAILABLE:
        raise HTTPException(status_code=500, detail="Whisper not available. Please install FFmpeg and openai-whisper")
    
    # Validate language and model
    if language not in WHISPER_LANGUAGES:
        language = "auto"
    
    if model not in WHISPER_MODELS:
        model = "base"
    
    # Generate processing ID
    process_id = f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename.replace('.zip', '')}"
    
    # Save uploaded file
    upload_path = UPLOADS_DIR / f"{process_id}.zip"
    
    try:
        # Save file
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Initialize processing status
        processing_status[process_id] = {
            "status": "uploaded",
            "progress": 0,
            "message": "File uploaded successfully",
            "filename": file.filename,
            "language": language,
            "model": model,
            "started_at": datetime.now().isoformat(),
            "error": None,
            "results": None
        }
        
        # Start background processing with custom settings
        background_tasks.add_task(process_whatsapp_export, process_id, str(upload_path), language, model)
        
        return {
            "success": True,
            "process_id": process_id,
            "message": f"File uploaded. Processing with {WHISPER_LANGUAGES[language]} language and {model} model...",
            "status_url": f"/status/{process_id}"
        }
        
    except Exception as e:
        # Cleanup on error
        if upload_path.exists():
            upload_path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/status/{process_id}")
async def get_processing_status(process_id: str):
    """Get processing status for a given process ID"""
    
    if process_id not in processing_status:
        raise HTTPException(status_code=404, detail="Process not found")
    
    return processing_status[process_id]


@app.get("/download/{process_id}/{file_type}")
async def download_result(process_id: str, file_type: str):
    """Download processed results"""
    
    if process_id not in processing_status:
        raise HTTPException(status_code=404, detail="Process not found")
    
    status = processing_status[process_id]
    
    if status["status"] != "completed" or not status["results"]:
        raise HTTPException(status_code=400, detail="Processing not completed or failed")
    
    # Get the appropriate file
    results = status["results"]
    
    if file_type == "txt" and "txt_file" in results:
        file_path = results["txt_file"]
    elif file_type == "json" and "json_file" in results:
        file_path = results["json_file"]
    elif file_type == "csv" and "csv_file" in results:
        file_path = results["csv_file"]
    elif file_type == "metadata" and "metadata_file" in results:
        file_path = results["metadata_file"]
    else:
        raise HTTPException(status_code=400, detail="Invalid file type or file not available")
    
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    filename = Path(file_path).name
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@app.get("/models")
async def get_available_models():
    """Get available Whisper models and languages"""
    return {
        "models": WHISPER_MODELS,
        "languages": WHISPER_LANGUAGES,
        "whisper_available": WHISPER_AVAILABLE
    }


async def process_whatsapp_export(process_id: str, zip_path: str, language: str = "auto", model: str = "base"):
    """Background task to process WhatsApp export with custom settings"""
    
    def update_status(status: str, progress: int, message: str, error: str = None, results: dict = None):
        processing_status[process_id].update({
            "status": status,
            "progress": progress,
            "message": message,
            "error": error,
            "results": results,
            "updated_at": datetime.now().isoformat()
        })
    
    try:
        # Convert language setting
        whisper_language = None if language == "auto" else language
        
        # Step 1: Extract ZIP
        update_status("extracting", 10, "Extracting ZIP file...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            extractor = WhatsAppZipExtractor(zip_path)
            extract_result = extractor.extract(output_dir=temp_dir)
            
            if not extract_result["success"]:
                update_status("failed", 0, "Extraction failed", extract_result["error"])
                return
            
            # Step 2: Parse chat
            update_status("parsing", 25, f"Parsing chat file... Found {len(extract_result['audio_files'])} audio files")
            
            parser = WhatsAppChatParser(extract_result['chat_file'])
            parse_result = parser.parse()
            
            if not parse_result["success"]:
                update_status("failed", 0, "Chat parsing failed", parse_result["error"])
                return
            
            # Step 3: Match audio files
            update_status("matching", 40, f"Matching audio files to messages... Found {parse_result['audio_messages']} potential audio messages")
            
            chat_audio_messages = parser.get_audio_message_details()
            matcher = AudioMatcher(extract_result['audio_files'], chat_audio_messages)
            match_result = matcher.match_audio_files()
            
            if not match_result["success"]:
                update_status("failed", 0, "Audio matching failed", match_result["error"])
                return
            
            # Step 4: Transcribe audio with enhanced settings
            update_status("transcribing", 60, f"Transcribing {match_result['matches_found']} matched audio files using {model} model with {language} language...")
            
            transcriber = AudioTranscriber(
                model_name=model, 
                language=whisper_language,
                enable_preprocessing=True,
                enable_postprocessing=True
            )
            
            if match_result['matches_found'] > 0:
                transcription_queue = matcher.get_transcription_queue()
                
                # Convert to absolute paths
                for item in transcription_queue:
                    item['audio_file_path'] = str(Path(item['audio_file_path']).absolute())
                
                transcription_result = transcriber.transcribe_queue(transcription_queue)
                
                if not transcription_result["success"]:
                    update_status("failed", 0, "Transcription failed", transcription_result.get("error", "Unknown error"))
                    return
                
                successful_transcriptions = transcription_result['successful_transcriptions']
            else:
                successful_transcriptions = 0
                # Create empty transcriber for output generation
                transcriber = AudioTranscriber(model_name=model, language=whisper_language)
            
            # Step 5: Generate outputs
            update_status("generating", 80, f"Generating output files... Successfully transcribed {successful_transcriptions} audio messages")
            
            generator = WhatsAppOutputGenerator(parser, transcriber, extract_result['chat_file'])
            
            # Generate only TXT first to get accurate summary
            txt_result = generator.generate_output(output_format="txt")
            summary = generator.get_output_summary()
            
            # Generate remaining formats
            results = {}
            results["summary"] = summary
            
            if txt_result["success"]:
                results["txt_file"] = txt_result["output_file"]
                if txt_result.get("metadata_file"):
                    results["metadata_file"] = txt_result["metadata_file"]
            
            # Generate JSON and CSV without affecting summary
            for output_format in ["json", "csv"]:
                result = generator.generate_output(output_format=output_format)
                if result["success"]:
                    results[f"{output_format}_file"] = result["output_file"]
            
            # Add processing settings to results
            results["settings"] = {
                "model": model,
                "language": language,
                "language_display": f"Auto-detect" if language == "auto" else language.upper()
            }
            
            # Complete
            update_status(
                "completed", 
                100, 
                f"Processing completed! Transcribed {successful_transcriptions} audio messages from {summary['total_messages']} total messages.",
                results=results
            )
    
    except Exception as e:
        update_status("failed", 0, "Processing failed due to unexpected error", str(e))
    
    finally:
        # Cleanup uploaded file
        try:
            Path(zip_path).unlink()
        except:
            pass


def get_html_interface():
    """Generate the HTML interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WhatsApp Audio Transcriber</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #25D366 0%, #128C7E 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 600;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .settings {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
        }
        
        .settings h3 {
            color: #128C7E;
            margin-bottom: 20px;
            font-size: 1.3em;
        }
        
        .settings-row {
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        
        .setting-group {
            flex: 1;
            min-width: 200px;
        }
        
        .setting-group label {
            display: block;
            margin-bottom: 5px;
            color: #555;
            font-weight: 500;
        }
        
        .setting-group select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            background: white;
        }
        
        .setting-group select:focus {
            outline: none;
            border-color: #25D366;
        }
        
        .upload-area {
            border: 3px dashed #25D366;
            border-radius: 15px;
            padding: 60px 20px;
            text-align: center;
            background: #f8fff9;
            margin-bottom: 30px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            background: #f0fdf0;
            border-color: #128C7E;
        }
        
        .upload-area.dragover {
            background: #e6ffe6;
            border-color: #128C7E;
        }
        
        .upload-icon {
            font-size: 4em;
            color: #25D366;
            margin-bottom: 20px;
        }
        
        .upload-text {
            font-size: 1.3em;
            color: #333;
            margin-bottom: 15px;
        }
        
        .upload-subtext {
            color: #666;
            font-size: 1em;
        }
        
        .file-input {
            display: none;
        }
        
        .btn {
            background: linear-gradient(135deg, #25D366 0%, #128C7E 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 10px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(37, 211, 102, 0.3);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .progress-container {
            display: none;
            margin-top: 30px;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 15px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #25D366 0%, #128C7E 100%);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .progress-text {
            text-align: center;
            color: #333;
            font-weight: 500;
        }
        
        .results-container {
            display: none;
            margin-top: 30px;
            padding: 30px;
            background: #f8fff9;
            border-radius: 15px;
            border: 2px solid #25D366;
        }
        
        .results-title {
            font-size: 1.5em;
            color: #128C7E;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .download-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            justify-content: center;
            margin-bottom: 20px;
        }
        
        .download-btn {
            background: #fff;
            color: #25D366;
            border: 2px solid #25D366;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .download-btn:hover {
            background: #25D366;
            color: white;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat {
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #25D366;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .error {
            background: #fee;
            border: 2px solid #f66;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            color: #c33;
            text-align: center;
        }
        
        .instructions {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
        }
        
        .instructions h3 {
            color: #128C7E;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .instructions ol {
            margin-left: 20px;
            line-height: 1.6;
        }
        
        .instructions li {
            margin-bottom: 8px;
            color: #555;
        }
        
        @media (max-width: 600px) {
            .container {
                margin: 10px;
                border-radius: 15px;
            }
            
            .header {
                padding: 30px 20px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .content {
                padding: 30px 20px;
            }
            
            .upload-area {
                padding: 40px 15px;
            }
            
            .stats {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .settings-row {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎵 WhatsApp Audio Transcriber</h1>
            <p>Convert your voice messages to text using AI</p>
        </div>
        
        <div class="content">
            <div class="instructions">
                <h3>📱 How to export your WhatsApp chat:</h3>
                <ol>
                    <li>Open WhatsApp and go to the chat you want to transcribe</li>
                    <li>Tap the 3-dot menu → <strong>More</strong> → <strong>Export chat</strong></li>
                    <li>Choose <strong>"Include media"</strong> to get audio files</li>
                    <li>Save the ZIP file and upload it here</li>
                </ol>
            </div>
            
            <div class="settings">
                <h3>⚙️ Transcription Settings:</h3>
                <div class="settings-row">
                    <div class="setting-group">
                        <label for="languageSelect">Language:</label>
                        <select id="languageSelect">
                            <option value="auto">Auto-detect</option>
                            <option value="en">English</option>
                            <option value="pt">Portuguese</option>
                            <option value="es">Spanish</option>
                            <option value="fr">French</option>
                            <option value="de">German</option>
                            <option value="it">Italian</option>
                            <option value="ru">Russian</option>
                            <option value="ja">Japanese</option>
                            <option value="ko">Korean</option>
                            <option value="zh">Chinese</option>
                        </select>
                    </div>
                    <div class="setting-group">
                        <label for="modelSelect">AI Model (accuracy vs speed):</label>
                        <select id="modelSelect">
                            <option value="tiny">Tiny (Fastest, Lower accuracy)</option>
                            <option value="base" selected>Base (Balanced)</option>
                            <option value="small">Small (Good accuracy)</option>
                            <option value="medium">Medium (High accuracy)</option>
                            <option value="large">Large (Best accuracy, Slower)</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📁</div>
                <div class="upload-text">Click to upload your WhatsApp export</div>
                <div class="upload-subtext">Or drag and drop your ZIP file here</div>
                <input type="file" id="fileInput" class="file-input" accept=".zip" onchange="uploadFile()">
            </div>
            
            <div class="progress-container" id="progressContainer">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="progress-text" id="progressText">Processing...</div>
            </div>
            
            <div class="results-container" id="resultsContainer">
                <div class="results-title">🎉 Transcription Complete!</div>
                
                <div class="download-buttons">
                    <a href="#" class="download-btn" id="downloadTxt">📄 Download TXT</a>
                    <a href="#" class="download-btn" id="downloadJson">📊 Download JSON</a>
                    <a href="#" class="download-btn" id="downloadCsv">📈 Download CSV</a>
                    <a href="#" class="download-btn" id="downloadMeta">ℹ️ Download Metadata</a>
                </div>
                
                <div class="stats" id="statsContainer">
                    <!-- Stats will be populated by JavaScript -->
                </div>
            </div>
            
            <div class="error" id="errorContainer" style="display: none;">
                <strong>Error:</strong> <span id="errorMessage"></span>
            </div>
        </div>
    </div>

    <script>
        let currentProcessId = null;
        let statusInterval = null;

        // Drag and drop functionality
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            if (fileInput.files.length > 0) {
                handleFile(fileInput.files[0]);
            }
        }

        async function handleFile(file) {
            if (!file.name.endsWith('.zip')) {
                showError('Please select a ZIP file');
                return;
            }

            hideError();
            showProgress();

            const formData = new FormData();
            formData.append('file', file);
            
            // Add settings
            const language = document.getElementById('languageSelect').value;
            const model = document.getElementById('modelSelect').value;
            formData.append('language', language);
            formData.append('model', model);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Upload failed');
                }

                const result = await response.json();
                currentProcessId = result.process_id;
                
                // Start polling for status
                statusInterval = setInterval(checkStatus, 2000);
                
            } catch (error) {
                hideProgress();
                showError(error.message);
            }
        }

        async function checkStatus() {
            if (!currentProcessId) return;

            try {
                const response = await fetch(`/status/${currentProcessId}`);
                if (!response.ok) {
                    throw new Error('Status check failed');
                }

                const status = await response.json();
                updateProgress(status);

                if (status.status === 'completed') {
                    clearInterval(statusInterval);
                    showResults(status);
                } else if (status.status === 'failed') {
                    clearInterval(statusInterval);
                    hideProgress();
                    showError(status.error || 'Processing failed');
                }

            } catch (error) {
                clearInterval(statusInterval);
                hideProgress();
                showError('Status check failed: ' + error.message);
            }
        }

        function updateProgress(status) {
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            
            progressFill.style.width = status.progress + '%';
            progressText.textContent = status.message;
        }

        function showResults(status) {
            hideProgress();
            
            const resultsContainer = document.getElementById('resultsContainer');
            const statsContainer = document.getElementById('statsContainer');
            
            // Update download links
            const processId = currentProcessId;
            document.getElementById('downloadTxt').href = `/download/${processId}/txt`;
            document.getElementById('downloadJson').href = `/download/${processId}/json`;
            document.getElementById('downloadCsv').href = `/download/${processId}/csv`;
            document.getElementById('downloadMeta').href = `/download/${processId}/metadata`;
            
            // Show stats if available
            if (status.results && status.results.summary) {
                const summary = status.results.summary;
                statsContainer.innerHTML = `
                    <div class="stat">
                        <div class="stat-number">${summary.total_messages || 0}</div>
                        <div class="stat-label">Total Messages</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">${summary.audio_messages || 0}</div>
                        <div class="stat-label">Audio Messages</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">${summary.successfully_transcribed || 0}</div>
                        <div class="stat-label">Transcribed</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number">${Math.round((summary.transcription_rate || 0) * 100)}%</div>
                        <div class="stat-label">Success Rate</div>
                    </div>
                `;
            }
            
            resultsContainer.style.display = 'block';
        }

        function showProgress() {
            document.getElementById('progressContainer').style.display = 'block';
            document.getElementById('resultsContainer').style.display = 'none';
        }

        function hideProgress() {
            document.getElementById('progressContainer').style.display = 'none';
        }

        function showError(message) {
            const errorContainer = document.getElementById('errorContainer');
            const errorMessage = document.getElementById('errorMessage');
            
            errorMessage.textContent = message;
            errorContainer.style.display = 'block';
        }

        function hideError() {
            document.getElementById('errorContainer').style.display = 'none';
        }
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    print("🌐 Starting WhatsApp Audio Transcriber Web Interface...")
    print("📱 Access the interface at: http://localhost:8000")
    print("🔄 Processing status will update in real-time")
    print("📁 Output files will be available for download")
    print("\n⚠️  Make sure you have FFmpeg installed for audio processing!")
    print("✨ Ready to transcribe WhatsApp audio messages!\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)