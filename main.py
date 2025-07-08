"""
WhatsApp Audio Transcriber - Main Entry Point

This script provides both CLI and web interface options for transcribing WhatsApp audio messages.

Usage:
    python main.py                          # Start web interface
    python main.py web                      # Start web interface  
    python main.py cli <zip_file>           # CLI processing
    python main.py --help                   # Show help
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

def start_web_interface():
    """Start the web interface"""
    print("🌐 Starting WhatsApp Audio Transcriber Web Interface...")
    print("📱 The web interface will open at: http://localhost:8000")
    print("🔄 You can upload ZIP files and track processing in real-time")
    print("📁 Download transcribed results in multiple formats")
    print("\n⚠️  Requirements:")
    print("   - FFmpeg must be installed and accessible")
    print("   - Internet connection for downloading Whisper models (first run)")
    print("\n✨ Starting server...\n")
    
    try:
        import uvicorn
        from web_app import app
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except ImportError:
        print("❌ FastAPI/Uvicorn not installed. Install with:")
        print("   pip install fastapi uvicorn")
    except Exception as e:
        print(f"❌ Failed to start web interface: {e}")


def process_cli(zip_file_path: str, output_format: str = "txt"):
    """Process WhatsApp export via command line"""
    
    from src.zip_extractor import WhatsAppZipExtractor
    from src.chat_parser import WhatsAppChatParser
    from src.audio_matcher import AudioMatcher
    from src.transcriber import AudioTranscriber, WHISPER_AVAILABLE
    from src.output_generator import WhatsAppOutputGenerator
    import tempfile
    import shutil
    
    zip_path = Path(zip_file_path)
    
    if not zip_path.exists():
        print(f"❌ File not found: {zip_file_path}")
        return False
    
    if not zip_path.suffix.lower() == '.zip':
        print(f"❌ File must be a ZIP file: {zip_file_path}")
        return False
    
    if not WHISPER_AVAILABLE:
        print("❌ Whisper not available. Please install FFmpeg and openai-whisper:")
        print("   pip install openai-whisper")
        print("   # Also install FFmpeg from https://ffmpeg.org/")
        return False
    
    print(f"🎵 Processing WhatsApp export: {zip_path.name}")
    print("=" * 60)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Extract ZIP
            print("1️⃣ Extracting ZIP file...")
            extractor = WhatsAppZipExtractor(str(zip_path))
            extract_result = extractor.extract(output_dir=temp_dir)
            
            if not extract_result["success"]:
                print(f"❌ Extraction failed: {extract_result['error']}")
                return False
            
            print(f"✅ Found {len(extract_result['audio_files'])} audio files")
            
            # Step 2: Parse chat
            print("2️⃣ Parsing chat messages...")
            parser = WhatsAppChatParser(extract_result['chat_file'])
            parse_result = parser.parse()
            
            if not parse_result["success"]:
                print(f"❌ Chat parsing failed: {parse_result['error']}")
                return False
            
            print(f"✅ Parsed {parse_result['total_messages']} messages, {parse_result['audio_messages']} potential audio")
            
            # Step 3: Match audio files
            print("3️⃣ Matching audio files to messages...")
            chat_audio_messages = parser.get_audio_message_details()
            matcher = AudioMatcher(extract_result['audio_files'], chat_audio_messages)
            match_result = matcher.match_audio_files()
            
            if not match_result["success"]:
                print(f"❌ Audio matching failed: {match_result['error']}")
                return False
            
            print(f"✅ Matched {match_result['matches_found']} audio files using {match_result['method_used']}")
            
            # Step 4: Transcribe audio
            print("4️⃣ Transcribing audio messages...")
            transcriber = AudioTranscriber(model_name="base", language=None)
            
            if match_result['matches_found'] > 0:
                transcription_queue = matcher.get_transcription_queue()
                
                # Convert to absolute paths
                for item in transcription_queue:
                    item['audio_file_path'] = str(Path(item['audio_file_path']).absolute())
                
                transcription_result = transcriber.transcribe_queue(transcription_queue)
                
                if not transcription_result["success"]:
                    print(f"❌ Transcription failed: {transcription_result.get('error', 'Unknown error')}")
                    return False
                
                successful = transcription_result['successful_transcriptions']
                total = transcription_result['total_files']
                print(f"✅ Successfully transcribed {successful}/{total} audio files")
                
                # Show transcription previews
                for result in transcriber.transcription_results:
                    if result.status == "success":
                        filename = Path(result.audio_file_path).name
                        preview = result.transcription_text[:60] + "..." if len(result.transcription_text) > 60 else result.transcription_text
                        print(f"   🎵 {filename}: \"{preview}\"")
            else:
                print("⚠️  No audio files to transcribe")
            
            # Step 5: Generate output
            print("5️⃣ Generating output files...")
            generator = WhatsAppOutputGenerator(parser, transcriber, extract_result['chat_file'])
            
            result = generator.generate_output(output_format=output_format)
            
            if not result["success"]:
                print(f"❌ Output generation failed: {result['error']}")
                return False
            
            output_file = Path(result["output_file"])
            file_size_kb = result["file_size_bytes"] / 1024
            
            print(f"✅ Generated output: {output_file.name} ({file_size_kb:.1f} KB)")
            
            if result.get("metadata_file"):
                metadata_file = Path(result["metadata_file"])
                print(f"📊 Metadata file: {metadata_file.name}")
            
            # Show summary
            summary = generator.get_output_summary()
            print("\n📊 Processing Summary:")
            print(f"   Total messages: {summary['total_messages']}")
            print(f"   Text messages: {summary['text_messages']}")
            print(f"   Audio messages: {summary['audio_messages']}")
            print(f"   Successfully transcribed: {summary['successfully_transcribed']}")
            print(f"   Transcription rate: {summary['transcription_rate']:.1%}")
            print(f"   Participants: {', '.join(summary['participants'])}")
            
            print(f"\n🎉 Processing complete! Check the 'outputs' folder for your files.")
            return True
    
    except Exception as e:
        print(f"💥 Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_help():
    """Show help information"""
    print("""
🎵 WhatsApp Audio Transcriber
============================

Convert WhatsApp voice messages to readable text using AI.

Usage:
    python main.py                    # Start web interface (default)
    python main.py web               # Start web interface
    python main.py cli <zip_file>    # Process via command line

Examples:
    python main.py                          # Start web server
    python main.py cli my_chat.zip         # Process ZIP file
    python main.py cli my_chat.zip json    # Process and output JSON

Supported output formats:
    txt     # Human-readable text file (default)
    json    # Structured data with metadata
    csv     # Spreadsheet-compatible format

Requirements:
    - FFmpeg installed and in PATH
    - Python packages: openai-whisper, fastapi, uvicorn
    - WhatsApp chat export (ZIP file with audio)

How to export WhatsApp chat:
    1. Open WhatsApp → Select chat
    2. Menu (⋮) → More → Export chat
    3. Choose "Include media"
    4. Save ZIP file

Web Interface:
    - Upload ZIP files via drag & drop
    - Real-time progress tracking
    - Download results in multiple formats
    - Mobile-friendly interface

Command Line:
    - Direct processing without web browser
    - Faster for batch processing
    - Detailed progress output

Output Files:
    - Text files with transcribed audio messages
    - JSON files with full metadata
    - CSV files for data analysis
    - Processing statistics and logs

🌐 Web interface: http://localhost:8000
📁 Output folder: ./outputs/
📋 Logs: Detailed progress information

For more info: https://github.com/your-repo/whatsapp-transcriber
""")


def main():
    """Main entry point"""
    
    # Parse command line arguments
    if len(sys.argv) == 1:
        # No arguments - start web interface
        start_web_interface()
        return
    
    command = sys.argv[1].lower()
    
    if command in ['--help', '-h', 'help']:
        show_help()
    elif command == 'web':
        start_web_interface()
    elif command == 'cli':
        if len(sys.argv) < 3:
            print("❌ CLI mode requires a ZIP file path")
            print("Usage: python main.py cli <zip_file> [output_format]")
            print("Example: python main.py cli my_chat.zip txt")
            return
        
        zip_file = sys.argv[2]
        output_format = sys.argv[3] if len(sys.argv) > 3 else "txt"
        
        if output_format not in ["txt", "json", "csv"]:
            print(f"❌ Invalid output format: {output_format}")
            print("Supported formats: txt, json, csv")
            return
        
        success = process_cli(zip_file, output_format)
        sys.exit(0 if success else 1)
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'python main.py --help' for usage information")
        print("Available commands: web, cli, help")


if __name__ == "__main__":
    main()