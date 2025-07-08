"""
WhatsApp ZIP extractor module
Handles extraction and analysis of WhatsApp exported ZIP files
"""

import zipfile
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import tempfile
from config import TEMP_DIR, SUPPORTED_AUDIO_FORMATS


class WhatsAppZipExtractor:
    """Extract and analyze WhatsApp ZIP exports"""
    
    def __init__(self, zip_path: str):
        self.zip_path = Path(zip_path)
        self.extract_dir = None
        self.chat_file = None
        self.audio_files = []
        self.media_files = []
        
    def extract(self, output_dir: Optional[str] = None) -> Dict[str, any]:
        """
        Extract ZIP file and analyze contents
        
        Returns:
            Dict with extraction results and file paths
        """
        if output_dir:
            self.extract_dir = Path(output_dir)
        else:
            # Create temporary directory
            self.extract_dir = TEMP_DIR / f"extract_{self.zip_path.stem}"
        
        try:
            # Create extraction directory
            self.extract_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract ZIP file
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)
                logger.info(f"Extracted ZIP to: {self.extract_dir}")
            
            # Analyze extracted contents
            analysis = self._analyze_contents()
            
            return {
                "success": True,
                "extract_dir": str(self.extract_dir),
                "chat_file": self.chat_file,
                "audio_files": self.audio_files,
                "media_files": self.media_files,
                "analysis": analysis
            }
            
        except zipfile.BadZipFile:
            logger.error(f"Invalid ZIP file: {self.zip_path}")
            return {"success": False, "error": "Invalid ZIP file"}
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _analyze_contents(self) -> Dict[str, any]:
        """Analyze extracted contents and categorize files"""
        
        # Find chat file
        chat_files = list(self.extract_dir.rglob("*.txt"))
        if chat_files:
            # Usually named "_chat.txt" or "WhatsApp Chat with [Name].txt"
            self.chat_file = str(chat_files[0])  # Take the first .txt file
            logger.info(f"Found chat file: {Path(self.chat_file).name}")
        else:
            logger.warning("No chat file (.txt) found in ZIP")
        
        # Find audio files
        for ext in SUPPORTED_AUDIO_FORMATS:
            audio_files = list(self.extract_dir.rglob(f"*{ext}"))
            self.audio_files.extend([str(f) for f in audio_files])
        
        # Sort audio files by creation time (important for matching)
        self.audio_files.sort(key=lambda x: os.path.getctime(x))
        
        # Find other media files
        media_extensions = [".jpg", ".jpeg", ".png", ".gif", ".mp4", ".mov", ".pdf", ".doc", ".docx"]
        for ext in media_extensions:
            media_files = list(self.extract_dir.rglob(f"*{ext}"))
            self.media_files.extend([str(f) for f in media_files])
        
        analysis = {
            "total_files": len(list(self.extract_dir.rglob("*"))),
            "chat_file_found": self.chat_file is not None,
            "audio_count": len(self.audio_files),
            "media_count": len(self.media_files),
            "audio_formats": self._get_audio_formats(),
            "estimated_size_mb": self._get_directory_size()
        }
        
        logger.info(f"Analysis complete: {analysis}")
        return analysis
    
    def _get_audio_formats(self) -> Dict[str, int]:
        """Count audio files by format"""
        formats = {}
        for audio_file in self.audio_files:
            ext = Path(audio_file).suffix.lower()
            formats[ext] = formats.get(ext, 0) + 1
        return formats
    
    def _get_directory_size(self) -> float:
        """Calculate directory size in MB"""
        total_size = 0
        for file_path in self.extract_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return round(total_size / (1024 * 1024), 2)
    
    def get_audio_files_info(self) -> List[Dict[str, any]]:
        """Get detailed info about each audio file"""
        audio_info = []
        
        for i, audio_file in enumerate(self.audio_files):
            file_path = Path(audio_file)
            try:
                stat = file_path.stat()
                info = {
                    "index": i,
                    "filename": file_path.name,
                    "full_path": str(file_path),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_time": stat.st_ctime,
                    "modified_time": stat.st_mtime,
                    "extension": file_path.suffix.lower()
                }
                audio_info.append(info)
            except Exception as e:
                logger.warning(f"Could not get info for {audio_file}: {e}")
        
        return audio_info
    
    def cleanup(self):
        """Clean up extracted files"""
        if self.extract_dir and self.extract_dir.exists():
            try:
                shutil.rmtree(self.extract_dir)
                logger.info(f"Cleaned up: {self.extract_dir}")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def extract_whatsapp_zip(zip_path: str, output_dir: Optional[str] = None) -> Dict[str, any]:
    """
    Convenience function to extract WhatsApp ZIP
    
    Args:
        zip_path: Path to the WhatsApp ZIP file
        output_dir: Optional output directory for extraction
        
    Returns:
        Dictionary with extraction results
    """
    extractor = WhatsAppZipExtractor(zip_path)
    return extractor.extract(output_dir)


# Test function
if __name__ == "__main__":
    # Example usage
    test_zip = "sample_whatsapp_export.zip"
    
    if Path(test_zip).exists():
        with WhatsAppZipExtractor(test_zip) as extractor:
            result = extractor.extract()
            
            if result["success"]:
                print("✅ Extraction successful!")
                print(f"📁 Chat file: {result['chat_file']}")
                print(f"🎵 Audio files found: {len(result['audio_files'])}")
                print(f"📊 Analysis: {result['analysis']}")
                
                # Show audio files info
                audio_info = extractor.get_audio_files_info()
                for info in audio_info[:3]:  # Show first 3
                    print(f"   🎵 {info['filename']} ({info['size_mb']} MB)")
            else:
                print(f"❌ Extraction failed: {result['error']}")
    else:
        print(f"Test file {test_zip} not found. Create a sample to test.")