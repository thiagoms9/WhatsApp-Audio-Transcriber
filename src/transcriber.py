"""
Audio transcription module using faster-whisper (optimized for production)
Handles transcription of matched audio files with smaller memory footprint
"""

import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import tempfile
from loguru import logger
from tqdm import tqdm

# Import faster-whisper instead of openai-whisper
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not available - transcription will be disabled")

from config import WHISPER_MODEL, WHISPER_LANGUAGE, WHISPER_TASK


@dataclass
class TranscriptionResult:
    """Represents the result of transcribing a single audio file"""
    audio_file_path: str
    chat_index: int
    chat_message: Dict
    transcription_text: str
    confidence: float
    duration_seconds: float
    language: str
    status: str  # "success", "error", "skipped"
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None


class AudioTranscriber:
    """Transcribe audio files using faster-whisper with enhanced settings"""
    
    def __init__(self, model_name: str = WHISPER_MODEL, language: str = WHISPER_LANGUAGE, 
                 enable_preprocessing: bool = True, enable_postprocessing: bool = True):
        self.model_name = model_name
        self.language = language if language != "auto" else None  # Convert "auto" to None
        self.enable_preprocessing = enable_preprocessing
        self.enable_postprocessing = enable_postprocessing
        self.model = None
        self.transcription_results: List[TranscriptionResult] = []
        
    def load_model(self) -> bool:
        """Load the faster-whisper model"""
        if not WHISPER_AVAILABLE:
            logger.error("faster-whisper is not available. Please install it with: pip install faster-whisper")
            return False
        
        try:
            logger.info(f"Loading faster-whisper model: {self.model_name}")
            # faster-whisper uses CPU by default, which is much smaller
            self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            logger.info("✅ faster-whisper model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load faster-whisper model: {e}")
            return False
    
    def transcribe_queue(self, transcription_queue: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Transcribe a queue of audio files
        
        Args:
            transcription_queue: List of audio files to transcribe (from AudioMatcher)
            
        Returns:
            Dict with transcription results
        """
        if not self.model:
            if not self.load_model():
                return {
                    "success": False,
                    "error": "Could not load faster-whisper model",
                    "results": []
                }
        
        logger.info(f"Starting transcription of {len(transcription_queue)} audio files")
        
        total_files = len(transcription_queue)
        successful_transcriptions = 0
        
        # Process each file with progress bar
        with tqdm(total=total_files, desc="Transcribing audio files", unit="file") as pbar:
            for i, queue_item in enumerate(transcription_queue):
                audio_file_path = queue_item["audio_file_path"]
                chat_index = queue_item["chat_index"]
                chat_message = queue_item["chat_message"]
                
                pbar.set_description(f"Transcribing {Path(audio_file_path).name}")
                
                try:
                    # Transcribe the audio file
                    result = self._transcribe_single_file(
                        audio_file_path=audio_file_path,
                        chat_index=chat_index,
                        chat_message=chat_message
                    )
                    
                    self.transcription_results.append(result)
                    
                    if result.status == "success":
                        successful_transcriptions += 1
                        logger.info(f"✅ Transcribed: {Path(audio_file_path).name}")
                        logger.debug(f"   Text: {result.transcription_text[:100]}...")
                    else:
                        logger.warning(f"❌ Failed: {Path(audio_file_path).name} - {result.error_message}")
                    
                except Exception as e:
                    error_result = TranscriptionResult(
                        audio_file_path=audio_file_path,
                        chat_index=chat_index,
                        chat_message=chat_message,
                        transcription_text="",
                        confidence=0.0,
                        duration_seconds=0.0,
                        language="unknown",
                        status="error",
                        error_message=str(e)
                    )
                    self.transcription_results.append(error_result)
                    logger.error(f"💥 Exception transcribing {Path(audio_file_path).name}: {e}")
                
                pbar.update(1)
        
        success_rate = successful_transcriptions / total_files if total_files > 0 else 0
        
        return {
            "success": True,
            "total_files": total_files,
            "successful_transcriptions": successful_transcriptions,
            "failed_transcriptions": total_files - successful_transcriptions,
            "success_rate": success_rate,
            "results": self.transcription_results
        }
    
    def _transcribe_single_file(self, audio_file_path: str, chat_index: int, chat_message: Dict) -> TranscriptionResult:
        """Transcribe a single audio file with faster-whisper"""
        
        start_time = time.time()
        
        try:
            # Check if file exists and is readable
            if not Path(audio_file_path).exists():
                return self._create_error_result(audio_file_path, chat_index, chat_message, "Audio file not found")
            
            # Get file size for validation
            file_size = Path(audio_file_path).stat().st_size
            if file_size == 0:
                return self._create_error_result(audio_file_path, chat_index, chat_message, "Audio file is empty")
            
            logger.debug(f"Transcribing {Path(audio_file_path).name} ({file_size} bytes)")
            
            # Transcribe with faster-whisper
            segments, info = self.model.transcribe(
                audio_file_path,
                language=self.language,
                task=WHISPER_TASK,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=True,
                initial_prompt=self._get_context_prompt(chat_message),
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                word_timestamps=False  # Disable for faster processing
            )
            
            # Extract transcription text from segments
            transcription_text = ""
            segment_confidences = []
            
            for segment in segments:
                transcription_text += segment.text + " "
                # faster-whisper provides avg_logprob per segment
                if hasattr(segment, 'avg_logprob'):
                    segment_confidences.append(segment.avg_logprob)
            
            transcription_text = transcription_text.strip()
            
            # Post-process transcription if enabled
            if self.enable_postprocessing:
                transcription_text = self._postprocess_text(transcription_text, info.language)
            
            # Calculate confidence from segment avg_logprobs
            if segment_confidences:
                avg_logprob = sum(segment_confidences) / len(segment_confidences)
                confidence = max(0.0, min(1.0, (avg_logprob + 1.0)))
            else:
                confidence = 0.7  # Default confidence
            
            # Get duration from info
            duration = info.duration if hasattr(info, 'duration') else 0.0
            detected_language = info.language if hasattr(info, 'language') else "unknown"
            
            processing_time = time.time() - start_time
            
            return TranscriptionResult(
                audio_file_path=audio_file_path,
                chat_index=chat_index,
                chat_message=chat_message,
                transcription_text=transcription_text,
                confidence=confidence,
                duration_seconds=duration,
                language=detected_language,
                status="success",
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return self._create_error_result(audio_file_path, chat_index, chat_message, str(e), processing_time)
    
    def _get_context_prompt(self, chat_message: Dict) -> str:
        """Generate context prompt from chat message for better transcription"""
        try:
            sender = chat_message.get('sender', '')
            timestamp = chat_message.get('timestamp', '')
            
            # Create context based on language
            if self.language == "pt":
                return f"Esta é uma mensagem de áudio do WhatsApp de {sender}."
            elif self.language == "es":
                return f"Este es un mensaje de audio de WhatsApp de {sender}."
            elif self.language == "fr":
                return f"Ceci est un message audio WhatsApp de {sender}."
            else:
                return f"This is a WhatsApp voice message from {sender}."
        except:
            return ""
    
    def _postprocess_text(self, text: str, language: str) -> str:
        """Post-process transcription text for better quality"""
        if not text:
            return text
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Language-specific improvements
        if language == "pt":
            # Portuguese-specific corrections
            text = self._apply_portuguese_corrections(text)
        elif language == "en":
            # English-specific corrections
            text = self._apply_english_corrections(text)
        
        # General improvements
        text = self._apply_general_corrections(text)
        
        return text
    
    def _apply_portuguese_corrections(self, text: str) -> str:
        """Apply Portuguese-specific corrections"""
        corrections = {
            r'\bvc\b': 'você',
            r'\btb\b': 'também',
            r'\bpq\b': 'porque',
            r'\bblz\b': 'beleza',
            r'\bokay\b': 'ok',
            r'\bok\b': 'ok',
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _apply_english_corrections(self, text: str) -> str:
        """Apply English-specific corrections"""
        corrections = {
            r'\bu\b': 'you',
            r'\bur\b': 'your',
            r'\br\b': 'are',
            r'\bthru\b': 'through',
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _apply_general_corrections(self, text: str) -> str:
        """Apply general corrections"""
        # Remove repeated words
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
        
        # Fix common transcription artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
        text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical content
        text = re.sub(r'\.{3,}', '...', text)  # Normalize ellipses
        
        return text.strip()
    
    def _create_error_result(self, audio_file_path: str, chat_index: int, chat_message: Dict, 
                           error_msg: str, processing_time: float = 0.0) -> TranscriptionResult:
        """Create an error result"""
        return TranscriptionResult(
            audio_file_path=audio_file_path,
            chat_index=chat_index,
            chat_message=chat_message,
            transcription_text="",
            confidence=0.0,
            duration_seconds=0.0,
            language="unknown",
            status="error",
            error_message=error_msg,
            processing_time_seconds=processing_time
        )
    
    def get_transcription_summary(self) -> Dict[str, Any]:
        """Get a summary of all transcription results"""
        if not self.transcription_results:
            return {}
        
        successful = [r for r in self.transcription_results if r.status == "success"]
        failed = [r for r in self.transcription_results if r.status == "error"]
        
        total_duration = sum(r.duration_seconds for r in successful)
        total_processing_time = sum(r.processing_time_seconds for r in self.transcription_results if r.processing_time_seconds)
        avg_confidence = sum(r.confidence for r in successful) / len(successful) if successful else 0.0
        
        languages = {}
        for result in successful:
            lang = result.language
            languages[lang] = languages.get(lang, 0) + 1
        
        return {
            "total_files": len(self.transcription_results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.transcription_results),
            "total_audio_duration": total_duration,
            "total_processing_time": total_processing_time,
            "avg_confidence": avg_confidence,
            "languages_detected": languages,
            "failed_files": [r.audio_file_path for r in failed]
        }
    
    def get_transcription_for_output(self) -> List[Dict[str, Any]]:
        """Get transcription results formatted for output generation"""
        output_data = []
        
        for result in self.transcription_results:
            if result.status == "success":
                output_data.append({
                    "chat_index": result.chat_index,
                    "chat_message": result.chat_message,
                    "transcription": result.transcription_text,
                    "confidence": result.confidence,
                    "duration": result.duration_seconds,
                    "language": result.language,
                    "audio_filename": Path(result.audio_file_path).name
                })
        
        # Sort by chat index to maintain order
        output_data.sort(key=lambda x: x["chat_index"])
        return output_data


def transcribe_audio_files(transcription_queue: List[Dict[str, Any]], 
                          model_name: str = WHISPER_MODEL,
                          language: str = WHISPER_LANGUAGE) -> tuple[AudioTranscriber, Dict[str, Any]]:
    """
    Convenience function to transcribe audio files
    
    Args:
        transcription_queue: Queue of audio files to transcribe
        model_name: Whisper model to use
        language: Language to use (None for auto-detection)
        
    Returns:
        Tuple of (transcriber_instance, transcription_results)
    """
    transcriber = AudioTranscriber(model_name, language)
    results = transcriber.transcribe_queue(transcription_queue)
    return transcriber, results


# Test function
if __name__ == "__main__":
    if not WHISPER_AVAILABLE:
        print("❌ faster-whisper not available. Please install it with: pip install faster-whisper")
    else:
        print("✅ faster-whisper is available")
        print("Audio Transcriber module - run via test_transcriber.py")