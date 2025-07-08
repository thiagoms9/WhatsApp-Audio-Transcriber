"""
Output generator module
Generates final chat files with transcribed audio messages
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import csv
from loguru import logger
from config import OUTPUTS_DIR, OUTPUT_FORMAT, INCLUDE_METADATA, PRESERVE_ORIGINAL_FORMAT


class WhatsAppOutputGenerator:
    """Generate output files with transcribed audio messages"""
    
    def __init__(self, chat_parser, transcriber, original_chat_file: str):
        self.chat_parser = chat_parser
        self.transcriber = transcriber
        self.original_chat_file = Path(original_chat_file)
        self.output_data = []
        
    def generate_output(self, output_format: str = OUTPUT_FORMAT, 
                       output_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate output file with transcribed audio messages
        
        Args:
            output_format: Format for output ("txt", "json", "csv")
            output_filename: Custom filename (auto-generated if None)
            
        Returns:
            Dict with generation results
        """
        try:
            logger.info(f"Generating output in {output_format} format")
            
            # Prepare the output data
            self._prepare_output_data()
            
            # Generate filename if not provided
            if not output_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                chat_name = self.original_chat_file.stem.replace("WhatsApp Chat with ", "").replace(" ", "_")
                output_filename = f"{chat_name}_transcribed_{timestamp}.{output_format}"
            
            output_path = OUTPUTS_DIR / output_filename
            
            # Generate output based on format
            if output_format.lower() == "txt":
                self._generate_txt_output(output_path)
            elif output_format.lower() == "json":
                self._generate_json_output(output_path)
            elif output_format.lower() == "csv":
                self._generate_csv_output(output_path)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            # Generate metadata file if requested
            metadata_path = None
            if INCLUDE_METADATA:
                metadata_path = self._generate_metadata_file(output_path)
            
            file_size = output_path.stat().st_size if output_path.exists() else 0
            
            return {
                "success": True,
                "output_file": str(output_path),
                "metadata_file": str(metadata_path) if metadata_path else None,
                "format": output_format,
                "file_size_bytes": file_size,
                "total_messages": len(self.chat_parser.messages),
                "transcribed_audio_messages": len(self.transcriber.transcription_results),
                "successful_transcriptions": len([r for r in self.transcriber.transcription_results if r.status == "success"])
            }
            
        except Exception as e:
            logger.error(f"Output generation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _prepare_output_data(self):
        """Prepare data for output generation"""
        logger.info("Preparing output data")
        
        # Get transcription results organized by chat index
        transcription_map = {}
        for result in self.transcriber.transcription_results:
            if result.status == "success":
                transcription_map[result.chat_index] = result
        
        # Process each message
        transcribed_count = 0
        for message in self.chat_parser.messages:
            message_data = {
                "timestamp": message.timestamp.isoformat() if message.timestamp else None,
                "sender": message.sender,
                "original_content": message.content,
                "line_number": message.line_number,
                "is_audio": message.is_audio,
                "audio_index": message.audio_index
            }
            
            # Replace audio content with transcription if available
            if message.is_audio and message.audio_index is not None:
                if message.audio_index in transcription_map:
                    transcription = transcription_map[message.audio_index]
                    message_data["transcribed_content"] = transcription.transcription_text
                    message_data["transcription_confidence"] = transcription.confidence
                    message_data["detected_language"] = transcription.language
                    message_data["audio_duration"] = transcription.duration_seconds
                    message_data["audio_filename"] = Path(transcription.audio_file_path).name
                    transcribed_count += 1
                    logger.debug(f"Transcribed audio message {message.audio_index}: {transcription.transcription_text[:50]}...")
                else:
                    message_data["transcribed_content"] = None
                    message_data["transcription_note"] = "Audio file not found or transcription failed"
            else:
                message_data["transcribed_content"] = None
            
            self.output_data.append(message_data)
        
        logger.info(f"Prepared {len(self.output_data)} messages with {transcribed_count} transcriptions")
    
    def _generate_txt_output(self, output_path: Path):
        """Generate text output file"""
        logger.info(f"Generating TXT output: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("WhatsApp Chat with Transcribed Audio Messages\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original file: {self.original_chat_file.name}\n")
            f.write(f"Total messages: {len(self.output_data)}\n")
            
            audio_count = len([m for m in self.output_data if m['is_audio']])
            transcribed_count = len([m for m in self.output_data if m['transcribed_content']])
            f.write(f"Audio messages: {audio_count}\n")
            f.write(f"Successfully transcribed: {transcribed_count}\n")
            f.write("\n" + "=" * 50 + "\n\n")
            
            # Write messages
            for message in self.output_data:
                timestamp_str = ""
                if message['timestamp']:
                    dt = datetime.fromisoformat(message['timestamp'])
                    timestamp_str = dt.strftime("%m/%d/%y, %H:%M")
                
                sender = message['sender']
                
                # Determine content to display
                if message['transcribed_content']:
                    # Use transcription for audio messages
                    content = message['transcribed_content']
                    if INCLUDE_METADATA:
                        confidence = message.get('transcription_confidence', 0)
                        language = message.get('detected_language', 'unknown')
                        duration = message.get('audio_duration', 0)
                        content += f" [🎵 Audio transcribed: {confidence:.2f} confidence, {language}, {duration:.1f}s]"
                else:
                    # Use original content for text messages
                    content = message['original_content']
                
                # Format message line
                if PRESERVE_ORIGINAL_FORMAT:
                    # Match original WhatsApp format
                    f.write(f"{timestamp_str} - {sender}: {content}\n")
                else:
                    # Enhanced format with metadata
                    f.write(f"[{timestamp_str}] {sender}:\n")
                    f.write(f"  {content}\n")
                    if message['is_audio'] and INCLUDE_METADATA:
                        if message['transcribed_content']:
                            f.write(f"  📊 Original: {message['original_content']}\n")
                        else:
                            f.write(f"  ⚠️  {message.get('transcription_note', 'Audio not transcribed')}\n")
                    f.write("\n")
        
        logger.info(f"TXT output generated: {output_path}")
    
    def _generate_json_output(self, output_path: Path):
        """Generate JSON output file"""
        logger.info(f"Generating JSON output: {output_path}")
        
        output_structure = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "original_file": str(self.original_chat_file),
                "total_messages": len(self.output_data),
                "audio_messages": len([m for m in self.output_data if m['is_audio']]),
                "transcribed_messages": len([m for m in self.output_data if m['transcribed_content']]),
                "participants": list(set(m['sender'] for m in self.output_data if m['sender'])),
                "transcription_summary": self.transcriber.get_transcription_summary()
            },
            "messages": self.output_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_structure, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"JSON output generated: {output_path}")
    
    def _generate_csv_output(self, output_path: Path):
        """Generate CSV output file"""
        logger.info(f"Generating CSV output: {output_path}")
        
        fieldnames = [
            'timestamp', 'sender', 'original_content', 'transcribed_content',
            'is_audio', 'transcription_confidence', 'detected_language', 
            'audio_duration', 'audio_filename', 'line_number'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for message in self.output_data:
                row = {field: message.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        logger.info(f"CSV output generated: {output_path}")
    
    def _generate_metadata_file(self, output_path: Path) -> Path:
        """Generate metadata file with processing details"""
        metadata_path = output_path.with_suffix('.metadata.json')
        
        metadata = {
            "processing_info": {
                "generated_at": datetime.now().isoformat(),
                "original_chat_file": str(self.original_chat_file),
                "output_file": str(output_path),
                "processing_steps": [
                    "ZIP extraction",
                    "Chat parsing", 
                    "Audio file matching",
                    "Audio transcription",
                    "Output generation"
                ]
            },
            "chat_statistics": {
                "total_messages": len(self.chat_parser.messages),
                "total_audio_messages": len([m for m in self.output_data if m['is_audio']]),
                "successfully_transcribed": len([m for m in self.output_data if m['transcribed_content']]),
                "participants": list(set(m['sender'] for m in self.output_data if m['sender'])),
                "date_range": self.chat_parser._get_chat_info().get("date_range", {})
            },
            "transcription_results": self.transcriber.get_transcription_summary(),
            "audio_details": [
                {
                    "audio_index": result.chat_index,
                    "filename": Path(result.audio_file_path).name,
                    "sender": result.chat_message.get('sender', 'Unknown'),
                    "timestamp": result.chat_message.get('timestamp', 'Unknown'),
                    "transcription": result.transcription_text,
                    "confidence": result.confidence,
                    "language": result.language,
                    "duration_seconds": result.duration_seconds
                }
                for result in self.transcriber.transcription_results
                if result.status == "success"
            ]
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Metadata file generated: {metadata_path}")
        return metadata_path
    
    def get_output_summary(self) -> Dict[str, Any]:
        """Get summary of output generation"""
        if not self.output_data:
            return {"error": "No output data prepared"}
        
        total_messages = len(self.output_data)
        audio_messages = len([m for m in self.output_data if m['is_audio']])
        transcribed_messages = len([m for m in self.output_data if m['transcribed_content']])
        
        return {
            "total_messages": total_messages,
            "text_messages": total_messages - audio_messages,
            "audio_messages": audio_messages,
            "successfully_transcribed": transcribed_messages,
            "failed_transcriptions": audio_messages - transcribed_messages,
            "transcription_rate": transcribed_messages / audio_messages if audio_messages > 0 else 0,
            "participants": list(set(m['sender'] for m in self.output_data if m['sender']))
        }


def generate_transcribed_output(chat_parser, transcriber, original_chat_file: str,
                              output_format: str = OUTPUT_FORMAT,
                              output_filename: Optional[str] = None) -> tuple[WhatsAppOutputGenerator, Dict[str, Any]]:
    """
    Convenience function to generate transcribed output
    
    Args:
        chat_parser: Parsed chat data
        transcriber: Transcription results
        original_chat_file: Path to original chat file
        output_format: Output format ("txt", "json", "csv")
        output_filename: Custom output filename
        
    Returns:
        Tuple of (generator_instance, generation_results)
    """
    generator = WhatsAppOutputGenerator(chat_parser, transcriber, original_chat_file)
    results = generator.generate_output(output_format, output_filename)
    return generator, results


# Test function
if __name__ == "__main__":
    print("Output Generator module - run via test_output_generator.py")