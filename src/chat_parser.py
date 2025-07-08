"""
WhatsApp chat parser module
Parses WhatsApp chat.txt files and identifies audio message locations
"""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from config import AUDIO_PATTERNS, DATETIME_FORMATS


@dataclass
class ChatMessage:
    """Represents a single chat message"""
    timestamp: datetime
    sender: str
    content: str
    line_number: int
    is_audio: bool = False
    audio_index: Optional[int] = None
    original_line: str = ""


class WhatsAppChatParser:
    """Parse WhatsApp chat.txt files and identify audio messages"""
    
    def __init__(self, chat_file_path: str):
        self.chat_file_path = Path(chat_file_path)
        self.messages: List[ChatMessage] = []
        self.audio_messages: List[ChatMessage] = []
        self.raw_lines: List[str] = []
        
    def parse(self) -> Dict[str, any]:
        """
        Parse the chat file and identify all messages
        
        Returns:
            Dict with parsing results
        """
        try:
            # Read the file
            with open(self.chat_file_path, 'r', encoding='utf-8') as f:
                self.raw_lines = f.readlines()
            
            logger.info(f"Loaded {len(self.raw_lines)} lines from chat file")
            
            # Parse messages
            self._parse_messages()
            
            # Identify audio messages
            self._identify_audio_messages()
            
            return {
                "success": True,
                "total_lines": len(self.raw_lines),
                "total_messages": len(self.messages),
                "audio_messages": len(self.audio_messages),
                "chat_info": self._get_chat_info()
            }
            
        except FileNotFoundError:
            logger.error(f"Chat file not found: {self.chat_file_path}")
            return {"success": False, "error": "Chat file not found"}
        except UnicodeDecodeError:
            logger.error("Could not decode chat file. Try different encoding.")
            return {"success": False, "error": "Encoding error"}
        except Exception as e:
            logger.error(f"Parsing failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _parse_messages(self):
        """Parse individual messages from chat lines"""
        current_message = None
        
        for line_num, line in enumerate(self.raw_lines):
            line = line.strip()
            if not line:
                continue
            
            # Try to match message pattern
            message_match = self._parse_message_line(line)
            
            if message_match:
                # Save previous message if exists
                if current_message:
                    self.messages.append(current_message)
                
                # Start new message
                timestamp, sender, content = message_match
                current_message = ChatMessage(
                    timestamp=timestamp,
                    sender=sender,
                    content=content,
                    line_number=line_num,
                    original_line=line
                )
            else:
                # This is a continuation of the previous message
                if current_message:
                    current_message.content += "\n" + line
                    current_message.original_line += "\n" + line
        
        # Don't forget the last message
        if current_message:
            self.messages.append(current_message)
        
        logger.info(f"Parsed {len(self.messages)} messages")
    
    def _parse_message_line(self, line: str) -> Optional[Tuple[datetime, str, str]]:
        """
        Parse a single message line to extract timestamp, sender, and content
        
        Returns:
            Tuple of (timestamp, sender, content) or None if not a message line
        """
        # Multiple regex patterns for different WhatsApp export formats
        patterns = [
            # US format: M/D/YY, HH:MM - Sender: Content (most common)
            r'^(\d{1,2}\/\d{1,2}\/\d{2,4}, \d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\s*-\s*([^:]+):\s*(.*)$',
            
            # European format: DD/MM/YYYY, HH:MM - Sender: Content  
            r'^(\d{1,2}\/\d{1,2}\/\d{4}, \d{1,2}:\d{2}(?::\d{2})?)\s*-\s*([^:]+):\s*(.*)$',
            
            # German format: DD.MM.YYYY, HH:MM - Sender: Content
            r'^(\d{1,2}\.\d{1,2}\.\d{2,4}, \d{1,2}:\d{2}(?::\d{2})?)\s*-\s*([^:]+):\s*(.*)$',
            
            # ISO format: YYYY-MM-DD, HH:MM - Sender: Content
            r'^(\d{4}-\d{1,2}-\d{1,2}, \d{1,2}:\d{2}(?::\d{2})?)\s*-\s*([^:]+):\s*(.*)$',
            
            # With brackets: [Date] Sender: Content
            r'^\[(\d{1,2}\/\d{1,2}\/\d{2,4}, \d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\]\s*([^:]+):\s*(.*)$',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                timestamp_str, sender, content = match.groups()
                
                # Parse timestamp
                timestamp = self._parse_timestamp(timestamp_str)
                if timestamp:
                    return timestamp, sender.strip(), content.strip()
        
        return None
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string into datetime object"""
        timestamp_str = timestamp_str.strip()
        
        # Try all configured datetime formats
        for fmt in DATETIME_FORMATS:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse timestamp: {timestamp_str}")
        return None
    
    def _identify_audio_messages(self):
        """Identify which messages are audio messages"""
        audio_count = 0
        
        for message in self.messages:
            # Check if message content matches audio patterns
            is_audio = False
            
            for pattern in AUDIO_PATTERNS:
                if re.search(pattern, message.content, re.IGNORECASE):
                    is_audio = True
                    logger.debug(f"Audio pattern matched: '{pattern}' in content: '{message.content[:100]}'")
                    break
            
            if is_audio:
                message.is_audio = True
                message.audio_index = audio_count
                self.audio_messages.append(message)
                audio_count += 1
                logger.debug(f"Found audio message {audio_count}: {message.sender} at {message.timestamp}")
        
        logger.info(f"Identified {len(self.audio_messages)} potential audio messages")
    
    def _get_chat_info(self) -> Dict[str, any]:
        """Get general information about the chat"""
        if not self.messages:
            return {}
        
        senders = set(msg.sender for msg in self.messages)
        timestamps = [msg.timestamp for msg in self.messages if msg.timestamp]
        
        if timestamps:
            date_range = (min(timestamps), max(timestamps))
        else:
            date_range = (None, None)
        
        return {
            "participants": list(senders),
            "participant_count": len(senders),
            "date_range": {
                "start": date_range[0].isoformat() if date_range[0] else None,
                "end": date_range[1].isoformat() if date_range[1] else None
            },
            "total_days": (date_range[1] - date_range[0]).days if all(date_range) else 0
        }
    
    def get_audio_message_details(self) -> List[Dict[str, any]]:
        """Get detailed information about audio messages"""
        details = []
        
        for msg in self.audio_messages:
            detail = {
                "audio_index": msg.audio_index,
                "line_number": msg.line_number,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                "sender": msg.sender,
                "original_content": msg.content,
                "original_line": msg.original_line
            }
            details.append(detail)
        
        return details
    
    def get_context_around_audio(self, audio_index: int, context_lines: int = 2) -> List[str]:
        """Get context lines around an audio message"""
        if audio_index >= len(self.audio_messages):
            return []
        
        audio_msg = self.audio_messages[audio_index]
        start_line = max(0, audio_msg.line_number - context_lines)
        end_line = min(len(self.raw_lines), audio_msg.line_number + context_lines + 1)
        
        return self.raw_lines[start_line:end_line]
    
    def prepare_for_transcription(self) -> List[Dict[str, any]]:
        """Prepare audio message data for transcription matching"""
        transcription_data = []
        
        for msg in self.audio_messages:
            data = {
                "audio_index": msg.audio_index,
                "line_number": msg.line_number,
                "timestamp": msg.timestamp,
                "sender": msg.sender,
                "placeholder_text": msg.content,
                "context_before": [],
                "context_after": []
            }
            
            # Add context for better matching
            msg_index = self.messages.index(msg)
            if msg_index > 0:
                data["context_before"] = [self.messages[msg_index-1].content]
            if msg_index < len(self.messages) - 1:
                data["context_after"] = [self.messages[msg_index+1].content]
            
            transcription_data.append(data)
        
        return transcription_data


# Convenience function
def parse_whatsapp_chat(chat_file_path: str) -> Tuple[WhatsAppChatParser, Dict[str, any]]:
    """
    Parse WhatsApp chat file
    
    Returns:
        Tuple of (parser_instance, parse_results)
    """
    parser = WhatsAppChatParser(chat_file_path)
    results = parser.parse()
    return parser, results


# Test function
if __name__ == "__main__":
    # Example usage
    test_chat = "test_chat.txt"
    
    if Path(test_chat).exists():
        parser, results = parse_whatsapp_chat(test_chat)
        
        if results["success"]:
            print("✅ Chat parsing successful!")
            print(f"📊 Results: {results}")
            
            # Show audio messages
            audio_details = parser.get_audio_message_details()
            print(f"\n🎵 Found {len(audio_details)} audio messages:")
            for detail in audio_details[:3]:  # Show first 3
                print(f"   {detail['audio_index']}: {detail['sender']} at {detail['timestamp']}")
        else:
            print(f"❌ Parsing failed: {results['error']}")
    else:
        print(f"Test file {test_chat} not found.")