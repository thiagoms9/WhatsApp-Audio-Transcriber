"""
Audio matcher module
Matches chat audio message references with actual audio files
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class AudioMatch:
    """Represents a matched audio message and file"""
    chat_index: int                # Index in chat audio messages
    file_index: int                # Index in audio files list  
    chat_message: Dict             # Chat message data
    audio_file_path: str           # Path to audio file
    confidence: float              # Matching confidence (0-1)
    match_method: str              # How the match was determined
    time_diff_minutes: Optional[float] = None  # Time difference if timestamp matching


@dataclass 
class AudioFileInfo:
    """Information about an audio file"""
    index: int
    file_path: str
    filename: str
    size_bytes: int
    created_time: datetime
    modified_time: datetime
    extension: str


class AudioMatcher:
    """Match chat audio references with actual audio files"""
    
    def __init__(self, audio_files: List[str], chat_audio_messages: List[Dict]):
        self.audio_files = audio_files
        self.chat_audio_messages = chat_audio_messages
        self.audio_file_info: List[AudioFileInfo] = []
        self.matches: List[AudioMatch] = []
        
        # Prepare audio file info
        self._prepare_audio_file_info()
    
    def _prepare_audio_file_info(self):
        """Prepare detailed info about audio files"""
        for i, file_path in enumerate(self.audio_files):
            try:
                path_obj = Path(file_path)
                stat = path_obj.stat()
                
                # Try to extract date from filename (e.g., PTT-20241009-WA0010.opus)
                filename_date = self._extract_date_from_filename(path_obj.name)
                
                info = AudioFileInfo(
                    index=i,
                    file_path=file_path,
                    filename=path_obj.name,
                    size_bytes=stat.st_size,
                    created_time=filename_date if filename_date else datetime.fromtimestamp(stat.st_ctime),
                    modified_time=datetime.fromtimestamp(stat.st_mtime),
                    extension=path_obj.suffix.lower()
                )
                self.audio_file_info.append(info)
                
            except Exception as e:
                logger.warning(f"Could not get info for {file_path}: {e}")
    
    def _extract_date_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract date from WhatsApp audio filename (e.g., PTT-20241009-WA0010.opus)"""
        import re
        
        # Pattern for WhatsApp audio files: PTT-YYYYMMDD-WA####.opus
        pattern = r'PTT-(\d{8})-WA\d+'
        match = re.search(pattern, filename)
        
        if match:
            date_str = match.group(1)  # e.g., "20241009"
            try:
                # Parse YYYYMMDD format
                return datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                logger.warning(f"Could not parse date from filename: {filename}")
        
        return None
    
    def match_audio_files(self) -> Dict[str, any]:
        """
        Match audio files to chat messages using multiple strategies
        
        Returns:
            Dict with matching results
        """
        try:
            logger.info(f"Matching {len(self.audio_file_info)} audio files to {len(self.chat_audio_messages)} chat messages")
            
            # Strategy 1: Try filename date matching first
            filename_matches = self._try_filename_date_matching()
            logger.info(f"Filename date matching found {filename_matches} matches")
            
            if filename_matches >= len(self.audio_file_info) * 0.75:  # If 75%+ files matched
                match_method = "filename_date_matching"
                logger.info("Using filename date matching as primary method")
                
            # Strategy 2: Perfect count match - use sequential order
            elif len(self.audio_file_info) == len(self.chat_audio_messages):
                self._match_by_sequential_order()
                match_method = "sequential_perfect_count"
                
            # Strategy 3: More chat messages than files - use timestamp matching
            elif len(self.chat_audio_messages) > len(self.audio_file_info):
                if filename_matches == 0:  # Only if filename matching didn't work
                    self._match_by_timestamp_correlation()
                    match_method = "timestamp_correlation"
                else:
                    match_method = "filename_date_matching"
                
            # Strategy 4: More files than chat messages - use best available matching
            else:
                self._match_by_best_available()
                match_method = "best_available"
            
            # Validate matches
            self._validate_matches()
            
            return {
                "success": True,
                "method_used": match_method,
                "matches_found": len(self.matches),
                "total_audio_files": len(self.audio_file_info),
                "total_chat_messages": len(self.chat_audio_messages),
                "match_quality": self._calculate_match_quality()
            }
            
        except Exception as e:
            logger.error(f"Audio matching failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _try_exact_filename_matching(self) -> int:
        """Try to match using exact filenames mentioned in chat messages"""
        logger.info("Trying exact filename matching")
        
        matches_found = 0
        used_chat_indices = set()
        
        for audio_file in self.audio_file_info:
            audio_filename = Path(audio_file.file_path).name
            
            # Look for chat messages that mention this exact filename
            for i, chat_msg in enumerate(self.chat_audio_messages):
                if i in used_chat_indices:
                    continue
                
                chat_content = chat_msg.get('original_content', '')
                
                # Check if the filename appears in the chat message
                if audio_filename in chat_content:
                    match = AudioMatch(
                        chat_index=i,
                        file_index=audio_file.index,
                        chat_message=chat_msg,
                        audio_file_path=audio_file.file_path,
                        confidence=1.0,  # Very high confidence for exact matches
                        match_method="exact_filename_matching",
                        time_diff_minutes=0  # Perfect match
                    )
                    
                    self.matches.append(match)
                    used_chat_indices.add(i)
                    matches_found += 1
                    
                    logger.info(f"Exact filename match: {audio_filename} -> {chat_msg.get('sender', 'Unknown')}")
                    break
        
        logger.info(f"Exact filename matching found {matches_found} matches")
        return matches_found
    
    def _try_filename_date_matching(self) -> int:
        """Try to match using dates extracted from filenames"""
        logger.info("Trying filename date matching")
        
        matches_found = 0
        used_chat_indices = set()
        
        for audio_file in self.audio_file_info:
            # Check if we extracted a valid date from filename
            # The date should be reasonable (after 2020 and before 2030)
            if not audio_file.created_time or audio_file.created_time.year < 2020 or audio_file.created_time.year > 2030:
                logger.debug(f"Skipping {audio_file.filename} - no valid date extracted")
                continue
                
            logger.debug(f"Processing {audio_file.filename} with date {audio_file.created_time.date()}")
            
            best_match = None
            best_score = float('inf')
            best_chat_index = -1
            
            # Find the chat message with closest date
            for i, chat_msg in enumerate(self.chat_audio_messages):
                if i in used_chat_indices:
                    continue
                    
                chat_time = self._parse_chat_timestamp(chat_msg.get('timestamp'))
                if not chat_time:
                    continue
                
                # Calculate time difference in days (only using date, not time)
                audio_date = audio_file.created_time.date()
                chat_date = chat_time.date()
                time_diff_days = abs((audio_date - chat_date).days)
                
                logger.debug(f"  vs {chat_msg.get('sender')} {chat_date}: {time_diff_days} days diff")
                
                # Prefer matches within reasonable time windows
                if time_diff_days < best_score:
                    best_score = time_diff_days
                    best_match = chat_msg
                    best_chat_index = i
            
            if best_match and best_score <= 7:  # Within 7 days
                confidence = max(0.3, 1.0 - (best_score / 7))  # Higher confidence for closer dates
                
                match = AudioMatch(
                    chat_index=best_chat_index,
                    file_index=audio_file.index,
                    chat_message=best_match,
                    audio_file_path=audio_file.file_path,
                    confidence=confidence,
                    match_method="filename_date_matching",
                    time_diff_minutes=best_score * 24 * 60  # Convert days to minutes for display
                )
                
                self.matches.append(match)
                used_chat_indices.add(best_chat_index)
                matches_found += 1
                
                logger.info(f"Filename date match: {audio_file.filename} -> {best_match.get('sender', 'Unknown')} (±{best_score} days)")
            else:
                logger.debug(f"No good match found for {audio_file.filename} (best: {best_score} days)")
        
        logger.info(f"Filename date matching found {matches_found} matches")
        return matches_found
        """Match by sequential order when counts are equal"""
        logger.info("Using sequential order matching (perfect count match)")
        
        for i in range(len(self.audio_file_info)):
            audio_file = self.audio_file_info[i]
            chat_msg = self.chat_audio_messages[i]
            
            # Calculate time difference for validation
            chat_time = self._parse_chat_timestamp(chat_msg.get('timestamp'))
            time_diff = None
            if chat_time:
                time_diff = abs((audio_file.created_time - chat_time).total_seconds() / 60)
            
            match = AudioMatch(
                chat_index=i,
                file_index=i,
                chat_message=chat_msg,
                audio_file_path=audio_file.file_path,
                confidence=0.9,  # High confidence for perfect count
                match_method="sequential_order",
                time_diff_minutes=time_diff
            )
            
            self.matches.append(match)
            logger.debug(f"Sequential match {i}: {audio_file.filename} -> {chat_msg.get('sender', 'Unknown')}")
    
    def _match_by_timestamp_correlation(self):
        """Match by correlating timestamps when more chat messages than files"""
        logger.info("Using timestamp correlation matching")
        
        # Sort audio files by creation time
        sorted_audio = sorted(self.audio_file_info, key=lambda x: x.created_time)
        
        used_chat_indices = set()
        
        for audio_file in sorted_audio:
            best_match = None
            best_score = float('inf')
            best_chat_index = -1
            
            # Find the chat message with closest timestamp
            for i, chat_msg in enumerate(self.chat_audio_messages):
                if i in used_chat_indices:
                    continue
                    
                chat_time = self._parse_chat_timestamp(chat_msg.get('timestamp'))
                if not chat_time:
                    continue
                
                # Calculate time difference in minutes
                time_diff = abs((audio_file.created_time - chat_time).total_seconds() / 60)
                
                # Prefer matches within reasonable time windows
                if time_diff < best_score:
                    best_score = time_diff
                    best_match = chat_msg
                    best_chat_index = i
            
            if best_match and best_score < 1440:  # Within 24 hours
                confidence = max(0.1, 1.0 - (best_score / 1440))  # Confidence decreases with time diff
                
                match = AudioMatch(
                    chat_index=best_chat_index,
                    file_index=audio_file.index,
                    chat_message=best_match,
                    audio_file_path=audio_file.file_path,
                    confidence=confidence,
                    match_method="timestamp_correlation",
                    time_diff_minutes=best_score
                )
                
                self.matches.append(match)
                used_chat_indices.add(best_chat_index)
                logger.debug(f"Timestamp match: {audio_file.filename} -> {best_match.get('sender', 'Unknown')} (±{best_score:.1f}min)")
    
    def _match_by_best_available(self):
        """Match using best available strategy when more files than messages"""
        logger.info("Using best available matching")
        
        # For now, use sequential order but only match up to the number of chat messages
        max_matches = min(len(self.audio_file_info), len(self.chat_audio_messages))
        
        for i in range(max_matches):
            audio_file = self.audio_file_info[i]
            chat_msg = self.chat_audio_messages[i]
            
            match = AudioMatch(
                chat_index=i,
                file_index=i,
                chat_message=chat_msg,
                audio_file_path=audio_file.file_path,
                confidence=0.7,  # Medium confidence
                match_method="best_available",
                time_diff_minutes=None
            )
            
            self.matches.append(match)
            logger.debug(f"Best available match {i}: {audio_file.filename}")
    
    def _parse_chat_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse chat timestamp string into datetime"""
        if not timestamp_str:
            return None
            
        try:
            # Handle ISO format from chat parser
            if 'T' in timestamp_str:
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                return datetime.fromisoformat(timestamp_str)
        except Exception as e:
            logger.warning(f"Could not parse timestamp: {timestamp_str} - {e}")
            return None
    
    def _validate_matches(self):
        """Validate and improve matches"""
        valid_matches = []
        
        for match in self.matches:
            # Basic validation
            if match.confidence > 0.1:  # Minimum confidence threshold
                valid_matches.append(match)
            else:
                logger.warning(f"Rejecting low confidence match: {match.confidence}")
        
        self.matches = valid_matches
        logger.info(f"Validated {len(self.matches)} matches")
    
    def _calculate_match_quality(self) -> Dict[str, any]:
        """Calculate overall match quality metrics"""
        if not self.matches:
            return {"overall_quality": 0.0, "avg_confidence": 0.0}
        
        avg_confidence = sum(m.confidence for m in self.matches) / len(self.matches)
        
        # Calculate overall quality based on various factors
        coverage = len(self.matches) / max(len(self.audio_file_info), len(self.chat_audio_messages))
        overall_quality = (avg_confidence * 0.7) + (coverage * 0.3)
        
        return {
            "overall_quality": round(overall_quality, 3),
            "avg_confidence": round(avg_confidence, 3),
            "coverage": round(coverage, 3),
            "match_rate": f"{len(self.matches)}/{len(self.audio_file_info)}"
        }
    
    def get_match_summary(self) -> List[Dict[str, any]]:
        """Get summary of all matches for review"""
        summary = []
        
        for match in self.matches:
            audio_file = self.audio_file_info[match.file_index]
            
            summary_item = {
                "match_id": len(summary),
                "audio_file": audio_file.filename,
                "audio_path": match.audio_file_path,
                "file_size_mb": round(audio_file.size_bytes / (1024*1024), 2),
                "chat_sender": match.chat_message.get('sender', 'Unknown'),
                "chat_timestamp": match.chat_message.get('timestamp', 'Unknown'),
                "chat_line": match.chat_message.get('line_number', -1),
                "confidence": match.confidence,
                "match_method": match.match_method,
                "time_diff_minutes": match.time_diff_minutes
            }
            summary.append(summary_item)
        
        return summary
    
    def get_transcription_queue(self) -> List[Dict[str, any]]:
        """Get ordered list of files ready for transcription"""
        queue = []
        
        # Sort matches by chat message order
        sorted_matches = sorted(self.matches, key=lambda x: x.chat_index)
        
        for i, match in enumerate(sorted_matches):
            queue_item = {
                "transcription_order": i,
                "audio_file_path": match.audio_file_path,
                "chat_index": match.chat_index,
                "chat_message": match.chat_message,
                "match_confidence": match.confidence,
                "audio_filename": Path(match.audio_file_path).name
            }
            queue.append(queue_item)
        
        return queue


def match_audio_to_chat(audio_files: List[str], chat_audio_messages: List[Dict]) -> Tuple[AudioMatcher, Dict[str, any]]:
    """
    Convenience function to match audio files to chat messages
    
    Args:
        audio_files: List of audio file paths
        chat_audio_messages: List of chat audio message data
        
    Returns:
        Tuple of (matcher_instance, match_results)
    """
    matcher = AudioMatcher(audio_files, chat_audio_messages)
    results = matcher.match_audio_files()
    return matcher, results


# Test function
if __name__ == "__main__":
    # Example usage
    print("Audio Matcher module - run via test_audio_matcher.py")