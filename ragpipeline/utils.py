"""
Utility functions for document intelligence system.
"""

import re
import os
import json
from typing import List, Dict, Any, Tuple
import logging
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

class TextProcessor:
    """Text processing utilities."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-"]', '', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with spaces
        
        return text.strip()
    
    def extract_key_phrases(self, text: str, n_phrases: int = 20) -> List[str]:
        """Extract key phrases from text."""
        text = text.lower()
        
        # Extract noun phrases using simple patterns
        noun_phrases = []
        
        # Pattern for adjective + noun combinations
        adj_noun_pattern = r'\b([a-z]+(?:al|ic|ive|ous|ful|less|able))\s+([a-z]+(?:tion|sion|ment|ness|ity|ing|ed)?)\b'
        matches = re.findall(adj_noun_pattern, text)
        noun_phrases.extend([f"{adj} {noun}" for adj, noun in matches])
        
        # Pattern for noun + noun combinations
        noun_noun_pattern = r'\b([a-z]+(?:tion|sion|ment|ness|ity|er|or))\s+([a-z]+(?:tion|sion|ment|ness|ity|ing|ed)?)\b'
        matches = re.findall(noun_noun_pattern, text)
        noun_phrases.extend([f"{noun1} {noun2}" for noun1, noun2 in matches])
        
        # Count frequency and return top phrases
        phrase_counts = Counter(noun_phrases)
        return [phrase for phrase, _ in phrase_counts.most_common(n_phrases)]
    
    def calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        if not sentences or not words:
            return 0.0
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Lexical diversity
        unique_words = set(word.lower() for word in words if word.isalpha())
        lexical_diversity = len(unique_words) / len(words) if words else 0
        
        # Complex word ratio (words > 6 characters)
        complex_words = [word for word in words if len(word) > 6 and word.isalpha()]
        complex_ratio = len(complex_words) / len(words) if words else 0
        
        # Combine metrics (normalized to 0-100)
        complexity = (
            min(avg_sentence_length / 20, 1) * 30 +  # Sentence length component
            lexical_diversity * 40 +                  # Diversity component
            complex_ratio * 30                        # Complexity component
        )
        
        return complexity

class SectionClassifier:
    """Classify document sections based on content type."""
    
    SECTION_TYPES = {
        'abstract': ['abstract', 'summary', 'overview'],
        'introduction': ['introduction', 'background', 'motivation'],
        'methodology': ['method', 'approach', 'technique', 'procedure'],
        'results': ['result', 'finding', 'outcome', 'data'],
        'discussion': ['discussion', 'analysis', 'interpretation'],
        'conclusion': ['conclusion', 'summary', 'future work'],
        'references': ['reference', 'bibliography', 'citation'],
        'financial': ['revenue', 'profit', 'financial', 'earnings', 'cost'],
        'technical': ['technical', 'implementation', 'system', 'algorithm'],
        'business': ['strategy', 'market', 'business', 'competitive', 'customer']
    }
    
    def classify_section(self, title: str, content: str) -> str:
        """Classify section type based on title and content."""
        title_lower = title.lower()
        content_lower = content.lower()
        
        best_type = 'general'
        best_score = 0
        
        for section_type, keywords in self.SECTION_TYPES.items():
            score = 0
            
            # Check title keywords (higher weight)
            for keyword in keywords:
                if keyword in title_lower:
                    score += 3
                if keyword in content_lower[:200]:  # Check beginning of content
                    score += 1
            
            if score > best_score:
                best_score = score
                best_type = section_type
        
        return best_type

class OutputFormatter:
    """Format output according to specifications."""
    
    @staticmethod
    def format_output(sections: List[Any], subsections: List[Any], 
                     metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Format final output according to challenge specifications."""
        return {
            "metadata": metadata,
            "extracted_sections": [
                {
                    "document": section.document_name,
                    "page_number": section.page_number,
                    "section_title": section.section_title,
                    "importance_rank": getattr(section, 'importance_rank', i + 1),
                    "content_length": len(section.content),
                    "section_type": getattr(section, 'section_type', 'general'),
                    "relevance_score": round(section.importance_score, 2)
                }
                for i, section in enumerate(sections)
            ],
            "subsection_analysis": [
                {
                    "document": subsection.document_name,
                    "page_number": subsection.page_number,
                    "refined_text": subsection.refined_text,
                    "relevance_score": round(subsection.relevance_score, 2),
                    "original_length": len(subsection.content),
                    "refined_length": len(subsection.refined_text)
                }
                for subsection in subsections
            ]
        }

class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = None
        self.checkpoints = {}
    
    def start_monitoring(self):
        """Start performance monitoring."""
        import time
        self.start_time = time.time()
        self.logger.info("Performance monitoring started")
    
    def checkpoint(self, name: str):
        """Add a performance checkpoint."""
        import time
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.checkpoints[name] = elapsed
            self.logger.info(f"Checkpoint '{name}': {elapsed:.2f}s")
    
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary."""
        return self.checkpoints.copy()

class ConfigManager:
    """Manage system configuration."""
    
    DEFAULT_CONFIG = {
        "max_sections_per_document": 10,
        "max_subsections_total": 15,
        "min_section_length": 50,
        "min_subsection_length": 100,
        "relevance_threshold": 30,
        "max_content_preview_length": 200,
        "sentence_model_name": "all-MiniLM-L6-v2",
        "processing_timeout": 60,
    }
    
    def __init__(self, config_path: str = None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            self.config.update(user_config)
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value."""
        self.config[key] = value

def validate_input_files(pdf_paths: List[str]) -> Tuple[bool, str]:
    """Validate input PDF files."""
    if not pdf_paths:
        return False, "No PDF files provided"
    
    if len(pdf_paths) > 10:
        return False, "Too many PDF files (maximum 10 allowed)"
    
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            return False, f"PDF file not found: {pdf_path}"
        
        if not pdf_path.lower().endswith('.pdf'):
            return False, f"File is not a PDF: {pdf_path}"
        
        # Check file size (max 50MB per file)
        file_size = os.path.getsize(pdf_path)
        if file_size > 50 * 1024 * 1024:
            return False, f"PDF file too large (>50MB): {pdf_path}"
    
    return True, "All files valid"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe processing."""
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:96] + ext
    
    return filename

def estimate_processing_time(pdf_paths: List[str]) -> float:
    """Estimate processing time based on file sizes."""
    total_size = sum(os.path.getsize(path) for path in pdf_paths)
    
    # Rough estimate: 1MB per second base + overhead
    base_time = total_size / (1024 * 1024)  # MB
    overhead = len(pdf_paths) * 2  # 2 seconds per file overhead
    
    return base_time + overhead