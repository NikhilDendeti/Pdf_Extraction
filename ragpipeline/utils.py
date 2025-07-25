# """
# Utility functions for document intelligence system.
# """

# import re
# import os
# import json
# from typing import List, Dict, Any, Tuple
# import logging
# from collections import Counter
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords

# class TextProcessor:
#     """Text processing utilities."""
    
#     def __init__(self):
#         self.stop_words = set(stopwords.words('english'))
        
#     def clean_text(self, text: str) -> str:
#         """Clean and normalize text."""
#         # Remove extra whitespace
#         text = re.sub(r'\s+', ' ', text)
        
#         # Remove special characters but keep punctuation
#         text = re.sub(r'[^\w\s.,!?;:()\-"]', '', text)
        
#         # Fix common PDF extraction issues
#         text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words
#         text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with spaces
        
#         return text.strip()
    
#     def extract_key_phrases(self, text: str, n_phrases: int = 20) -> List[str]:
#         """Extract key phrases from text."""
#         text = text.lower()
        
#         # Extract noun phrases using simple patterns
#         noun_phrases = []
        
#         # Pattern for adjective + noun combinations
#         adj_noun_pattern = r'\b([a-z]+(?:al|ic|ive|ous|ful|less|able))\s+([a-z]+(?:tion|sion|ment|ness|ity|ing|ed)?)\b'
#         matches = re.findall(adj_noun_pattern, text)
#         noun_phrases.extend([f"{adj} {noun}" for adj, noun in matches])
        
#         # Pattern for noun + noun combinations
#         noun_noun_pattern = r'\b([a-z]+(?:tion|sion|ment|ness|ity|er|or))\s+([a-z]+(?:tion|sion|ment|ness|ity|ing|ed)?)\b'
#         matches = re.findall(noun_noun_pattern, text)
#         noun_phrases.extend([f"{noun1} {noun2}" for noun1, noun2 in matches])
        
#         # Count frequency and return top phrases
#         phrase_counts = Counter(noun_phrases)
#         return [phrase for phrase, _ in phrase_counts.most_common(n_phrases)]
    
#     def calculate_text_complexity(self, text: str) -> float:
#         """Calculate text complexity score."""
#         sentences = sent_tokenize(text)
#         words = word_tokenize(text)
        
#         if not sentences or not words:
#             return 0.0
        
#         # Average sentence length
#         avg_sentence_length = len(words) / len(sentences)
        
#         # Lexical diversity
#         unique_words = set(word.lower() for word in words if word.isalpha())
#         lexical_diversity = len(unique_words) / len(words) if words else 0
        
#         # Complex word ratio (words > 6 characters)
#         complex_words = [word for word in words if len(word) > 6 and word.isalpha()]
#         complex_ratio = len(complex_words) / len(words) if words else 0
        
#         # Combine metrics (normalized to 0-100)
#         complexity = (
#             min(avg_sentence_length / 20, 1) * 30 +  # Sentence length component
#             lexical_diversity * 40 +                  # Diversity component
#             complex_ratio * 30                        # Complexity component
#         )
        
#         return complexity

# class SectionClassifier:
#     """Classify document sections based on content type."""
    
#     SECTION_TYPES = {
#         'abstract': ['abstract', 'summary', 'overview'],
#         'introduction': ['introduction', 'background', 'motivation'],
#         'methodology': ['method', 'approach', 'technique', 'procedure'],
#         'results': ['result', 'finding', 'outcome', 'data'],
#         'discussion': ['discussion', 'analysis', 'interpretation'],
#         'conclusion': ['conclusion', 'summary', 'future work'],
#         'references': ['reference', 'bibliography', 'citation'],
#         'financial': ['revenue', 'profit', 'financial', 'earnings', 'cost'],
#         'technical': ['technical', 'implementation', 'system', 'algorithm'],
#         'business': ['strategy', 'market', 'business', 'competitive', 'customer']
#     }
    
#     def classify_section(self, title: str, content: str) -> str:
#         """Classify section type based on title and content."""
#         title_lower = title.lower()
#         content_lower = content.lower()
        
#         best_type = 'general'
#         best_score = 0
        
#         for section_type, keywords in self.SECTION_TYPES.items():
#             score = 0
            
#             # Check title keywords (higher weight)
#             for keyword in keywords:
#                 if keyword in title_lower:
#                     score += 3
#                 if keyword in content_lower[:200]:  # Check beginning of content
#                     score += 1
            
#             if score > best_score:
#                 best_score = score
#                 best_type = section_type
        
#         return best_type

# class OutputFormatter:
#     """Format output according to specifications."""
    
#     @staticmethod
#     def format_output(sections: List[Any], subsections: List[Any], 
#                      metadata: Dict[str, Any]) -> Dict[str, Any]:
#         """Format final output according to challenge specifications."""
#         return {
#             "metadata": metadata,
#             "extracted_sections": [
#                 {
#                     "document": section.document_name,
#                     "page_number": section.page_number,
#                     "section_title": section.section_title,
#                     "importance_rank": getattr(section, 'importance_rank', i + 1),
#                     "content_length": len(section.content),
#                     "section_type": getattr(section, 'section_type', 'general'),
#                     "relevance_score": round(section.importance_score, 2)
#                 }
#                 for i, section in enumerate(sections)
#             ],
#             "subsection_analysis": [
#                 {
#                     "document": subsection.document_name,
#                     "page_number": subsection.page_number,
#                     "refined_text": subsection.refined_text,
#                     "relevance_score": round(subsection.relevance_score, 2),
#                     "original_length": len(subsection.content),
#                     "refined_length": len(subsection.refined_text)
#                 }
#                 for subsection in subsections
#             ]
#         }

# class PerformanceMonitor:
#     """Monitor system performance and resource usage."""
    
#     def __init__(self):
#         self.logger = logging.getLogger(__name__)
#         self.start_time = None
#         self.checkpoints = {}
    
#     def start_monitoring(self):
#         """Start performance monitoring."""
#         import time
#         self.start_time = time.time()
#         self.logger.info("Performance monitoring started")
    
#     def checkpoint(self, name: str):
#         """Add a performance checkpoint."""
#         import time
#         if self.start_time:
#             elapsed = time.time() - self.start_time
#             self.checkpoints[name] = elapsed
#             self.logger.info(f"Checkpoint '{name}': {elapsed:.2f}s")
    
#     def get_summary(self) -> Dict[str, float]:
#         """Get performance summary."""
#         return self.checkpoints.copy()

# class ConfigManager:
#     """Manage system configuration."""
    
#     DEFAULT_CONFIG = {
#         "max_sections_per_document": 10,
#         "max_subsections_total": 15,
#         "min_section_length": 50,
#         "min_subsection_length": 100,
#         "relevance_threshold": 30,
#         "max_content_preview_length": 200,
#         "sentence_model_name": "all-MiniLM-L6-v2",
#         "processing_timeout": 60,
#     }
    
#     def __init__(self, config_path: str = None):
#         self.config = self.DEFAULT_CONFIG.copy()
#         if config_path and os.path.exists(config_path):
#             self.load_config(config_path)
    
#     def load_config(self, config_path: str):
#         """Load configuration from file."""
#         try:
#             with open(config_path, 'r') as f:
#                 user_config = json.load(f)
#             self.config.update(user_config)
#         except Exception as e:
#             logging.warning(f"Failed to load config from {config_path}: {e}")
    
#     def get(self, key: str, default=None):
#         """Get configuration value."""
#         return self.config.get(key, default)
    
#     def set(self, key: str, value):
#         """Set configuration value."""
#         self.config[key] = value

# def validate_input_files(pdf_paths: List[str]) -> Tuple[bool, str]:
#     """Validate input PDF files."""
#     if not pdf_paths:
#         return False, "No PDF files provided"
    
#     if len(pdf_paths) > 10:
#         return False, "Too many PDF files (maximum 10 allowed)"
    
#     for pdf_path in pdf_paths:
#         if not os.path.exists(pdf_path):
#             return False, f"PDF file not found: {pdf_path}"
        
#         if not pdf_path.lower().endswith('.pdf'):
#             return False, f"File is not a PDF: {pdf_path}"
        
#         # Check file size (max 50MB per file)
#         file_size = os.path.getsize(pdf_path)
#         if file_size > 50 * 1024 * 1024:
#             return False, f"PDF file too large (>50MB): {pdf_path}"
    
#     return True, "All files valid"

# def sanitize_filename(filename: str) -> str:
#     """Sanitize filename for safe processing."""
#     # Remove path components
#     filename = os.path.basename(filename)
    
#     # Replace invalid characters
#     filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
#     # Limit length
#     if len(filename) > 100:
#         name, ext = os.path.splitext(filename)
#         filename = name[:96] + ext
    
#     return filename

# def estimate_processing_time(pdf_paths: List[str]) -> float:
#     """Estimate processing time based on file sizes."""
#     total_size = sum(os.path.getsize(path) for path in pdf_paths)
    
#     # Rough estimate: 1MB per second base + overhead
#     base_time = total_size / (1024 * 1024)  # MB
#     overhead = len(pdf_paths) * 2  # 2 seconds per file overhead
    
#     return base_time + overhead

# # Add this to utils.py or a separate file and import it into main.py

# class DocumentIntelligenceSystem:
#     def __init__(self):
#         print("[INIT] DocumentIntelligenceSystem initialized")

#     def process_documents(self, pdf_paths: List[str], persona: str, job: str) -> Dict[str, Any]:
#         print(f"[PROCESSING] Persona: {persona}")
#         print(f"[PROCESSING] Job: {job}")
#         print(f"[PROCESSING] PDF files: {pdf_paths}")

#         # Simulate some dummy processing
#         import time
#         start_time = time.time()

#         extracted_sections = []
#         for i, path in enumerate(pdf_paths):
#             print(f"⏳ Processing {path}")
#             extracted_sections.append({
#                 "document": os.path.basename(path),
#                 "page_number": 1,
#                 "section_title": f"Dummy Section {i+1}",
#                 "importance_rank": i+1,
#                 "content_length": 1234,
#                 "section_type": "general",
#                 "relevance_score": 75.0
#             })

#         duration = time.time() - start_time
#         return {
#             "metadata": {
#                 "processing_time_seconds": round(duration, 2)
#             },
#             "extracted_sections": extracted_sections,
#             "subsection_analysis": []
#         }



import json
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

@dataclass
class DocumentSection:
    document_name: str
    page_number: int
    section_title: str
    content: str
    importance_score: float
    section_type: str = "content"

@dataclass
class SubSection:
    document_name: str
    page_number: int
    content: str
    refined_text: str
    relevance_score: float

class DocumentIntelligenceSystem:
    def __init__(self):
        """Initialize the document intelligence system with lightweight models."""
        self.setup_logging()
        self.sentence_model = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.stop_words = set(stopwords.words('english'))
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_sentence_model(self):
        """Load sentence transformer model lazily."""
        if self.sentence_model is None:
            try:
                # Use a lightweight model that's under 500MB
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Sentence transformer model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load sentence transformer: {e}")
                self.sentence_model = None

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF with page information."""
        try:
            doc = fitz.open(pdf_path)
            pages_text = {}
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                pages_text[page_num + 1] = text
                
            doc.close()
            return pages_text
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {e}")
            return {}

    def identify_sections(self, pages_text: Dict[int, str], doc_name: str) -> List[DocumentSection]:
        """Identify and extract sections from document pages."""
        sections = []
        
        for page_num, text in pages_text.items():
            if not text.strip():
                continue
                
            # Split text into potential sections based on common patterns
            section_patterns = [
                r'\n([A-Z][A-Z\s]{10,50})\n',  # ALL CAPS headers
                r'\n(\d+\.?\s+[A-Z][^.\n]{10,100})\n',  # Numbered sections
                r'\n([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:?\s*)\n',  # Title case headers
                r'\n(Abstract|Introduction|Methodology|Results|Discussion|Conclusion|References)\b',  # Academic sections
            ]
            
            sections_found = []
            for pattern in section_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    sections_found.append((match.start(), match.group(1).strip()))
            
            # If no clear sections found, create sections based on paragraphs
            if not sections_found:
                paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
                for i, para in enumerate(paragraphs[:5]):  # Limit to first 5 paragraphs
                    section_title = f"Section {i+1}"
                    if len(para) > 100:
                        sections.append(DocumentSection(
                            document_name=doc_name,
                            page_number=page_num,
                            section_title=section_title,
                            content=para,
                            importance_score=0.0
                        ))
            else:
                # Sort sections by position in text
                sections_found.sort(key=lambda x: x[0])
                
                for i, (pos, title) in enumerate(sections_found):
                    # Extract content for this section
                    start_pos = pos
                    end_pos = sections_found[i+1][0] if i+1 < len(sections_found) else len(text)
                    content = text[start_pos:end_pos].strip()
                    
                    if len(content) > 50:  # Minimum content length
                        sections.append(DocumentSection(
                            document_name=doc_name,
                            page_number=page_num,
                            section_title=title,
                            content=content,
                            importance_score=0.0
                        ))
        
        return sections

    def extract_persona_keywords(self, persona_description: str, job_description: str) -> List[str]:
        """Extract relevant keywords from persona and job descriptions."""
        combined_text = f"{persona_description} {job_description}"
        
        # Remove stopwords and extract meaningful terms
        words = word_tokenize(combined_text.lower())
        keywords = [word for word in words if word.isalpha() and word not in self.stop_words and len(word) > 3]
        
        # Detect domain and add domain-specific keywords
        from config import DOMAIN_KEYWORDS, PERSONA_PREFERENCES
        
        text_lower = combined_text.lower()
        domain_keywords = []
        
        # Travel planning specific detection
        if any(term in text_lower for term in ['travel', 'trip', 'planner', 'vacation', 'tour']):
            domain_keywords.extend(DOMAIN_KEYWORDS.get('travel', []))
            domain_keywords.extend(DOMAIN_KEYWORDS.get('planning', []))
        
        # Academic detection
        if any(term in text_lower for term in ['research', 'phd', 'academic', 'study']):
            domain_keywords.extend(DOMAIN_KEYWORDS.get('academic', []))
        
        # Business detection
        if any(term in text_lower for term in ['business', 'analyst', 'investment', 'financial']):
            domain_keywords.extend(DOMAIN_KEYWORDS.get('business', []))
            domain_keywords.extend(DOMAIN_KEYWORDS.get('financial', []))
        
        # Technical detection
        if any(term in text_lower for term in ['technical', 'software', 'system', 'engineering']):
            domain_keywords.extend(DOMAIN_KEYWORDS.get('technical', []))
        
        # Extract specific patterns for travel planning
        travel_patterns = [
            r'\b(plan|planning|itinerary|schedule|organize)\w*\b',
            r'\b(day|days|trip|vacation|holiday|travel)\w*\b',
            r'\b(group|friends|college|student|young)\w*\b',
            r'\b(budget|affordable|cost|price|cheap)\w*\b',
            r'\b(hotel|restaurant|activity|attraction|city)\w*\b',
            r'\b(food|cuisine|dining|culture|history)\w*\b'
        ]
        
        for pattern in travel_patterns:
            matches = re.findall(pattern, combined_text.lower())
            keywords.extend(matches)
        
        # Add domain-specific keywords
        keywords.extend(domain_keywords)
        
        # Extract important terms using general patterns
        important_patterns = [
            r'\b(research|analysis|study|review|methodology|data|performance|trend|strategy)\w*\b',
            r'\b(financial|technical|academic|business|scientific|clinical)\w*\b',
            r'\b(PhD|analyst|student|researcher|manager|specialist|planner)\w*\b'
        ]
        
        for pattern in important_patterns:
            matches = re.findall(pattern, combined_text.lower())
            keywords.extend(matches)
        
        # Remove duplicates and return top keywords
        unique_keywords = list(set(keywords))
        
        # Prioritize keywords by frequency and importance
        keyword_freq = {}
        for keyword in unique_keywords:
            # Count frequency in combined text
            freq = combined_text.lower().count(keyword)
            # Boost domain-specific keywords
            if keyword in domain_keywords:
                freq *= 2
            keyword_freq[keyword] = freq
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, freq in sorted_keywords[:25]]

    def calculate_relevance_score(self, section: DocumentSection, persona_keywords: List[str], 
                                job_description: str) -> float:
        """Calculate relevance score for a section based on persona and job requirements."""
        content_lower = section.content.lower()
        title_lower = section.section_title.lower()
        
        # Keyword matching score (0-40 points)
        keyword_matches = sum(1 for keyword in persona_keywords if keyword in content_lower)
        keyword_score = min(keyword_matches * 2, 40)
        
        # Title relevance score (0-20 points)
        title_matches = sum(1 for keyword in persona_keywords if keyword in title_lower)
        title_score = min(title_matches * 5, 20)
        
        # Content length and quality score (0-20 points)
        content_quality = min(len(section.content) / 500 * 10, 20)
        
        # Semantic similarity score (if model available) (0-20 points)
        semantic_score = 0
        if self.sentence_model:
            try:
                job_embedding = self.sentence_model.encode([job_description])
                content_embedding = self.sentence_model.encode([section.content[:512]])  # Limit length
                similarity = cosine_similarity(job_embedding, content_embedding)[0][0]
                semantic_score = similarity * 20
            except Exception as e:
                self.logger.warning(f"Semantic similarity calculation failed: {e}")
        
        total_score = keyword_score + title_score + content_quality + semantic_score
        return min(total_score, 100)  # Cap at 100

    def rank_sections(self, sections: List[DocumentSection], persona_description: str, 
                     job_description: str) -> List[DocumentSection]:
        """Rank sections based on relevance to persona and job."""
        self.load_sentence_model()
        persona_keywords = self.extract_persona_keywords(persona_description, job_description)
        
        # Calculate relevance scores
        for section in sections:
            section.importance_score = self.calculate_relevance_score(
                section, persona_keywords, job_description
            )
        
        # Sort by importance score (descending)
        sections.sort(key=lambda x: x.importance_score, reverse=True)
        
        # Add importance rank
        for i, section in enumerate(sections):
            section.importance_rank = i + 1
            
        return sections

    def extract_subsections(self, top_sections: List[DocumentSection], 
                          persona_description: str, job_description: str) -> List[SubSection]:
        """Extract and refine subsections from top-ranked sections."""
        subsections = []
        persona_keywords = self.extract_persona_keywords(persona_description, job_description)
        
        for section in top_sections[:10]:  # Process top 10 sections
            # Split content into smaller chunks
            sentences = sent_tokenize(section.content)
            
            # Group sentences into meaningful subsections
            current_subsection = []
            for sentence in sentences:
                current_subsection.append(sentence)
                
                # Create subsection when we have 2-4 sentences or reach certain length
                if len(current_subsection) >= 2 and len(' '.join(current_subsection)) > 200:
                    subsection_text = ' '.join(current_subsection)
                    refined_text = self.refine_text(subsection_text, persona_keywords)
                    
                    # Calculate relevance score for subsection
                    relevance_score = self.calculate_subsection_relevance(
                        subsection_text, persona_keywords, job_description
                    )
                    
                    if relevance_score > 30:  # Only include relevant subsections
                        subsections.append(SubSection(
                            document_name=section.document_name,
                            page_number=section.page_number,
                            content=subsection_text,
                            refined_text=refined_text,
                            relevance_score=relevance_score
                        ))
                    
                    current_subsection = []
            
            # Handle remaining sentences
            if current_subsection and len(' '.join(current_subsection)) > 100:
                subsection_text = ' '.join(current_subsection)
                refined_text = self.refine_text(subsection_text, persona_keywords)
                relevance_score = self.calculate_subsection_relevance(
                    subsection_text, persona_keywords, job_description
                )
                
                if relevance_score > 30:
                    subsections.append(SubSection(
                        document_name=section.document_name,
                        page_number=section.page_number,
                        content=subsection_text,
                        refined_text=refined_text,
                        relevance_score=relevance_score
                    ))
        
        # Sort by relevance score and return top subsections
        subsections.sort(key=lambda x: x.relevance_score, reverse=True)
        return subsections[:15]  # Return top 15 subsections

    def refine_text(self, text: str, keywords: List[str]) -> str:
        """Refine text by highlighting key information and removing noise."""
        # Remove extra whitespace and clean up
        refined = re.sub(r'\s+', ' ', text.strip())
        
        # Extract sentences that contain keywords
        sentences = sent_tokenize(refined)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence)
            elif len(sentence) > 100 and len(relevant_sentences) < 3:
                # Include longer descriptive sentences if we don't have many keyword matches
                relevant_sentences.append(sentence)
        
        # Return refined text (max 300 words)
        result = ' '.join(relevant_sentences)
        words = result.split()
        if len(words) > 300:
            result = ' '.join(words[:300]) + "..."
            
        return result

    def calculate_subsection_relevance(self, text: str, keywords: List[str], 
                                     job_description: str) -> float:
        """Calculate relevance score for a subsection."""
        text_lower = text.lower()
        
        # Keyword density score
        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
        keyword_score = min(keyword_matches * 15, 60)
        
        # Text quality score (length, completeness)
        quality_score = min(len(text) / 200 * 20, 40)
        
        return keyword_score + quality_score

    def process_documents(self, pdf_paths: List[str], persona_description: str, 
                         job_description: str) -> Dict[str, Any]:
        """Main processing function to analyze documents."""
        start_time = time.time()
        
        self.logger.info(f"Processing {len(pdf_paths)} documents")
        
        # Extract text from all PDFs
        all_sections = []
        document_names = []
        
        for pdf_path in pdf_paths:
            doc_name = os.path.basename(pdf_path)
            document_names.append(doc_name)
            
            self.logger.info(f"Processing {doc_name}")
            pages_text = self.extract_text_from_pdf(pdf_path)
            sections = self.identify_sections(pages_text, doc_name)
            all_sections.extend(sections)
        
        # Rank sections based on relevance
        ranked_sections = self.rank_sections(all_sections, persona_description, job_description)
        
        # Extract top sections (limit to ensure performance)
        top_sections = ranked_sections[:20]
        
        # Extract subsections
        subsections = self.extract_subsections(top_sections, persona_description, job_description)
        
        # Prepare output
        processing_time = time.time() - start_time
        
        output = {
            "metadata": {
                "input_documents": document_names,
                "persona": persona_description,
                "job_to_be_done": job_description,
                "processing_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": round(processing_time, 2),
                "total_sections_analyzed": len(all_sections),
                "top_sections_selected": len(top_sections)
            },
            "extracted_sections": [
                {
                    "document": section.document_name,
                    "page_number": section.page_number,
                    "section_title": section.section_title,
                    "importance_rank": i + 1,
                    "relevance_score": round(section.importance_score, 2),
                    "content_preview": section.content[:200] + "..." if len(section.content) > 200 else section.content
                }
                for i, section in enumerate(top_sections)
            ],
            "subsection_analysis": [
                {
                    "document": subsection.document_name,
                    "page_number": subsection.page_number,
                    "refined_text": subsection.refined_text,
                    "relevance_score": round(subsection.relevance_score, 2),
                    "original_content_length": len(subsection.content)
                }
                for subsection in subsections
            ]
        }
        
        self.logger.info(f"Processing completed in {processing_time:.2f} seconds")
        return output

    def load_challenge_input(self, input_file: str) -> Dict[str, Any]:
        """Load challenge input from JSON file."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                challenge_data = json.load(f)
            
            self.logger.info(f"Loaded challenge: {challenge_data.get('challenge_info', {}).get('challenge_id', 'unknown')}")
            return challenge_data
        except Exception as e:
            self.logger.error(f"Failed to load challenge input: {e}")
            raise

    def process_challenge(self, input_file: str, pdf_directory: str) -> Dict[str, Any]:
        """Process a complete challenge with JSON input format."""
        # Load challenge data
        challenge_data = self.load_challenge_input(input_file)
        
        # Extract challenge information
        challenge_info = challenge_data.get('challenge_info', {})
        documents_info = challenge_data.get('documents', [])
        persona_info = challenge_data.get('persona', {})
        job_info = challenge_data.get('job_to_be_done', {})
        
        # Build persona description
        persona_role = persona_info.get('role', 'Professional')
        persona_description = f"{persona_role} with expertise in planning and coordination."
        
        # If there's additional persona context, include it
        if 'expertise' in persona_info:
            persona_description += f" Specializes in {persona_info['expertise']}."
        if 'focus_areas' in persona_info:
            persona_description += f" Focuses on {', '.join(persona_info['focus_areas'])}."
        
        # Extract job description
        job_description = job_info.get('task', 'Complete the assigned task.')
        
        # Build PDF paths from document information
        pdf_paths = []
        document_titles = {}
        
        for doc_info in documents_info:
            filename = doc_info.get('filename', '')
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, filename)
                if os.path.exists(pdf_path):
                    pdf_paths.append(pdf_path)
                    # Store title mapping for better output formatting
                    document_titles[filename] = doc_info.get('title', filename.replace('.pdf', ''))
                else:
                    self.logger.warning(f"PDF file not found: {pdf_path}")
        
        if not pdf_paths:
            raise ValueError("No valid PDF files found for processing")
        
        # Process documents using existing method
        result = self.process_documents(pdf_paths, persona_description, job_description)
        
        # Enhance metadata with challenge information
        result['metadata'].update({
            'challenge_id': challenge_info.get('challenge_id', 'unknown'),
            'test_case_name': challenge_info.get('test_case_name', 'unknown'),
            'challenge_description': challenge_info.get('description', ''),
            'document_titles': document_titles,
            'persona_role': persona_role,
            'original_task': job_description
        })
        
        # Update document references in output to include titles
        for section in result['extracted_sections']:
            filename = section['document']
            if filename in document_titles:
                section['document_title'] = document_titles[filename]
        
        for subsection in result['subsection_analysis']:
            filename = subsection['document']
            if filename in document_titles:
                subsection['document_title'] = document_titles[filename]
        
        return result

def main():
    """Main execution function."""
    import sys
    
    # Support both old and new input formats
    if len(sys.argv) == 2:
        # New JSON input format
        input_file = sys.argv[1]
        
        # Determine PDF directory (assume same directory as input file or 'input' subdirectory)
        input_dir = os.path.dirname(input_file) if os.path.dirname(input_file) else '.'
        pdf_directory = os.path.join(input_dir, 'pdfs') if os.path.exists(os.path.join(input_dir, 'pdfs')) else input_dir
        
        # Initialize and run the system
        system = DocumentIntelligenceSystem()
        result = system.process_challenge(input_file, pdf_directory)
        
        # Generate output filename based on challenge info
        challenge_id = result['metadata'].get('challenge_id', 'challenge')
        test_case = result['metadata'].get('test_case_name', 'output')
        output_file = f"{challenge_id}_{test_case}_output.json"
        
    elif len(sys.argv) == 3:
        # JSON input with explicit PDF directory
        input_file = sys.argv[1]
        pdf_directory = sys.argv[2]
        
        system = DocumentIntelligenceSystem()
        result = system.process_challenge(input_file, pdf_directory)
        
        challenge_id = result['metadata'].get('challenge_id', 'challenge')
        test_case = result['metadata'].get('test_case_name', 'output')
        output_file = f"{challenge_id}_{test_case}_output.json"
        
    elif len(sys.argv) == 4:
        # Legacy format: python main.py <pdf_directory> <persona_file> <job_file>
        pdf_directory = sys.argv[1]
        persona_file = sys.argv[2]
        job_file = sys.argv[3]
        
        # Read persona and job descriptions
        with open(persona_file, 'r') as f:
            persona_description = f.read().strip()
        
        with open(job_file, 'r') as f:
            job_description = f.read().strip()
        
        # Get all PDF files
        pdf_paths = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        if not pdf_paths:
            print("No PDF files found in the specified directory")
            sys.exit(1)
        
        # Initialize and run the system
        system = DocumentIntelligenceSystem()
        result = system.process_documents(pdf_paths, persona_description, job_description)
        output_file = 'challenge1b_output.json'
        
    else:
        print("Usage:")
        print("  New format: python main.py <challenge_input.json> [pdf_directory]")
        print("  Legacy format: python main.py <pdf_directory> <persona_file> <job_file>")
        sys.exit(1)
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete. Output saved to {output_file}")
    print(f"Challenge: {result['metadata'].get('challenge_id', 'N/A')}")
    print(f"Test case: {result['metadata'].get('test_case_name', 'N/A')}")
    print(f"Processed {len(result['metadata']['input_documents'])} documents in {result['metadata']['processing_time_seconds']} seconds")

if __name__ == "__main__":
    main()