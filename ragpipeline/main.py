import json
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging
import argparse
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
        
        # Extract noun phrases and important terms
        important_patterns = [
            r'\b(research|analysis|study|review|methodology|data|performance|trend|strategy)\w*\b',
            r'\b(financial|technical|academic|business|scientific|clinical)\w*\b',
            r'\b(PhD|analyst|student|researcher|manager|specialist)\w*\b'
        ]
        
        for pattern in important_patterns:
            matches = re.findall(pattern, combined_text.lower())
            keywords.extend(matches)
        
        # Remove duplicates and return top keywords
        return list(set(keywords))[:20]

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

def main():
    import argparse
    import sys
    import os
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', required=True, help='Path to the input JSON file')
    parser.add_argument('--pdf_dir', required=True, help='Directory containing the PDF files')
    parser.add_argument('--output_json', required=True, help='Path to save the output JSON')

    args = parser.parse_args()

    input_json_path = args.input_json
    pdf_directory = args.pdf_dir
    output_json_path = args.output_json

    print(f"📥 Loading input JSON from: {input_json_path}")
    try:
        with open(input_json_path, 'r') as f:
            input_data = json.load(f)
            persona_description = input_data.get("persona", {}).get("role", "")
            job_description = input_data.get("job_to_be_done", {}).get("task", "")
        print("✅ Successfully loaded persona and job description")
    except Exception as e:
        print(f"❌ Failed to read input JSON file: {e}")
        sys.exit(1)

    print(f"📂 Looking for PDFs in: {pdf_directory}")
    try:
        pdf_paths = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    except Exception as e:
        print(f"❌ Could not read PDF directory: {e}")
        sys.exit(1)

    if not pdf_paths:
        print("⚠️ No PDF files found in the specified directory")
        sys.exit(1)

    print(f"📄 Found {len(pdf_paths)} PDF files")
    print(f"🧠 Initializing DocumentIntelligenceSystem...")

    system = DocumentIntelligenceSystem()

    print(f"🚀 Processing documents...")
    result = system.process_documents(pdf_paths, persona_description, job_description)

    print(f"💾 Saving results to: {output_json_path}")
    try:
        with open(output_json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print("✅ Output saved successfully")
    except Exception as e:
        print(f"❌ Failed to write output JSON: {e}")
        sys.exit(1)

    print(f"🎉 Processing complete!")
    print(f"📊 Processed {len(pdf_paths)} documents in {result['metadata']['processing_time_seconds']} seconds")
