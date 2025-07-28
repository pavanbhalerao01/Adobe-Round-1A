"""Text preprocessing utilities"""

import re
import unicodedata

def clean_text(text):
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    
    return text

def normalize_heading_text(text):
    """Normalize heading text for comparison"""
    # Remove common prefixes/suffixes
    text = re.sub(r'^(Chapter|Section|Part)\s+\d+:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+\.?\s*', '', text)  # Remove leading numbers
    text = re.sub(r'\s*\.\.\.*\s*\d*$', '', text)  # Remove trailing dots and page numbers
    
    return clean_text(text)

def extract_page_numbers(text):
    """Extract page numbers from text"""
    # Look for patterns like "... 15", "Page 15", etc.
    patterns = [
        r'.*?(\d+)$',  # Number at end
        r'.*?[Pp]age\s+(\d+)',  # "Page X"
        r'.*?[Pp]\.?\s*(\d+)',  # "P. X" or "p X"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    
    return None


# Train the model
# python pdf_extractor.py train


# Process a single PDF
# python pdf_extractor.py process document.pdf


# Batch process directory
# python pdf_extractor.py batch ./training_pdfs ./output