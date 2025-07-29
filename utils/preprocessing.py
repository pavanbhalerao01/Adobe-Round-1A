import re
import unicodedata

def clean_text(text):
    """Clean and normalize text"""
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    
    text = unicodedata.normalize('NFKD', text)
    
    
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    
    return text

def normalize_heading_text(text):
    """Normalize heading text for comparison"""
    
    text = re.sub(r'^(Chapter|Section|Part)\s+\d+:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+\.?\s*', '', text)  
    text = re.sub(r'\s*\.\.\.*\s*\d*$', '', text)  
    
    return clean_text(text)

def extract_page_numbers(text):
    """Extract page numbers from text"""
    
    patterns = [
        r'.*?(\d+)$',  
        r'.*?[Pp]age\s+(\d+)',  
        r'.*?[Pp]\.?\s*(\d+)',  
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    
    return None

# Process a single PDF
# python pdf_extractor.py process document.pdf


# Batch process directory
# python pdf_extractor.py batch ./input ./output
