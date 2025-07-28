"""Output refinement and postprocessing"""

import re
from typing import List, Dict, Any

def refine_title(title: str, features: List[Dict]) -> str:
    """Refine extracted title using heuristics"""
    if not title:
        return ""
    
    # Clean common title artifacts
    title = re.sub(r'^[^A-Za-z]*', '', title)  # Remove leading non-letters
    title = re.sub(r'[^A-Za-z\s]*$', '', title)  # Remove trailing non-letters
    title = title.strip()
    
    # If title is too short, try to find a better one
    if len(title) < 5:
        for feature in features[:20]:  # Check first 20 features
            text = feature['text']
            if (len(text) > 10 and 
                len(text) < 100 and 
                feature['page'] == 1 and
                feature['is_top_of_page']):
                return text
    
    return title

def refine_headings(headings: List[Dict], features: List[Dict]) -> List[Dict]:
    """Refine extracted headings"""
    refined = []
    
    for heading in headings:
        text = heading['text']
        
        # Skip invalid headings
        if len(text) < 3 or len(text) > 200:
            continue
        
        # Clean heading text
        cleaned_text = re.sub(r'^\s*[^\w]*\s*', '', text)  # Remove leading symbols
        cleaned_text = re.sub(r'\s*[^\w]*\s*$', '', cleaned_text)  # Remove trailing symbols
        
        if len(cleaned_text) < 3:
            continue
        
        heading['text'] = cleaned_text
        refined.append(heading)
    
    return refined

def validate_outline_structure(outline: List[Dict]) -> List[Dict]:
    """Validate and fix outline structure"""
    if not outline:
        return outline
    
    # Sort by page and position
    outline.sort(key=lambda x: (x['page'], x.get('y_pos', 0)))
    
    # Remove duplicates
    seen = set()
    unique_outline = []
    for item in outline:
        key = (item['text'].lower().strip(), item['page'])
        if key not in seen:
            seen.add(key)
            unique_outline.append(item)
    
    return unique_outline