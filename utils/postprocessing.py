import re
from typing import List, Dict, Any

def refine_title(title: str, features: List[Dict]) -> str:
    """Refine extracted title using heuristics"""
    if not title:
        return ""
    
   
    title = re.sub(r'^[^A-Za-z]*', '', title)  
    title = re.sub(r'[^A-Za-z\s]*$', '', title)  
    title = title.strip()
    
    
    if len(title) < 5:
        for feature in features[:20]:  
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
        
        
        if len(text) < 3 or len(text) > 200:
            continue
        
        
        cleaned_text = re.sub(r'^\s*[^\w]*\s*', '', text)  
        cleaned_text = re.sub(r'\s*[^\w]*\s*$', '', cleaned_text)  
        
        if len(cleaned_text) < 3:
            continue
        
        heading['text'] = cleaned_text
        refined.append(heading)
    
    return refined

def validate_outline_structure(outline: List[Dict]) -> List[Dict]:
    """Validate and fix outline structure"""
    if not outline:
        return outline
    
    
    outline.sort(key=lambda x: (x['page'], x.get('y_pos', 0)))
    
    
    seen = set()
    unique_outline = []
    for item in outline:
        key = (item['text'].lower().strip(), item['page'])
        if key not in seen:
            seen.add(key)
            unique_outline.append(item)
    
    return unique_outline
