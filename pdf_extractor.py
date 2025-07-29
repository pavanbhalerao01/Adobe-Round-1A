import os
import json
import pickle
import re
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class PDFOutlineExtractor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.title_patterns = [
            r'^[A-Z][A-Za-z\s\-:&]+$',  
            r'^\d+\.\s*[A-Z][A-Za-z\s]+',  
            r'^Chapter\s+\d+',  
            r'^Part\s+[IVX\d]+',  
            r'^[A-Z\s]{3,}$',  
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]*)*$',  
        ]
        
    def extract_text_features(self, pdf_path):
        """Extract features from PDF text blocks with text grouping"""
        doc = fitz.open(pdf_path)
        features = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
             
            grouped_spans = self._group_text_spans(blocks, page.rect)
            
            for group in grouped_spans:
                text = group['text'].strip()
                if len(text) < 3 or len(text) > 500:   
                    continue
                    
                
                feature = self._extract_group_features(group, text, page_num, page.rect)
                if feature:
                    features.append(feature)
        
        doc.close()
        return features
    
    def _group_text_spans(self, blocks, page_rect):
        """Group text spans that likely belong to the same text element"""
        all_spans = []
        
        
        for block in blocks:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if len(text) < 1:
                        continue
                    
                    bbox = span["bbox"]
                    all_spans.append({
                        'text': text,
                        'bbox': bbox,
                        'font_size': span["size"],
                        'font_flags': span["flags"],
                        'font': span.get("font", ""),
                        'x_center': (bbox[0] + bbox[2]) / 2,
                        'y_center': (bbox[1] + bbox[3]) / 2,
                        'line_height': bbox[3] - bbox[1]
                    })
        
         
        all_spans.sort(key=lambda x: x['y_center'])
        
         
        groups = []
        current_group = None
        
        for span in all_spans:
            if current_group is None:
                current_group = self._create_new_group(span)
            else:
                
                if self._should_group_spans(current_group, span):
                    self._add_span_to_group(current_group, span)
                else:
                     
                    groups.append(self._finalize_group(current_group))
                    current_group = self._create_new_group(span)
        
         
        if current_group:
            groups.append(self._finalize_group(current_group))
        
        return groups
    
    def _create_new_group(self, span):
        """Create a new text group starting with the given span"""
        return {
            'spans': [span],
            'text': span['text'],
            'bbox': span['bbox'],
            'font_size': span['font_size'],
            'font_flags': span['font_flags'],
            'font': span['font'],
            'avg_line_height': span['line_height']
        }
    
    def _should_group_spans(self, group, span):
        """Determine if a span should be added to the existing group"""
        last_span = group['spans'][-1]
        
         
        vertical_distance = span['y_center'] - last_span['y_center']
        avg_line_height = group['avg_line_height']
        
         
        if vertical_distance > avg_line_height * 2.5:
            return False
        
         
        x_diff = abs(span['x_center'] - last_span['x_center'])
        page_width = 600   
        
         
        font_size_diff = abs(span['font_size'] - group['font_size'])
        font_similar = (font_size_diff < 2 and 
                       span['font'] == group['font'] and
                       span['font_flags'] == group['font_flags'])
        
         
        is_continuation = self._is_text_continuation(group['text'], span['text'])
        
        same_line = vertical_distance < avg_line_height * 0.5
        multi_line_title = (font_similar and 
                          vertical_distance < avg_line_height * 2.0 and
                          (x_diff < page_width * 0.3 or is_continuation))
        
        return same_line or multi_line_title
    
    def _is_text_continuation(self, existing_text, new_text):
        """Check if new text is likely a continuation of existing text"""
        
        combined = f"{existing_text} {new_text}".strip()
        
        if (not existing_text.endswith(('.', '!', '?', ':')) and 
            (new_text[0].islower() or new_text.startswith(('and', 'or', 'of', 'in', 'on', 'at', 'to', 'for')))):
            return True
        
         
        if (existing_text.istitle() and new_text.istitle()) or (existing_text.isupper() and new_text.isupper()):
            return True
        
        title_keywords = ['introduction', 'chapter', 'part', 'section', 'analysis', 'study', 'research', 'method']
        if any(keyword in combined.lower() for keyword in title_keywords):
            return True
        
        return False
    
    def _add_span_to_group(self, group, span):
        """Add a span to an existing group"""
        group['spans'].append(span)
        group['text'] = f"{group['text']} {span['text']}".strip()
        
        old_bbox = group['bbox']
        new_bbox = span['bbox']
        group['bbox'] = [
            min(old_bbox[0], new_bbox[0]),   
            min(old_bbox[1], new_bbox[1]),   
            max(old_bbox[2], new_bbox[2]),   
            max(old_bbox[3], new_bbox[3])    
        ]
        
        total_height = sum(s['line_height'] for s in group['spans'])
        group['avg_line_height'] = total_height / len(group['spans'])
    
    def _finalize_group(self, group):
        """Finalize a text group and clean up the text"""
        
        text = group['text']
        text = re.sub(r'\s+', ' ', text)   
        text = text.strip()
        
        group['text'] = text
        return group
    
    def _extract_group_features(self, group, text, page_num, page_rect):
        """Extract features from a text group"""
        bbox = group['bbox']
        font_size = group['font_size']
        font_flags = group['font_flags']
        
        x_pos = bbox[0] / page_rect.width
        y_pos = bbox[1] / page_rect.height
        width = (bbox[2] - bbox[0]) / page_rect.width
        height = (bbox[3] - bbox[1]) / page_rect.height
        
        is_bold = bool(font_flags & 2**4)
        is_italic = bool(font_flags & 2**1)
        word_count = len(text.split())
        char_count = len(text)
        line_count = len(group['spans'])   
        
        has_number = bool(re.search(r'\d', text))
        starts_with_number = bool(re.match(r'^\d+\.?\s*', text))
        is_all_caps = text.isupper() and len(text) > 3
        is_title_case = text.istitle()
        
        is_left_aligned = x_pos < 0.1
        is_centered = 0.3 < x_pos < 0.7
        is_top_of_page = y_pos < 0.2
        
        is_multi_line = line_count > 1
        avg_words_per_line = word_count / line_count if line_count > 0 else word_count
        
        return {
            'text': text,
            'page': page_num + 1,
            'font_size': font_size,
            'is_bold': int(is_bold),
            'is_italic': int(is_italic),
            'x_pos': x_pos,
            'y_pos': y_pos,
            'width': width,
            'height': height,
            'word_count': word_count,
            'char_count': char_count,
            'line_count': line_count,
            'is_multi_line': int(is_multi_line),
            'avg_words_per_line': avg_words_per_line,
            'has_number': int(has_number),
            'starts_with_number': int(starts_with_number),
            'is_all_caps': int(is_all_caps),
            'is_title_case': int(is_title_case),
            'is_left_aligned': int(is_left_aligned),
            'is_centered': int(is_centered),
            'is_top_of_page': int(is_top_of_page)
        }
    
    def generate_training_data(self, pdf_folder, json_folder):
        """Generate training data from PDF files and their corresponding JSON labels"""
        training_data = []
        
        for pdf_file in os.listdir(pdf_folder):
            if not pdf_file.endswith('.pdf'):
                continue
                
            pdf_path = os.path.join(pdf_folder, pdf_file)
            json_file = pdf_file.replace('.pdf', '.json')
            json_path = os.path.join(json_folder, json_file)
            
            if not os.path.exists(json_path):
                continue
                
            print(f"Processing {pdf_file}...")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            
            
            features = self.extract_text_features(pdf_path)
            
            labeled_data = self._label_features(features, ground_truth)
            training_data.extend(labeled_data)
        
        return training_data
    
    def _label_features(self, features, ground_truth):
        """Label features based on ground truth with improved matching for multi-line titles"""
        labeled_data = []
        title = ground_truth.get('title', '').strip()
        outline = ground_truth.get('outline', [])
        
        heading_texts = set()
        heading_levels = {}
        
        for item in outline:
            text = item['text'].strip()
            level = item['level']
            heading_texts.add(text)
            heading_levels[text] = level
        
        for feature in features:
            text = feature['text'].strip()
            
            
            if text == title and title:
                label = 'title'
            elif text in heading_texts:
                level = heading_levels[text]
                if level == 'H1':
                    label = 'h1'
                elif level == 'H2':
                    label = 'h2'
                elif level == 'H3':
                    label = 'h3'
                else:
                    label = 'text'
            else:
                label = 'text'
                
                
                for heading_text in heading_texts:
                    if self._advanced_text_match(text, heading_text):
                        level = heading_levels[heading_text]
                        if level == 'H1':
                            label = 'h1'
                        elif level == 'H2':
                            label = 'h2'
                        elif level == 'H3':
                            label = 'h3'
                        break
                
                
                if title and self._advanced_text_match(text, title):
                    label = 'title'
            
            feature['label'] = label
            labeled_data.append(feature)
        
        return labeled_data
    
    def _advanced_text_match(self, text1, text2):
        """Advanced text matching that handles multi-line and partial matches"""
        
        if text1.strip() == text2.strip():
            return True
        
        clean1 = re.sub(r'[^\w\s]', '', text1.lower()).strip()
        clean2 = re.sub(r'[^\w\s]', '', text2.lower()).strip()
        
        
        if clean1 == clean2:
            return True
        
        
        if clean1 in clean2 or clean2 in clean1:
            return True
        
        
        words1 = set(clean1.split())
        words2 = set(clean2.split())
        
        if not words1 or not words2:
            return False
        
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        
        jaccard = len(intersection) / len(union) if union else 0
        
        
        min_words = min(len(words1), len(words2))
        if min_words <= 3:
            return jaccard > 0.7 or len(intersection) >= min_words
        
        
        return jaccard > 0.6
    
    def _text_similarity(self, text1, text2):
        """Calculate text similarity using Jaccard similarity"""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0
    
    def _fuzzy_title_match(self, text1, text2):
        """Additional fuzzy matching for titles"""
        return self._advanced_text_match(text1, text2)
    
    def train_model(self, training_data):
        """Train the classification model with updated features"""
        df = pd.DataFrame(training_data)
        
        
        feature_columns = [
            'font_size', 'is_bold', 'is_italic', 'x_pos', 'y_pos', 'width', 'height',
            'word_count', 'char_count', 'line_count', 'is_multi_line', 'avg_words_per_line',
            'has_number', 'starts_with_number', 'is_all_caps', 'is_title_case', 
            'is_left_aligned', 'is_centered', 'is_top_of_page'
        ]
        
        X = df[feature_columns].values
        y = df['label'].values
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=49)
        
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        
        y_pred = self.model.predict(X_test_scaled)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.model, self.scaler
    
    def save_model(self, model_path='models/outline_extractor.pkl'):
        """Save the trained model and scaler"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='models/outline_extractor.pkl'):
        """Load the trained model and scaler"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        
        print(f"Model loaded from {model_path}")
    
    def predict_outline(self, pdf_path):
        """Predict outline for a PDF file"""
        features = self.extract_text_features(pdf_path)
        
        if not features:
            return {"title": "", "outline": []}
        
        
        feature_columns = [
            'font_size', 'is_bold', 'is_italic', 'x_pos', 'y_pos', 'width', 'height',
            'word_count', 'char_count', 'line_count', 'is_multi_line', 'avg_words_per_line',
            'has_number', 'starts_with_number', 'is_all_caps', 'is_title_case', 
            'is_left_aligned', 'is_centered', 'is_top_of_page'
        ]
        
        X = np.array([[f[col] for col in feature_columns] for f in features])
        X_scaled = self.scaler.transform(X)
        
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
       
        title_candidates = []
        headings = []
        
        for i, (feature, pred, prob) in enumerate(zip(features, predictions, probabilities)):
            max_prob = max(prob)
            pred_idx = np.argmax(prob)
            class_label = self.model.classes_[pred_idx]
            
            
            if class_label == 'title' and max_prob > 0.4:  
                title_candidates.append({
                    'text': feature['text'],
                    'confidence': max_prob,
                    'page': feature['page'],
                    'font_size': feature['font_size'],
                    'is_bold': feature['is_bold'],
                    'y_pos': feature['y_pos'],
                    'is_multi_line': feature['is_multi_line'],
                    'line_count': feature['line_count'],
                    'feature_idx': i
                })
            elif class_label in ['h1', 'h2', 'h3'] and max_prob > 0.5:
                headings.append({
                    'level': class_label.upper(),
                    'text': feature['text'],
                    'page': feature['page'],
                    'confidence': max_prob,
                    'y_pos': feature['y_pos']
                })
        
        
        title = self._select_best_title(title_candidates, features)
        
        
        if not title:
            title = self._extract_title_fallback(features)
        
        
        title, headings = self._apply_rules(title, headings, features)
        
        
        headings.sort(key=lambda x: (x['page'], x.get('y_pos', 0)))
        
        
        outline = [{'level': h['level'], 'text': h['text'], 'page': h['page']} for h in headings]
        
        return {
            "title": title,
            "outline": outline
        }
    
    def _select_best_title(self, title_candidates, features):
        """Select the best title from candidates with multi-line support"""
        if not title_candidates:
            return ""
        
        
        def title_score(candidate):
            score = candidate['confidence']
            
           
            if candidate['page'] == 1:
                score += 0.4
            elif candidate['page'] <= 3:
                score += 0.2
            
            
            if candidate['y_pos'] < 0.2:
                score += 0.3
            elif candidate['y_pos'] < 0.4:
                score += 0.1
            
            
            if candidate['is_bold']:
                score += 0.2
            
           
            if candidate['is_multi_line']:
                score += 0.1
            
            
            max_font = max(f['font_size'] for f in features)
            if max_font > 0:
                font_ratio = candidate['font_size'] / max_font
                if font_ratio > 0.8:
                    score += 0.3
                elif font_ratio > 0.6:
                    score += 0.1
            
            
            text_len = len(candidate['text'])
            word_count = len(candidate['text'].split())
            
            
            if 10 <= text_len <= 200 and 2 <= word_count <= 25:
                score += 0.1
            elif text_len < 5 or text_len > 300:
                score -= 0.3
            
            return score
        
        
        title_candidates.sort(key=title_score, reverse=True)
        return title_candidates[0]['text']
    
    def _extract_title_fallback(self, features):
        """Fallback title extraction using heuristics with multi-line support"""
        candidates = []
        
        for feature in features:
            text = feature['text'].strip()
            
            
            if len(text) < 5 or len(text) > 300:
                continue
            
           
            if feature['page'] > 3:
                continue
            
            score = 0
            
            
            if feature['page'] == 1:
                score += 4
            elif feature['page'] <= 2:
                score += 2
            
            if feature['y_pos'] < 0.2:  
                score += 3
            elif feature['y_pos'] < 0.4:
                score += 1
            
            if feature['is_centered']:
                score += 2
            
            
            if feature['is_bold']:
                score += 3
            if feature['font_size'] > 14:
                score += 2
            if feature['is_title_case'] or feature['is_all_caps']:
                score += 2
            
            
            if feature.get('is_multi_line', False):
                score += 1
            
           
            if any(re.match(pattern, text) for pattern in self.title_patterns):
                score += 2
            
            
            if re.search(r'^\d+$', text):
                score -= 5
            if re.search(r'^page\s+\d+', text, re.IGNORECASE):
                score -= 5
            if len(text.split()) > 20:
                score -= 2
            
            
            title_words = ['introduction', 'analysis', 'study', 'research', 'report', 'guide', 'manual']
            if any(word in text.lower() for word in title_words):
                score += 1
            
            if score > 0:
                candidates.append((score, text, feature))
        
       
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        
        return ""
    
    def _apply_rules(self, title, headings, features):
        """Apply rule-based refinements"""
        
        if title:
            
            title = re.sub(r'^\s*[^\w]*\s*', '', title)
            title = re.sub(r'\s*[^\w]*\s*$', '', title)
            title = re.sub(r'\s+', ' ', title).strip()
        
        
        refined_headings = []
        for heading in headings:
            text = heading['text']
            
           
            if len(text) < 3 or len(text) > 300:
                continue
            
            
            cleaned_text = re.sub(r'^\s*[^\w]*\s*', '', text)
            cleaned_text = re.sub(r'\s*[^\w]*\s*$', '', cleaned_text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            
            if len(cleaned_text) < 3:
                continue
            
            
            if (len(cleaned_text.split()) > 15 and 
                not heading['text'][0].isupper() and
                heading['confidence'] < 0.7):
                continue
            
            heading['text'] = cleaned_text
            refined_headings.append(heading)
        
        return title, refined_headings


def train_extractor():
    """Train the PDF outline extractor"""
    extractor = PDFOutlineExtractor()
    
    
    print("Generating training data...")
    training_data = extractor.generate_training_data('training_pdfs/', 'training_labels/')
    
    if not training_data:
        print("No training data found! Please ensure PDF files and corresponding JSON labels are in the correct folders.")
        return
    
    print(f"Generated {len(training_data)} training samples")
    
    
    print("Training model...")
    extractor.train_model(training_data)
    
    
    extractor.save_model()
    print("Training complete!")


def process_pdf(pdf_path, model_path='models/outline_extractor.pkl'):
    """Process a single PDF file"""
    extractor = PDFOutlineExtractor()
    extractor.load_model(model_path)
    
    result = extractor.predict_outline(pdf_path)
    return result

def process_directory(input_dir, output_dir, model_path='models/outline_extractor.pkl'):
    """Process all PDF files in a directory"""
    extractor = PDFOutlineExtractor()
    extractor.load_model(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.pdf', '.json'))
            
            print(f"Processing {filename}...")
            result = extractor.predict_outline(pdf_path)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Output saved to {output_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train_extractor()
        elif sys.argv[1] == "process" and len(sys.argv) > 2:
            pdf_path = sys.argv[2]
            result = process_pdf(pdf_path)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif sys.argv[1] == "batch" and len(sys.argv) > 3:
            input_dir = sys.argv[2]
            output_dir = sys.argv[3]
            process_directory(input_dir, output_dir)
    else:
        print("Usage:")
        print("  python pdf_extractor.py train")
        print("  python pdf_extractor.py process <pdf_path>")
        print("  python pdf_extractor.py batch ./training_pdfs ./output")
