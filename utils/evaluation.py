"""Model evaluation utilities"""

import json
from typing import Dict, List, Tuple

def calculate_accuracy(predicted: Dict, ground_truth: Dict) -> Dict[str, float]:
    """Calculate accuracy metrics for title and outline extraction"""
    
    # Title accuracy
    pred_title = predicted.get('title', '').strip().lower()
    true_title = ground_truth.get('title', '').strip().lower()
    title_accuracy = 1.0 if pred_title == true_title else 0.0
    
    # Outline accuracy
    pred_outline = predicted.get('outline', [])
    true_outline = ground_truth.get('outline', [])
    
    # Create sets for comparison
    pred_headings = set((h['level'], h['text'].strip().lower(), h['page']) for h in pred_outline)
    true_headings = set((h['level'], h['text'].strip().lower(), h['page']) for h in true_outline)
    
    if not true_headings:
        outline_accuracy = 1.0 if not pred_headings else 0.0
    else:
        correct = len(pred_headings.intersection(true_headings))
        total = len(true_headings)
        outline_accuracy = correct / total
    
    # Overall accuracy
    overall_accuracy = (title_accuracy + outline_accuracy) / 2
    
    return {
        'title_accuracy': title_accuracy,
        'outline_accuracy': outline_accuracy,
        'overall_accuracy': overall_accuracy,
        'predicted_headings': len(pred_headings),
        'true_headings': len(true_headings),
        'correct_headings': len(pred_headings.intersection(true_headings)) if true_headings else 0
    }

def evaluate_model(extractor, test_files: List[Tuple[str, str]]) -> Dict[str, float]:
    """Evaluate model on test files"""
    total_metrics = {
        'title_accuracy': 0.0,
        'outline_accuracy': 0.0,
        'overall_accuracy': 0.0
    }
    
    results = []
    
    for pdf_path, json_path in test_files:
        # Predict
        predicted = extractor.predict_outline(pdf_path)
        
        # Load ground truth
        with open(json_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        # Calculate metrics
        metrics = calculate_accuracy(predicted, ground_truth)
        results.append(metrics)
        
        # Accumulate totals
        for key in total_metrics:
            total_metrics[key] += metrics[key]
    
    # Average metrics
    num_files = len(test_files)
    for key in total_metrics:
        total_metrics[key] /= num_files
    
    return total_metrics, results