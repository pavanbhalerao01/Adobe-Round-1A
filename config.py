"""Configuration settings for PDF Outline Extractor"""

# Model parameters
MODEL_CONFIG = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

# Feature extraction parameters
FEATURE_CONFIG = {
    'min_text_length': 3,
    'max_text_length': 200,
    'title_max_length': 100,
    'heading_min_confidence': 0.6,
    'title_min_confidence': 0.7
}

# File paths
PATHS = {
    'model_dir': 'models/',
    'training_pdfs': 'training_pdfs/',
    'training_labels': 'training_labels/',
    'output_dir': 'output/',
    'logs_dir': 'logs/'
}

# Processing parameters
PROCESSING_CONFIG = {
    'max_pages': 50,
    'timeout_seconds': 10,
    'max_memory_mb': 512
}