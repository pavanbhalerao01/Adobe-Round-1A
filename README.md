# PDF Information Extractor

This project is a Python-based tool for extracting and processing information from PDF documents. It uses machine learning models to analyze and extract structured data from PDF files.

## Project Structure

- `input/` - Directory containing input PDF files
- `output/` - Directory containing extracted JSON outputs
- `test/` - Directory containing test PDF files
- `training_labels/` - Directory containing training label JSON files
- `utils/` - Utility functions and helper modules
  - `preprocessing.py` - Data preprocessing utilities
  - `postprocessing.py` - Output processing utilities
  - `evaluation.py` - Evaluation metrics and tools
  - `models/` - Trained model files

## Setup

1. Create a virtual environment:
```bash
python -m venv env
```

2. Activate the virtual environment:
- Windows:
```bash
.\env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main script for PDF extraction is `pdf_extractor.py`. You can process PDF files by placing them in the `input/` directory.

## Dependencies

The project uses several Python libraries including:
- PyMuPDF (fitz)
- NumPy
- Pandas
- SciPy
- Joblib

## Docker Support

A Dockerfile is included for containerized deployment of the application.

## License

[Add your license information here]
