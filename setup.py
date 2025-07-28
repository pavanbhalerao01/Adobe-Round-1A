from setuptools import setup, find_packages

setup(
    name="pdf-outline-extractor",
    version="1.0.0",
    description="ML-based PDF outline and title extraction",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "PyMuPDF>=1.23.5",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "joblib>=1.3.2"
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'pdf-extract=pdf_extractor:main',
        ],
    }
)