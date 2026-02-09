# ChaosGuard AI

ChaosGuard AI is a deep learning–based system for **multi-label toxicity detection** on **Bangla and Banglish (code-mixed)** text using **XLM-RoBERTa**.  
The system integrates **uncertainty estimation** and **explainable AI (XAI)** to provide reliable and transparent predictions.

---

## Features

- Multi-label toxicity classification (10 categories)
- Supports Bangla and Banglish (code-mixed text)
- Uncertainty estimation using MC Dropout
- Predictive entropy for confidence analysis
- Explainability using Integrated Gradients (Captum)
- Flask-based REST API with web interface

---

## Toxicity Categories

Vulgar-based, Religious-Hostility, Troll-based, Insult-based, Loathe-based, Threat-based, Race-based, Humiliation-based, Political-Chaos, Non-Toxic

---

## Requirements

- Python 3.10 or 3.11 (recommended)
- Python 3.12 supported with slow tokenizer
- CPU or GPU

---

## Installation and Run

Create a virtual environment:
python -m venv venv

Activate the virtual environment:
Windows: venv\Scripts\activate  
Linux / macOS: source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Run the application:
python main.py

Open in browser:
http://127.0.0.1:5000

---

## Environment Variables

MODEL_DIR (optional): Path to the trained XLM-R model  
Default: ./trained_xlmr_model

PORT: Automatically handled during deployment

---

## API Endpoints

POST /api/analyze  
Returns probability, uncertainty, entropy, and toxicity decision for each label.

POST /api/explain  
Returns Integrated Gradients–based token-level explanations for a selected label.

---

## Deployment

GitHub does not run Flask applications directly.  
Deploy this repository using services such as Render, Railway, or Hugging Face Spaces.

Recommended configuration:
Python version: 3.11  
Start command: python main.py  
MODEL_DIR: ./trained_xlmr_model

---

## Common Issues

If you encounter a SentencePiece error, install it using:
pip install sentencepiece

Ensure templates/index.html exists for the web interface.

Google Drive paths (e.g., M:\My Drive\...) may work locally but should not be used for deployment.

---

## Research Use

This project is suitable for:
- IEEE conference papers
- NLP research on code-mixed and low-resource languages
- Toxicity detection with uncertainty-aware explainability

---


