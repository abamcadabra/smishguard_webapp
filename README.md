# SmishGuard Web Application

SmishGuard is an AI-powered phishing and smishing detection web application designed to analyze suspicious URLs and SMS/text messages using Natural Language Processing (NLP) and Deep Learning techniques.

The project combines multiple machine learning models to classify potentially malicious content and provide real-time prediction results through a web interface.

---

# Features

* URL phishing detection
* SMS / text smishing detection
* Deep learning inference with BiLSTM models
* Transformer-based text classification
* Fusion-based prediction system
* Web interface for real-time analysis
* Prediction logging and monitoring
* FastAPI backend

---

# Project Structure

```text
smishguard_webapp/
├── app/
│   ├── inference/
│   │   ├── text_inference.py
│   │   └── url_inference.py
│   ├── processing/
│   │   ├── text_preprocess.py
│   │   └── url_preprocess.py
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   └── index.html
│   ├── fusion.py
│   ├── logging_store.py
│   ├── main.py
│   ├── monitoring.py
│   ├── schemas.py
│   └── utils.py
│
├── models/
│   ├── text_model/
│   └── url_model/
│
├── prediction_logs.jsonl
├── requirements.txt
└── README.md
```

---

# Technologies Used

## Backend

* Python
* FastAPI
* TensorFlow / Keras
* Transformers
* Scikit-learn

## NLP & Deep Learning

* BiLSTM
* FastText Embedding
* BERT-based tokenizer
* URL sequence processing
* Text preprocessing pipeline

## Frontend

* HTML
* CSS
* Jinja2 Templates

---

# Model Architecture

## URL Detection Model

The URL phishing detection module treats URLs as sequential textual data.

Pipeline:

1. URL preprocessing
2. Tokenization / sequence encoding
3. FastText embedding representation
4. BiLSTM classification
5. Threshold-based prediction

Main model:

* `url_bilstm_fasttext.keras`

---

## Text Smishing Detection Model

The SMS/text detection module analyzes suspicious messages using transformer-based tokenization and deep learning classification.

Pipeline:

1. Text preprocessing
2. Transformer tokenizer
3. Embedding generation
4. BiLSTM classification
5. Prediction output

Main model:

* `bert_bilstm.keras`

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/abamcadabra/smishguard_webapp.git
cd smishguard_webapp
```

---

## 2. Create Virtual Environment

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Application URL:

```text
http://127.0.0.1:8000
```

---

# API Endpoints

## Home Page

```http
GET /
```

Displays the web interface.

---

## Prediction Endpoint

```http
POST /predict
```

Example request body:

```json
{
  "message": "Your account has been suspended. Click here now.",
  "url": "http://suspicious-example.com"
}
```

---

# Example Workflow

1. User enters a suspicious message or URL.
2. Input is preprocessed.
3. Appropriate AI model performs inference.
4. Prediction scores are generated.
5. Results are displayed on the web interface.
6. Logs are optionally stored for monitoring.

---

# Research Context

This project is related to phishing and smishing detection research using NLP-based approaches.

Key concepts:

* Natural Language Processing (NLP)
* Word Embedding
* FastText
* Transformer Tokenization
* BiLSTM Networks
* URL Sequence Classification
* Cybersecurity AI Applications

---

# Future Improvements

Potential future enhancements:

* Real-time API deployment
* Cloud deployment
* Database integration
* Model ensemble improvements
* Explainable AI visualization
* Multi-language support
* Browser extension integration
* Real-time monitoring dashboard

---

# Notes

* Large model files may require Git LFS.
* Python version 3.10+ is recommended.
* GPU acceleration is optional but recommended for training.

---

# Author

Atima Yaklinhom and Thichachol Ruenghiran
Bachelor of Computer Science
Thammasat University

---

# License

This project is for educational and research purposes.
