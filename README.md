# Bilingual-Text-Analysis
A machine learning–based system to detect and classify toxic comments in English, Marathi, and code-mixed (English–Marathi) text, designed for real-world social media platforms like YouTube.

📖 Overview

This project aims to classify YouTube comments into different toxicity levels using transformer-based NLP models. It supports multilingual and code-mixed text, focusing on Indian languages—especially Marathi.

❓ Problem Statement

Most existing toxicity detection systems are optimized for English-only content and perform poorly on Indian languages or code-mixed text. This project addresses that gap by building a multilingual toxicity classification pipeline.

✨ Features

Multilingual support (English + Marathi)

Handles code-mixed comments

Multi-class toxicity classification

Transformer-based deep learning models

Supports both local inference and API-based moderation

Scalable pipeline for large datasets

🚦 Toxicity Classes
Label	Description
0	Non-Toxic
1	Partially Toxic
2	Toxic

🛠️ Tech Stack
Python
PyTorch
Hugging Face Transformers
Pandas & NumPy
Scikit-learn

REST APIs (Perspective / Moderation APIs)

🤖 Models Used

XLM-RoBERTa – Multilingual transformer (recommended)

IndicBERT (AI4Bharat) – Optimized for Indian languages

Unbiased Toxic RoBERTa – Toxicity-specific model

(Optional) API-based models for fast moderation

📂 Dataset

Source: YouTube comments

Languages: English, Marathi, Code-Mixed

Format:

comment_id,user,text,label


⚠️ Dataset is anonymized and used for academic/research purposes only.

🧱 Project Architecture
``data/
 ├── raw_comments.csv
 ├── train_data.csv
 └── processed_data.csv

models/
 ├── fine_tuned_model/
 └── tokenizer/

scripts/
 ├── preprocess.py
 ├── train.py
 ├── inference.py
 └── api_integration.py

notebooks/
 └── experiments.ipynb``

⚙️ Installation
`git clone https://github.com/your-username/multilingual-toxic-comment-classifier.git
cd multilingual-toxic-comment-classifier`

`pip install -r requirements.txt`

▶️ Usage
Run Inference on a CSV
python scripts/inference.py --input data/test_comments.csv

🏋️ Training the Model
``python scripts/train.py \
  --model xlm-roberta-base \
  --epochs 3 \
  --batch_size 8``

🔍 Inference

Example:

``comment = "तु खूप वाईट आहेस, stop spreading hate"
prediction = classify_comment(comment)
print(prediction)``


Output:

Toxic

🌐 API Integration

Supports external moderation APIs:

Perspective API

OpenAI Moderation API

Hugging Face Inference API

Useful for quick classification without training.

📊 Results & Evaluation

Accuracy

Precision / Recall

F1-score

Confusion Matrix

Detailed evaluation results available in the notebooks/ directory.

⚠️ Challenges

Code-mixed language ambiguity

Romanized Marathi text

Class imbalance in toxicity labels

Limited labeled data for Indian languages

🚀 Future Improvements

Romanized Marathi support

Explainable AI (attention visualization)

REST API deployment

Support for more Indian languages

Active learning for label improvement

📚 References

Hugging Face Transformers

AI4Bharat (IndicBERT)

Google Perspective API

Jigsaw Toxic Comment Dataset
