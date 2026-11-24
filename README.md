# MailGuard AI – Email Spam Detection System

## Project Overview
MailGuard AI is a Machine Learning-based application designed to classify text messages as Spam or Not Spam.

The application provides a simple and user-friendly interface using Streamlit.

## Problem Statement
Unwanted spam messages can create inconvenience and may contain misleading information. This project uses Machine Learning to identify potentially unwanted messages.

## Technologies Used
- Python
- Machine Learning
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Streamlit
- Pandas
- GitHub

## Features
- Spam and Not Spam classification
- Prediction confidence display
- Prediction history
- CSV download
- Clear history option
- Interactive dashboard
- Web-based application

## Machine Learning Workflow
1. Load the SMS Spam Collection dataset.
2. Preprocess the text messages.
3. Convert text into numerical features using TF-IDF.
4. Train a Multinomial Naive Bayes model.
5. Save the trained model and vectorizer.
6. Deploy the application using Streamlit Community Cloud.

## Project Files
- `app.py` – Streamlit application
- `train_model.py` – Model training
- `data_exploration.py` – Dataset exploration
- `spam_model.pkl` – Trained ML model
- `vectorizer.pkl` – TF-IDF vectorizer
- `requirements.txt` – Project dependencies

## Future Enhancements
- Improve model performance with additional datasets.
- Support email message classification.
- Add more advanced text preprocessing.
- Improve the user interface.

## Author
S. Swetha

AI & Data Science Student

## Disclaimer
This application is an educational project. Predictions are model estimates and may not always be correct.
## Live Demo

[🚀 Open MailGuard AI](https://emailspamdetection-4hch8eu42fk6epkbtvtgxk.streamlit.app/)

