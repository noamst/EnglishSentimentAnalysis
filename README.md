# EnglishSentimentAnalysis
This project aims to perform sentiment analysis on English text data using an LSTM (Long Short-Term Memory) neural network architecture. The key goals of the project were to train a deep learning model to classify sentiments into categories and build an interactive web app for user interaction using Streamlit.

## Features

- Sentiment analysis for English text.
- Model built using LSTM for sequential data processing.
- Dataset used from Kaggle ([Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)).
- Streamlit-powered web application for sentiment prediction.

## Dataset

The dataset used for this project was sourced from Kaggle. It contains labeled text data for sentiment classification. The dataset includes:

- Text: The English sentences to analyze.
- Sentiment Labels: Categories such as `positive`, `neutral`, and `negative` for each text entry.

## Architecture

### Model: LSTM
- **Embedding Layer**: To convert the input text into dense vector representations.
- **LSTM Layer**: Captures the sequential patterns in the text.
- **Dense Output Layer**: Produces probabilities for the sentiment categories using softmax activation.

### Preprocessing Steps
1. Tokenization of text data using TensorFlow's Tokenizer.
2. Conversion of tokens to padded sequences for consistent input length.
3. One-hot encoding of sentiment labels for training the model.

## Application

The `app.py` file is a Streamlit application that provides an interactive interface for users to input English text and obtain sentiment predictions.

### Features
- Input text field for English sentences.
- Real-time sentiment prediction with probability scores.

### How to Run
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Access the application in your web browser at `http://localhost:8501`.

## Project Limitations

Although the project setup is complete, the final implementation of the sentiment analysis model and its training process was not completed. Specifically:

- The model achieves accuracy of 84%.
- The accuracy of predictions and fine-tuning of hyperparameters remains incomplete.

## Future Work

- Perform hyperparameter tuning to optimize the model.
- Expand the Streamlit app to support advanced visualizations and additional features.

## Directory Structure

```
.
├── app.py              # Streamlit app for sentiment analysis
├── requirements.txt    # Python dependencies
├── word_index.json     # Saved word index for text tokenization
├── model               # Folder for storing the trained model
├── README.md           # Project documentation
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Streamlit
- NumPy
- Pandas
- Scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Acknowledgments

- **Kaggle Dataset**: [Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset).
- TensorFlow and Streamlit for their robust tools for machine learning and web app development.
