<b><h1>Twitter Sentiment Analysis using BERT</h1></b>

<b>Project Overview</b>

This project is aimed at performing sentiment analysis on a Twitter dataset from Kaggle. The main objective is to classify tweets into positive, negative, or neutral sentiments. The project uses BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art model for NLP tasks, to achieve high accuracy.

The project also includes the preprocessing of raw text data, such as removing special characters, tokenization, lemmatization, removing stopwords, and converting text to lowercase. The model is deployed using Streamlit, enabling a user-friendly web interface for real-time tweet sentiment analysis.
Project Features

<b>Data Preprocessing:</b><br>
&nbsp;&nbsp;&nbsp;&nbsp;Removing Special Characters: Cleaning the dataset by removing unnecessary characters (e.g., hashtags, mentions, punctuation).<br>
&nbsp;&nbsp;&nbsp;&nbsp;Tokenization: Splitting text into tokens (individual words or phrases).<br>
&nbsp;&nbsp;&nbsp;&nbsp;Removing Stopwords: Eliminating common words that don’t contribute to sentiment (e.g., "the", "is", "in").<br>
&nbsp;&nbsp;&nbsp;&nbsp;Lemmatization: Converting words to their base or root form.<br>
&nbsp;&nbsp;&nbsp;&nbsp;Lowercasing: Converting all text to lowercase for uniformity.



<b>Sentiment Classification using BERT:<b><br>
&nbsp;&nbsp;&nbsp;&nbsp;Fine-tuning the BERT model to predict tweet sentiment (positive, negative, or neutral).<br>

**Deployment:**  
&nbsp;&nbsp;&nbsp;&nbsp;A Streamlit web application is created to provide a simple interface for users to input tweets and receive sentiment predictions in real-time.

<b>Dataset</b>

The dataset used in this project is obtained from Kaggle. It consists of labeled tweets for sentiment analysis, with three sentiment categories: positive, negative, and neutral.

    Kaggle Dataset Link: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

<b>Dataset Structure:</b>

    Text: The tweet itself.
    Sentiment: The label indicating the sentiment (positive, negative, neutral).

<b>Technology Stack</b>

    Programming Language: Python
    Libraries:
        Pandas: For data manipulation and preprocessing.
        NLTK: For text processing tasks such as stopword, removal, tokenization and lemmatization.
        Transformers (Hugging Face): For implementing the BERT model.
        Streamlit: For deploying the application and creating an interactive web interface.
        Matplotlib, Seaborn: For data visualization.

Setup and Installation

To run this project locally, follow these steps:
    1. Clone the repository:

    bash
   
    git clone https://github.com/your-username/twitter-sentiment-analysis-bert.git
   
    cd twitter-sentiment-analysis-bert

2. Download the Dataset:

Download the dataset from Kaggle and place it in the project's data folder. You can use the Kaggle API to download it directly.

    bash

    kaggle datasets download -d <dataset-name>

3. Preprocessing the Dataset:

Run the preprocessing script to clean and prepare the data for training:

    bash

    python preprocess.py

This script performs the following tasks:

    Remove special characters, URLs, and mentions.
    Tokenize text.
    Remove stopwords.
    Apply lemmatization.
    Convert text to lowercase.

4. Train the BERT Model:

Once the data is preprocessed, train the BERT model on the dataset:

    bash

    python train.py

This script will fine-tune the pre-trained BERT model on the Twitter dataset.
5. Run the Streamlit Web Application:

After the model is trained, you can run the Streamlit app for real-time sentiment analysis:

    bash

    streamlit run app.py
 
This will launch a web interface where you can input tweets and receive sentiment predictions.

<b>Results</b>

The BERT model achieves 0.76% accuracy on the test set, outperforming traditional machine learning approaches due to its ability to capture the contextual meaning of words in tweets. Detailed evaluation metrics such as precision, recall, and F1-score are available in the training logs.
 
<b>Conclusion</b>

This project demonstrates the application of state-of-the-art NLP models (BERT) for sentiment analysis on social media data, leveraging both text preprocessing and deep learning. The deployment using Streamlit provides a simple interface for real-time sentiment predictions, making it a useful tool for businesses or researchers to gauge public sentiment.

