import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.cluster import KMeans

class AttributeExtractor:
    def __init__(self, sentiment_model_name='distilbert-base-uncased', feature_model_name='BAAI/bge-small-en-v1.5'):
        self.feature_tokenizer = AutoTokenizer.from_pretrained(feature_model_name)
        self.feature_model = AutoModel.from_pretrained(feature_model_name)

        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

    def extract_common_features(self, text_series, n_dimensions=50):
        '''
        Extracts common features from a list of product descriptions.
        Returns a list of n common features.
        '''
        # Prepare data for feature extraction
        text_clean = text_series.fillna("").str.strip()
        text_clean = ["Represent this sentence for searching relevant passages: " + desc for desc in text_clean]
        
        # Tokenize texts
        encoded_inputs = self.feature_tokenizer(
            text_clean,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # Create embeddings for each product description
        with torch.no_grad():
            feature_outputs = self.feature_model(**encoded_inputs)
            embeddings = feature_outputs.last_hidden_state[:, 0].numpy()

        # Reduce dimensionality of embeddings
        reducer = umap.UMAP(
            n_components=n_dimensions,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42
        )

        reduced_embeddings = reducer.fit_transform(embeddings)

        # Reduce to 2-D for visualization
        reducer_2d = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42
        )

        reduced_embeddings_2d = reducer_2d.fit_transform(embeddings)

        # Cluster embeddings to find common features
        kmeans = KMeans(n_clusters=12, random_state=42)

        # Prepare output
        return pd.DataFrame({
            'text': text_series,
            'cluster': kmeans.fit_predict(reduced_embeddings),
        }, index=text_series.index), reduced_embeddings_2d, reduced_embeddings

    def extract_sentiment(self, text_series):
        '''
        Extracts sentiment from a list of product descriptions.
        Returns a list of sentiment labels.
        '''
        # Preprocess text
        text_clean = text_series.fillna("").str.strip()
        
        # Tokenize text
        encoded = self.sentiment_tokenizer(
            list(text_clean),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )

        # Predict sentiment
        with torch.no_grad():
            output = self.sentiment_model(**encoded)

        # Get predictions
        predicted_label = torch.argmax(output.logits, dim=1).numpy()
        predicted_prob = torch.softmax(output.logits, dim=1).numpy()

        # Prepare output
        return pd.DataFrame({
            'text': text_series,
            'sentiment': predicted_label,
            'sentiment_prob': predicted_prob[:, 1]
        }, index=text_series.index)