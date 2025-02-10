import transformers
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import umap.umap_ as umap

class ProductAttributeExtractor:
    def __init__(self, sentiment_model_name='distilbert-base-uncased', feature_model_name='BAAI/bge-small-en-v1.5'):
        self.feature_tokenizer = AutoTokenizer.from_pretrained(feature_model_name)
        self.feature_model = AutoModel.from_pretrained(feature_model_name)

        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

    def extract_common_features(self, text_list, n_dimensions):
        '''
        Extracts common features from a list of product descriptions.
        Returns a list of n common features.
        '''
        # Prepare data for feature extraction
        text_clean = ["Represent this sentence for searching relevant passages: " + desc for desc in text_list]
        
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
        # JA LEFT OFF HERE

        # Cluster embeddings to find common features

        # Create new cluster labels for each product description

    def extract_product_features(self, product_description, common_features):
        # (2) Determine each product's features from within the larger set
        product_features = self.nlp(product_description)
        relevant_features = self._filter_relevant_features(product_features, common_features)
        return relevant_features

    def categorize_sentiment(self, text):
        # (3) Categorize sentiment on a 0-10 scale of negative to positive
        sentiment = self.sentiment_analysis(text)[0]
        score = self._convert_sentiment_to_scale(sentiment)
        return score

    def extract_common_reviewer_sentiments(self, reviews, n):
        # (4) Determine a set of n common reviewer sentiments from all product reviews
        sentiments = []
        for review in reviews:
            sentiments.append(self.sentiment_analysis(review))
        common_sentiments = self._find_common_features(sentiments, n)
        return common_sentiments