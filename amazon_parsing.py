import pandas as pd
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

class AmazonJSONParser:
    def parser(self, JSON_file):
        '''
        Parses Amazon product JSON file.
        '''
        df = pd.read_json(JSON_file, lines=True)
        return df

class ProductAttributeExtractor:
    def __init__(self, max_features=100):
        self.nlp = spacy.load('en_core_web_sm')
        self.max_features = max_features

    def process_batch(self, df, text_column):
        '''
        Extract features from all descriptions
        '''
        all_features = []
        for text in df[text_column]:
            doc = self.nlp(text)
            # Get noun chunks
            features = [chunk.text.lower() for chunk in doc.noun_chunks
                        if chunk.root.pos_ in ['NOUN', 'PROPN']]
            all_features.extend(features)

        tfidf = TfidfVectorizer(max_features=self.max_features)
        tfidf.fit([' '.join(all_features)])
        feature_set = tfidf.set(tfidf.get_feature_names_out())