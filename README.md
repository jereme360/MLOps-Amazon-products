# auto-product-ds
## Automating the data science process with Large Language Models

The goal of this project is to replicate several essential functions of a data scientist:
- Exploratory data analysis (EDA)
- Data cleaning
- Feature engineering
- Classification model specification and testing
- Model evaluation and final recommendation

Products are classified as being highly-rated or not highly-rated across four separate metrics:
- A rating above 4 stars one month after launch
- A rating above 4 starts one month after launch and high engagment (above the median number of reviews)
- A sales rank in the top 25% of a given product category
- A rating above 4.75 starts one month after launch

The LLM used in this project is OpenAI's GPT4o which is able to receive both images and text as prompts.

This project uses Amazon product listings and review data from:

Justifying recommendations using distantly-labeled reviews and fined-grained aspects
Jianmo Ni, Jiacheng Li, Julian McAuley
Empirical Methods in Natural Language Processing (EMNLP), 2019