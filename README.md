# Humor Classification Project

This project is part of the coursework for the Natural Language Processing (KAN-CDSCO1002U) course. It aims to classify humor using various Natural Language Processing (NLP) models, with a focus on Large Language Models (LLMs).

The study explores the effectiveness of LLMs in classifying jokes, focusing on different levels of funniness. The problem centers on the inherent difficulty of understanding humor due to its subjective nature. The research question seeks to determine how accurately machine learning models can comprehend and classify humor.

The dataset used in this project consists of 550,000 Reddit jokes, classified into five classes based on their score.

## Project Structure

- `data`: This directory contains various types of data used in the project.
  - `external`: Contains data from third-party sources.
  - `interim`: Contains data ready to be used for model spesific preprocessing.
  - `processed`: Contains model specific preprocessed data.
- `models`: Contains the script to train the LLM models.
- `notebooks`: Contains Jupyter notebooks for exploratory data analysis, data preprocessing, and various model training and evaluation.
