# Advanced NLP Techniques for Journalism

This repository contains the implementation and resources for the paper *"Topic Modeling Techniques on News Articles and Political Speeches: Enhancing Journalism with Advanced NLP Technologies"*. The project combines advanced NLP techniques, such as BERTopic and GreekBERT, to process Greek-language news articles and political speeches, facilitating topic modeling, text classification, and dynamic visualizations.

## Contents
- **Python Scripts**:
  - `stopwords.py`: Handles the creation and management of Greek stopword lists.
  - `preprocess.py`: Preprocesses text data for NLP tasks.
  - `nlp_topic_modeling_articles.py`: Implements topic modeling for news articles using BERTopic.
  - `nlp_topic_modeling_speeches.py`: Implements topic modeling for political speeches using BERTopic.
  - `nlp_text_classification.py`: Implements text classification using GreekBERT and DistilBERT.
  - `export_mongo_to_json.py`: Exports MongoDB collections to JSON format.
  - `export_mongo_to_json_corrected.py`: Corrected version of the MongoDB export script.

- **Jupyter Notebooks**:
  - `umap_demo_visualization.ipynb`: Demonstrates UMAP visualizations for topic modeling.
  - `topic_modeling_pipeline.ipynb`: Pipeline for topic modeling with hyperparameter tuning.
  - `topic_modeling_tuning.ipynb`: Hyperparameter tuning for BERTopic.
  - `text_classification_comparison.ipynb`: Compares GreekBERT and DistilBERT for text classification.
  - `document_visualization.ipynb`: Visualizes documents and their clustering.

- **Data Availability**:
  The datasets (news articles and political speeches) used in this project are too large to include in this repository. They are available upon request. Please refer to the `README_DATA.md` file for more details.

## Features
- Topic modeling using BERTopic for both news articles and speeches.
- Text classification using fine-tuned GreekBERT and DistilBERT.
- Advanced visualizations of document clustering, topic trends, and classification performance.
- Exporting MongoDB collections to JSON for analysis.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Requirements
- Python 3.8 or later
- Required libraries are listed in `requirements.txt`. Install them using:
  ```bash
  pip install -r requirements.txt
