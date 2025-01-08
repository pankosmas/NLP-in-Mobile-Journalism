import re
from pymongo import MongoClient
from pymongo.errors import WriteError
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from hdbscan import HDBSCAN
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from stopwords import words_list
from umap import UMAP
import pandas as pd
import os


def remove_stopwords(text, stopwords):
    words = text.split()
    return " ".join(word for word in words if word not in stopwords)

# MongoDB Connection Setup
mongo_client = MongoClient("mongodb://localhost:27017/")  # Replace with your connection string
articles_db = mongo_client["dataset"]  # Replace with your database name

# Load data from MongoDB collections
articles_collection = articles_db["news_articles"]  # Replace with your articles collection name

# Convert MongoDB data to pandas DataFrames
articles_df = pd.DataFrame(list(articles_collection.find({}, {"_id": 0, "processed_text": 1})))

# Ensure datasets are non-empty
if articles_df.empty:
    raise ValueError("One or both collections are empty. Please check your MongoDB collections.")

# Topic Modeling Pipeline
# Define the output directory for saving models and figures
output_dir = "topic_modeling_results"
os.makedirs(output_dir, exist_ok=True)

# Load preprocessed data from MongoDB
articles_data = pd.DataFrame(list(articles_collection.find({}, {"_id": 0, "processed_text": 1, "timestamp": 2, "assigned_category": 3})))

# Initialize embedding model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Ensure all entries in 'processed_text' are strings
articles_data["processed_text"] = articles_data["processed_text"].fillna("").astype(str)

# Optionally, drop rows with empty 'processed_text'
articles_data = articles_data[articles_data["processed_text"].str.strip() != ""]

# Clean and preprocess text
articles_data["processed_text"] = articles_data["processed_text"].apply(lambda x: remove_stopwords(x, words_list))

# Precompute embeddings
print("Generating embeddings...")
articles_embeddings = embedding_model.encode(articles_data["processed_text"].tolist(), show_progress_bar=True, device="cuda")

# Initialize GPU-accelerated UMAP and HDBSCAN
umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', min_samples=5, prediction_data=True)

# Define custom CountVectorizer with stopwords
custom_vectorizer = CountVectorizer(stop_words=words_list)

# Train BERTopic for articles
print("Training BERTopic for articles...")

# Train BERTopic for articles
print("Training BERTopic for articles...")
articles_topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, calculate_probabilities=False, language="greek", vectorizer_model=custom_vectorizer)
articles_topics, articles_probs = articles_topic_model.fit_transform(articles_data["processed_text"].tolist(), articles_embeddings)

# Save the model and visualizations for speeches
articles_topic_model.save(os.path.join(output_dir, "articles_topic_model.bertopic"))
articles_topic_model.visualize_barchart(top_n_topics=10).write_html(os.path.join(output_dir, "articles_topics_barchart.html"))
articles_topic_model.visualize_topics().write_html(os.path.join(output_dir, "articles_topics_interactive.html"))
articles_topic_model.visualize_hierarchy().write_html(os.path.join(output_dir,"articles_topic_hierarchy.html"))
articles_topic_model.visualize_topics().write_html(os.path.join(output_dir,"articles_embedding_visualization.html"))
articles_topic_model.visualize_heatmap().write_html(os.path.join(output_dir,"articles_topic_similarity_heatmap.html"))
# speeches_topic_model.visualize_documents(speeches_embeddings).write_html(os.path.join(output_dir,"articles_documents.html")) ---> ERROR ITERATABLE
# speeches_topic_model.visualize_document_datamap().write_html(os.path.join(output_dir,"articles_documents_datamap.html")) ---> SAME ERROR, MISSING DOCS
# speeches_topic_model.visualize_distribution(speeches_probs[0]).write_html(os.path.join(output_dir,"articles_topic_prob_distribution.html")) ---> CANNOT VISUALIZE IF DECLARED FALSE 

# Get top documents for each topic
top_docs = articles_topic_model.get_representative_docs()

# Save top documents to a text file
with open("speeches_top_documents.txt", "w", encoding="utf-8") as f:
    for topic_num, docs in top_docs.items():
        f.write(f"Topic {topic_num}:\n")
        f.write("\n".join(docs[:5]))  # Save top 5 documents per topic
        f.write("\n\n")

# Get topic frequencies
topic_freq = articles_topic_model.get_topic_freq()
# Exclude the -1 topic (outliers)
topic_freq = topic_freq[topic_freq["Topic"] != -1]
# Plot the topic frequency distribution
plt.figure(figsize=(10, 6))
plt.bar(topic_freq["Topic"], topic_freq["Count"], color="skyblue")
plt.xlabel("Topics", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Topic Frequency Distribution", fontsize=16)
plt.xticks(topic_freq["Topic"])
plt.savefig("speeches_topic_frequency_distribution.png")
plt.show()

# Get topic info and all topics
topic_info = articles_topic_model.get_topic_info()
all_topics = articles_topic_model.get_topics()

# File to save the summary
output_file = "speeches_detailed_topic_summary.txt"

# Write the detailed topic summary to a file
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Detailed Topic Summary:\n")
    for _, row in topic_info.iterrows():
        topic_num = row["Topic"]
        f.write(f"Topic {topic_num}:\n")
        f.write(f"  Document Count: {row['Count']}\n")
        
        # Get top words for the topic
        if topic_num != -1:  # Skip outliers
            words = all_topics.get(topic_num, [])
            f.write("  Top Words:\n")
            for word, weight in words[:10]:  # Top 10 words
                f.write(f"    {word}: {weight:.4f}\n")
        f.write("\n")


# Extract top words from each topic ================================= EVALUATION METRICS ===========================================
topics = articles_topic_model.get_topics()
top_words_per_topic = [[word for word, _ in words] for words in topics.values()]
# Get the processed text for coherence calculation
texts = [doc.split() for doc in articles_data["processed_text"].tolist()]
# Create a dictionary from the processed texts
dictionary = Dictionary(texts)
dictionary.filter_extremes(no_below=5, no_above=0.5)
# Compute coherence using the "c_v" metric
coherence_model = CoherenceModel(topics=top_words_per_topic, texts=texts, dictionary=dictionary, coherence='c_v', processes=1)
coherence_score = coherence_model.get_coherence()

def topic_diversity(topics, top_k=10):
    # Flatten top-k words from all topics
    unique_words = set()
    total_words = 0
    for topic in topics.values():
        words = [word for word, _ in topic[:top_k]]
        unique_words.update(words)
        total_words += len(words)
    return len(unique_words) / total_words

# Compute topic diversity
diversity_score = topic_diversity(topics)

# Save evaluation results
with open("articles_topic_model_evaluation.txt", "w") as f:
    f.write(f"Topic Coherence (c_v): {coherence_score}\n")
    f.write(f"Topic Diversity: {diversity_score}\n")


# Generate word clouds for each topic
# font_path = 'C:/Windows/Fonts/Arial.ttf'
# for topic_num in range(len(speeches_topic_model.get_topics())):
#     # Get words for the topic
#     words = dict(speeches_topic_model.get_topic(topic_num))
    
#     # Generate a word cloud
#     wordcloud = WordCloud(width=800, height=400, background_color="white", font_path=font_path).generate_from_frequencies(words)
    
#     # Save the word cloud
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation="bilinear")
#     plt.axis("off")
#     plt.title(f"Topic {topic_num}", fontsize=16)
#     plt.savefig(f"topic_{topic_num}_wordcloud.png")
#     plt.close()

# ============================== DYNAMIC TOPIC MODELING FOR NEWS =========================================================
# Ensure 'timestamp' column is in datetime format
# speeches_data["timestamp"] = pd.to_datetime(speeches_data["timestamp"], unit="s")
# Generate topics over time
# speeches_topic_model.update_topics(speeches_data["processed_text"].tolist(), timestamps=speeches_data["timestamp"])
# # Visualize topic evolution
# evolution_fig = speeches_topic_model.visualize_barchart(top_n_topics=10, custom_labels=True)
# # Save as an HTML file
# evolution_fig.write_html("topic_evolution.html")

print("Topic modeling completed with optimizations. All models and visualizations saved.")
