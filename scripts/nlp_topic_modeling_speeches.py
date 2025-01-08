import re
from pymongo import MongoClient
from pymongo.errors import WriteError
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.coherencemodel import CoherenceModel
from hdbscan import HDBSCAN
from gensim.corpora import Dictionary
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
speeches_db = mongo_client["prime_minister"]

# Load data from MongoDB collections
speeches_collection = speeches_db["articles_speeches"]  # Replace with your speeches collection name

# Convert MongoDB data to pandas DataFrames
speeches_df = pd.DataFrame(list(speeches_collection.find({}, {"_id": 0, "processed_text": 1})))

# Ensure datasets are non-empty
if speeches_df.empty:
    raise ValueError("One or both collections are empty. Please check your MongoDB collections.")

# Topic Modeling Pipeline
# Define the output directory for saving models and figures
output_dir = "topic_modeling_results"
os.makedirs(output_dir, exist_ok=True)

# Load preprocessed data from MongoDB
speeches_data = pd.DataFrame(list(speeches_collection.find({}, {"_id": 0, "processed_text": 1})))

# Initialize embedding model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Ensure all entries in 'processed_text' are strings
speeches_data["processed_text"] = speeches_data["processed_text"].fillna("").astype(str)

# Optionally, drop rows with empty 'processed_text'
speeches_data = speeches_data[speeches_data["processed_text"].str.strip() != ""]

# Clean and preprocess text
speeches_data["processed_text"] = speeches_data["processed_text"].apply(lambda x: remove_stopwords(x, words_list))

# Precompute embeddings
print("Generating embeddings...")
# articles_embeddings = embedding_model.encode(articles_data["processed_text"].tolist(), show_progress_bar=True, device="cuda")
speeches_embeddings = embedding_model.encode(speeches_data["processed_text"].tolist(), show_progress_bar=True)

# Initialize GPU-accelerated UMAP and HDBSCAN
umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', min_samples=5, prediction_data=True)

# Define custom CountVectorizer with stopwords
custom_vectorizer = CountVectorizer(stop_words=words_list)

# Train BERTopic for speeches
print("Training BERTopic for speeches...")
speeches_topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, calculate_probabilities=False, language="greek", vectorizer_model=custom_vectorizer)
speeches_topics, speeches_probs = speeches_topic_model.fit_transform(speeches_data["processed_text"].tolist(), speeches_embeddings)

# Save the model and visualizations for speeches
speeches_topic_model.save(os.path.join(output_dir, "speeches_topic_model.bertopic"))
speeches_topic_model.visualize_barchart(top_n_topics=10).write_html(os.path.join(output_dir, "speeches_topics_barchart.html"))
speeches_topic_model.visualize_topics().write_html(os.path.join(output_dir, "speeches_topics_interactive.html"))
speeches_topic_model.visualize_hierarchy().write_html(os.path.join(output_dir,"speeches_topic_hierarchy.html"))
speeches_topic_model.visualize_topics().write_html(os.path.join(output_dir,"speeches_embedding_visualization.html"))
speeches_topic_model.visualize_heatmap().write_html(os.path.join(output_dir,"speeches_topic_similarity_heatmap.html"))
# speeches_topic_model.visualize_documents(speeches_embeddings).write_html(os.path.join(output_dir,"speeches_documents.html")) ---> ERROR ITERATABLE
# speeches_topic_model.visualize_document_datamap(speeches_data["processed_text"].tolist(), embeddings=speeches_embeddings).write_html(os.path.join(output_dir,"speeches_documents_datamap.html"))
# speeches_topic_model.visualize_distribution(speeches_probs[0]).write_html(os.path.join(output_dir,"speeches_topic_prob_distribution.html")) ---> CANNOT VISUALIZE IF DECLARED FALSE 

# Get top documents for each topic
top_docs = speeches_topic_model.get_representative_docs()

# Save top documents to a text file
with open("speeches_top_documents.txt", "w", encoding="utf-8") as f:
    for topic_num, docs in top_docs.items():
        f.write(f"Topic {topic_num}:\n")
        f.write("\n".join(docs[:5]))  # Save top 5 documents per topic
        f.write("\n\n")

# Get topic frequencies
topic_freq = speeches_topic_model.get_topic_freq()
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
topic_info = speeches_topic_model.get_topic_info()
all_topics = speeches_topic_model.get_topics()

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
topics = speeches_topic_model.get_topics()
top_words_per_topic = [[word for word, _ in words] for words in topics.values()]
# Get the processed text for coherence calculation
texts = [doc.split() for doc in speeches_data["processed_text"].tolist()]
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
with open("speeches_topic_model_evaluation.txt", "w") as f:
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

print("Topic modeling completed with optimizations. All models and visualizations saved.")
