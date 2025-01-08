# -*- coding: utf-8 -*-
"""Visualization of documents collected and category centers.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Xy38JLOq4yLaW396IGMylUKHVMXIuehd

# *Visualization of documents collected and category centers*
"""

!pip install pymongo sentence-transformers torch scikit-learn umap-learn matplotlib

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import pandas as pd
from pymongo import MongoClient

import pandas as pd

# Load the JSON file
articles = pd.read_json("/content/dataset.news_articles.json")  # Replace with your JSON file name

# Inspect the first few rows
print(articles.head())

# Ensure columns 'text' and 'assigned_category' are present
if 'text' not in articles.columns or 'assigned_category' not in articles.columns:
    raise ValueError("Required columns ('text', 'assigned_category') are missing.")

# Load SBERT model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Generate embeddings for articles' text
articles['text_embedding'] = articles['text'].apply(lambda x: model.encode(x, convert_to_tensor=False))

# Convert embeddings to a numpy array
text_embeddings = np.stack(articles['text_embedding'].to_list())

# Define categories and their keywords
iptc_categories = [
    {"name": "Τέχνες, Πολιτισμός, Ψυχαγωγία και Μέσα", "keywords": ["Τέχνες, Πολιτισμός, Ψυχαγωγία και Μέσα", "μουσική", "θέατρο", "τέχνες", "ιστορία", "μουσεία", "βιβλίο", "παραστάσεις", "σινεμά"]},
    {"name": "Διαμάχη, Πόλεμος, Ειρήνη", "keywords": ["Διαμάχη, Πόλεμος, Ειρήνη", "τρομοκρατία", "πραξικόπημα", "πόλεμος", "θύματα πολέμου", "εμπόλεμη ζώνη", "στρατός"]},
    {"name": "Έγκλημα, Νόμος, Δικαιοσύνη", "keywords": ["Έγκλημα, Νόμος, Δικαιοσύνη", "παρενόχληση", "έγκλημα", "δικαστήριο", "βανδαλισμοί", "δίκαιο", "νομική"]},
    {"name": "Καταστροφή, Ατύχημα, Επείγον Περιστατικό", "keywords": ["Καταστροφή, Ατύχημα, Επείγον Περιστατικό", "έκρηξη", "πνιγμός", "ατύχημα", "δυστύχημα", "καταστροφή"]},
    {"name": "Οικονομία, Επιχειρήσεις", "keywords": ["Οικονομία, Επιχειρήσεις", "αγορά", "επιχειρήσεις", "επενδύσεις", "οικονομία"]},
    {"name": "Εκπαίδευση", "keywords": ["Εκπαίδευση", "παιδεία", "μαθητές", "φοιτητές", "δάσκαλοι", "καθηγητές", "μάθηση", "σχολείο", "πανεπιστήμιο", "ΑΕΙ", "ΤΕΙ", "ΙΕΚ"]},
    {"name": "Περιβάλλον", "keywords": ["Περιβάλλον", "κλιματική αλλαγή", "μόλυνση περιβάλλοντος", "φύση", "ανανεώσιμες πηγές"]},
    {"name": "Υγεία", "keywords": ["Υγεία", "ασθένεια", "περίθαλψη", "ασφάλιση", "ιδιωτική ασφάλιση", "δημόσια ασφάλιση", "υγεία", "θεραπεία", "νοσοκομείο", "νοσηλευτές", "ιατροί"]},
    {"name": "Εργασία", "keywords": ["Εργασία", "εργασιακά", "αγορά εργασίας", "ανεργία", "σύνταξη", "συνταξιοδότηση"]},
    {"name": "Lifestyle", "keywords": ["καλή ζωή", "τρόπος ζωής", "lifestyle", "ελεύθερος χρόνος"]},
    {"name": "Πολιτική", "keywords": ["Πολιτική", "εκλογές", "κόμματα", "κυβέρνηση", "αντιπολίτευση", "διεθνείς σχέσεις", "πολιτικά", "βουλή", "κοινοβούλιο", "βουλευτές", "πρωθυπουργός", "πρόεδρος"]},
    {"name": "Θρησκεία", "keywords": ["Θρησκεία", "θρησκευτική διαμάχη", "θεός", "εκκλησία", "τελετή", "αιρέσεις", "χριστιανισμός", "μουσουλμανισμός"]},
    {"name": "Επιστήμη και Τεχνολογία", "keywords": ["Επιστήμη και Τεχνολογία", "βιοϊατρική επιστήμη", "μαθηματικά", "φυσική επιστήμη", "επιστημονικό ίδρυμα", "έρευνα", "τεχνολογία", "τεχνητή νοημοσύνη", "υπολογιστής"]},
    {"name": "Κοινωνία", "keywords": ["Κοινωνία", "κοινωνίες", "ισότητα", "δικαιώματα", "αξίες", "μετανάστευση", "δημογραφικά", "διακρίσεις", "οικογένεια"]},
    {"name": "Αθλητισμός", "keywords": ["αναβολικά", "επίτευγμα αθλητή", "διάκριση αθλητή", "μετάλλιο", "αθλητικό γεγονός", "αθλητική οργάνωση", "προπονητική", "αθλητισμός", "αθλήματα"]},
    {"name": "Καιρός", "keywords": ["Καιρός", "πρόγνωση καιρού", "στατιστική καιρού", "προειδοποίηση καιρικών φαινομένων", "βροχές", "καταιγίδες"]}
            # Add your categories and keywords
]

# Embed IPTC categories by averaging their keyword embeddings
category_embeddings = {
    category: np.mean(model.encode(keywords, convert_to_tensor=False), axis=0)
    for category, keywords in iptc_categories.items()
}

# Convert category embeddings to a numpy array
category_embedding_array = np.array(list(category_embeddings.values()))

# Save embeddings and metadata
np.save("text_embeddings.npy", text_embeddings)
np.save("category_embeddings.npy", category_embedding_array)
with open("category_names.json", "w") as f:
    json.dump(list(category_embeddings.keys()), f)

# Save the model
model.save("sbert_greek_model")

# Combine text and category embeddings for visualization
all_embeddings = np.vstack([text_embeddings, category_embedding_array])

# Perform t-SNE to reduce dimensionality
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
reduced_embeddings = tsne.fit_transform(all_embeddings)

# Split reduced embeddings
text_reduced = reduced_embeddings[:len(text_embeddings)]
category_reduced = reduced_embeddings[len(text_embeddings):]

# Plot texts and categories
plt.figure(figsize=(12, 8))
colors = plt.cm.get_cmap("tab10", len(category_names))

# Plot articles
for i, category in enumerate(category_names):
    indices = articles[articles['assigned_category'] == category].index
    plt.scatter(
        text_reduced[indices, 0],
        text_reduced[indices, 1],
        label=f"Articles: {category}",
        alpha=0.5,
        s=20,
        color=colors(i),
    )

# Plot category centers
for i, (category, coord) in enumerate(zip(category_names, category_reduced)):
    plt.scatter(
        coord[0],
        coord[1],
        label=f"Category Center: {category}",
        s=200,
        color=colors(i),
        edgecolor="black",
    )

plt.legend()
plt.title("2D Visualization of Articles and IPTC Categories")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

plt.savefig("articles_categories_visualization.png", dpi=300)

