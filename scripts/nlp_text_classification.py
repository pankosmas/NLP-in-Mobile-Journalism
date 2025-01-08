import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pymongo import MongoClient

def load_data_from_mongodb():
    client = MongoClient("mongodb://localhost:27017/")  # Update with your MongoDB URI
    db = client["dataset"]  # Replace with your database name
    collection = db["news_articles"]  # Replace with your collection name

    data = list(collection.find({}, {"_id": 0, "processed_text": 1, "assigned_category": 1}))
    texts, labels = [], []

    for i, doc in enumerate(data):
        if "processed_text" in doc and "assigned_category" in doc:
            texts.append(doc["processed_text"])
            labels.append(doc["assigned_category"])
        else:
            print(f"Skipping document {i}: {doc}")

    return texts, labels

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="longest", max_length=512)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

if __name__ == "__main__":
    texts, labels = load_data_from_mongodb()

    unique_labels = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    labels = [label_to_id[label] for label in labels]

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
    datasets = DatasetDict({"train": train_dataset, "test": test_dataset})

    models = {
        "GreekBERT": "nlpaueb/bert-base-greek-uncased-v1",
        "DistilBERT": "distilbert-base-multilingual-cased",
    }

    results = {}
    for model_name, model_path in models.items():
        print(f"\nFine-tuning {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=len(unique_labels)
        )

        tokenized_datasets = datasets.map(tokenize_function, batched=True, batch_size=1000)

        training_args = TrainingArguments(
            output_dir=f"./results/{model_name}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir=f"./logs/{model_name}",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            gradient_accumulation_steps=16,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        results[model_name] = eval_results

    for model_name, metrics in results.items():
        print(f"\nResults for {model_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
