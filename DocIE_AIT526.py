#!/usr/bin/env python
# coding: utf-8

# In[24]:


import os
import json

# 👇 Your local paths (change if needed)
train_inner_dir = "/Users/arun/Downloads/traindata"
dev_inner_dir = "/Users/arun/Downloads/dev"

# Show all files
train_inner_files = os.listdir(train_inner_dir)
dev_inner_files = os.listdir(dev_inner_dir)

# Helper function to load JSON list from file
def load_json_file(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Load all train data
train_data = []
for file in train_inner_files:
    path = os.path.join(train_inner_dir, file)
    train_data.extend(load_json_file(path))

# Load all dev data
dev_data = []
for file in dev_inner_files:
    path = os.path.join(dev_inner_dir, file)
    dev_data.extend(load_json_file(path))

print(f"✅ Loaded {len(train_data)} train and {len(dev_data)} dev documents")


# In[2]:


import re
from typing import List, Tuple

# Simple tokenizer (you'll later align this with HuggingFace tokenizers)
def extract_entities_bio(text: str, entities: List[dict]) -> Tuple[List[str], List[str]]:
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    labels = ["O"] * len(tokens)

    for entity in entities:
        for mention in entity["mentions"]:
            mention_tokens = re.findall(r"\w+|[^\w\s]", mention.strip(), re.UNICODE)
            for i in range(len(tokens) - len(mention_tokens) + 1):
                if tokens[i:i+len(mention_tokens)] == mention_tokens:
                    labels[i] = "B-" + entity["type"]
                    for j in range(1, len(mention_tokens)):
                        labels[i + j] = "I-" + entity["type"]
                    break

    return tokens, labels

# ✅ Preview 1st example
sample_doc = train_data[0]
tokens, labels = extract_entities_bio(sample_doc["doc"], sample_doc["entities"])

# Show preview
for t, l in zip(tokens[:40], labels[:40]):
    print(f"{t:15} {l}")


# In[3]:


from datasets import Dataset
from tqdm import tqdm

def prepare_ner_examples(data, limit=None):
    examples = []
    for doc in tqdm(data[:limit]):
        tokens, labels = extract_entities_bio(doc["doc"], doc["entities"])
        examples.append({
            "tokens": tokens,
            "ner_tags": labels
        })
    return examples

# ⚙️ You can set limit=5 to test faster
train_examples = prepare_ner_examples(train_data)
dev_examples = prepare_ner_examples(dev_data)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_list(train_examples)
dev_dataset = Dataset.from_list(dev_examples)

train_dataset[0]


# In[4]:


# Extract all unique labels from train dataset
label_list = list(set(tag for example in train_dataset for tag in example["ner_tags"]))
label_list = sorted(label_list)

# Create label mappings
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}


# In[5]:


import spacy
from spacy.tokens import Doc, Span
from spacy import displacy

# Load a blank English pipeline (not using a pretrained model here)
nlp = spacy.blank("en")

# Choose one document from your Hugging Face-style dataset
tokens = train_dataset[0]["tokens"]
labels = train_dataset[0]["ner_tags"]

# Convert BIO tags to SpaCy spans
def bio_to_spans(tokens, labels):
    spans = []
    start = None
    current_label = None

    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if start is not None:
                spans.append((start, i, current_label))
            start = i
            current_label = label[2:]
        elif label.startswith("I-") and start is not None and label[2:] == current_label:
            continue
        else:
            if start is not None:
                spans.append((start, i, current_label))
                start = None
                current_label = None
    if start is not None:
        spans.append((start, len(labels), current_label))
    return spans

# Convert to SpaCy Doc
doc = Doc(nlp.vocab, words=tokens)
spans = bio_to_spans(tokens, labels)
ents = []
for start, end, label in spans:
    try:
        span = Span(doc, start, end, label=label)
        ents.append(span)
    except:
        continue

doc.ents = ents

# 🖼️ Render the visualization
displacy.render(doc, style="ent", jupyter=True)


# In[31]:


from itertools import product

def prepare_relation_extraction_samples(train_data):
    all_samples = []

    for item in train_data:
        doc_text = item["doc"]
        entities = item["entities"]
        triples = item.get("triples", [])

        # Create mention lookup: only take the first mention for simplicity
        entity_id_to_mention = {e["id"]: e["mentions"][0] for e in entities}

        # Create lookup for existing relations (gold labels)
        existing_relations = {
            (t["head"], t["tail"]): t["relation"] for t in triples
        }

        # Generate all possible head-tail pairs
        for head_id, tail_id in product(entity_id_to_mention, repeat=2):
            if head_id == tail_id:
                continue  # skip self-pairs

            head_mention = entity_id_to_mention[head_id]
            tail_mention = entity_id_to_mention[tail_id]

            # Label is actual relation or "no_relation"
            label = existing_relations.get((head_mention, tail_mention), "no_relation")

            input_text = f"{head_mention} [SEP] {tail_mention} [SEP] {doc_text}"

            all_samples.append({
                "text": input_text,
                "head": head_mention,
                "tail": tail_mention,
                "relation": label
            })

    return all_samples

# 👇 Call this on your train_data
relation_data = prepare_relation_extraction_samples(train_data)

# Optional: check a few samples
for s in relation_data[:5]:
    print(s)


# In[32]:


from transformers import AutoTokenizer
from datasets import Dataset

# All unique labels including "no_relation"
unique_relations = sorted(set([sample["relation"] for sample in relation_data]))
label2id = {label: idx for idx, label in enumerate(unique_relations)}
id2label = {idx: label for label, idx in label2id.items()}


# In[33]:


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_and_encode(sample):
    encoding = tokenizer(
        sample["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    encoding["label"] = label2id[sample["relation"]]
    return encoding

# Convert list of dicts to Hugging Face Dataset
dataset = Dataset.from_list(relation_data)

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_and_encode)


# In[40]:


# No import needed — this method is built-in
train_test = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]


# In[63]:


model_checkpoint = "google/bert_uncased_L-4_H-256_A-4"

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, 
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)



# In[64]:


from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

training_args = TrainingArguments(
    output_dir="./re-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./re-logs",
    logging_steps=20,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# In[65]:


import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# In[66]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# In[67]:


trainer.train()


# In[68]:


save_path = "/Users/arun/Downloads/dev"  # your custom folder

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)


# In[69]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "/Users/arun/Downloads/dev"

model1 = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer1 = AutoTokenizer.from_pretrained(model_path)


# In[76]:


import gradio as gr
import torch
import json
import re
from itertools import combinations
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 🔧 Load model and tokenizer
model_path = "/Users/arun/Downloads/dev"
tokenizer1 = AutoTokenizer.from_pretrained(model_path)
model1 = AutoModelForSequenceClassification.from_pretrained(model_path)
model1.eval()

# 🧠 Inference Function
def predict_relation_from_json(json_file):
    with open(json_file.name, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_text = data["document"]
    doc_text_clean = re.sub(r"\s+", " ", doc_text.strip())

    # Heuristic entity extraction (capitalized words and phrases)
    matches = re.findall(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', doc_text_clean)
    entities = list(set(matches))

    # Head-tail pairs (limit for speed)
    entity_pairs = list(combinations(entities, 2))[:100]

    results = []
    for head, tail in entity_pairs:
        input_text = f"{head} [SEP] {tail} [SEP] {doc_text_clean}"
        inputs = tokenizer1(
    input_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512  # BERT's hard limit
)


        with torch.no_grad():
            outputs = model1(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).squeeze()

        pred_class = torch.argmax(probs).item()
        confidence = round(probs[pred_class].item(), 4)
        label = model1.config.id2label[pred_class]

        results.append({
            "head": head,
            "tail": tail,
            "predicted_relation": label,
            "confidence": confidence
        })

    # Return top 10 predictions
    top_predictions = sorted(results, key=lambda x: x["confidence"], reverse=True)[:10]
    return json.dumps(top_predictions, indent=2)

# 🚀 Gradio UI
iface = gr.Interface(
    fn=predict_relation_from_json,
    inputs=gr.File(label="Upload Test JSON File"),
    outputs=gr.Textbox(label="Predicted Relations"),
    title="Document-Level Relation Extraction (RE)",
    description="Upload a document-level RE JSON file. This app extracts head–tail entity pairs and predicts relations using your fine-tuned BERT model."
)

iface.launch()


# In[ ]:




