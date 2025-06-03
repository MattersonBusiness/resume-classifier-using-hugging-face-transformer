import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer

def load_and_preprocess_data(csv_path, model_name='bert-base-uncased', test_size=0.2):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    df['label_id'] = label_encoder.fit_transform(df['label'])

    # Save label mapping for future use
    label2id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    id2label = {idx: label for label, idx in label2id.items()}

    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label_id'], random_state=42)

    # Convert to Hugging Face Dataset objects
    train_ds = Dataset.from_pandas(train_df[['text', 'label_id']])
    test_ds = Dataset.from_pandas(test_df[['text', 'label_id']])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(example):
        return tokenizer(example['text'], padding='max_length', truncation=True)

    # Apply tokenization
    train_ds = train_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)

    return train_ds, test_ds, label2id, id2label