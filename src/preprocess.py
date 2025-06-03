import pandas as pd
from datasets import Dataset, ClassLabel
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def load_and_preprocess_data(csv_path, model_name, max_length=128):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Encode labels
    labels = sorted(df['label'].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df['label'] = df['label'].map(label2id)

    # Train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Cast label column (optional)
    class_label = ClassLabel(num_classes=len(label2id), names=list(label2id.keys()))
    train_dataset = train_dataset.cast_column("label", class_label)
    test_dataset = test_dataset.cast_column("label", class_label)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenization function
    def preprocess_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    # Apply tokenization
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    # Remove original text columns
    tokenized_train = tokenized_train.remove_columns(["text", "__index_level_0__"])
    tokenized_test = tokenized_test.remove_columns(["text", "__index_level_0__"])

    return tokenized_train, tokenized_test, label2id, id2label