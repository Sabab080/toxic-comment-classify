import logging
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Configuration (hyperparameters, paths, etc.)
def load_config():
    config = {
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "num_epochs": 3,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "dropout_prob": 0.3,
        "max_seq_length": 256,
        "logging_steps": 200
    }
    return config

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'comment_text' not in df.columns or 'toxic' not in df.columns:
            raise ValueError("Dataset is missing required columns: 'comment_text' and 'toxic'")
        
        df = df[['comment_text', 'toxic']]  # Only keep relevant columns
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        logging.info(f"Data loaded and split into {len(train_data)} train and {len(test_data)} test samples.")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error loading and preprocessing data: {str(e)}")
        raise

# Tokenize the data
def tokenize_data(data, tokenizer, config):
    try:
        encodings = tokenizer(data['comment_text'].tolist(), padding='max_length', truncation=True, max_length=config['max_seq_length'])
        return encodings
    except Exception as e:
        logging.error(f"Error tokenizing data: {str(e)}")
        raise

# Create custom Dataset class
class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Initialize and configure the model
def initialize_model():
    try:
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2,
            output_attentions=True  # Optional: To inspect attention weights
        )
        model.config.hidden_dropout_prob = 0.3
        model.config.attention_probs_dropout_prob = 0.3
        logging.info("Model initialized successfully.")
        return model
    except Exception as e:
        logging.error(f"Error initializing the model: {str(e)}")
        raise

# Define evaluation metrics
def compute_metrics(p):
    preds = p.predictions.argmax(axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    accuracy = accuracy_score(labels, preds)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# Train the model
def train_model(model, train_dataset, eval_dataset, config):
    try:
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=config['num_epochs'],
            per_device_train_batch_size=config['train_batch_size'],
            per_device_eval_batch_size=config['eval_batch_size'],
            evaluation_strategy="epoch",
            logging_dir='./logs',
            logging_steps=config['logging_steps'],
            load_best_model_at_end=True,
            fp16=True,
            weight_decay=config['weight_decay'],
            learning_rate=config['learning_rate'],
            gradient_accumulation_steps=4  # Simulating a larger batch size
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        logging.info("Starting model training...")
        trainer.train()
        logging.info("Model training completed successfully.")
        return trainer
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

# Evaluate the model
def evaluate_model(trainer, test_dataset):
    try:
        results = trainer.evaluate(test_dataset)
        logging.info(f"Evaluation results: {results}")
        return results
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        raise

# Save the trained model and tokenizer
def save_model(model, tokenizer):
    try:
        model.save_pretrained('./toxic_classifier_model')
        tokenizer.save_pretrained('./toxic_classifier_model')
        logging.info("Model and tokenizer saved successfully.")
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")
        raise

# Prediction function for new inputs
def predict(model, tokenizer, text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1)
        return "Toxic" if prediction.item() == 1 else "Non-Toxic"
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise

# Main function to orchestrate training, evaluation, and inference
def main(file_path='train.csv'):
    try:
        # Load configuration
        config = load_config()

        # Load and preprocess data
        train_data, test_data = load_and_preprocess_data(file_path)
        
        # Initialize tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # Tokenize data
        train_encodings = tokenize_data(train_data, tokenizer, config)
        test_encodings = tokenize_data(test_data, tokenizer, config)

        # Create datasets
        train_dataset = ToxicDataset(train_encodings, train_data['toxic'])
        test_dataset = ToxicDataset(test_encodings, test_data['toxic'])

        # Initialize model
        model = initialize_model()

        # Train model
        trainer = train_model(model, train_dataset, test_dataset, config)

        # Evaluate model
        evaluate_model(trainer, test_dataset)

        # Save model and tokenizer
        save_model(model, tokenizer)

        # Example prediction
        text = "You're awesome!"
        print(predict(model, tokenizer, text))  # Expected output: "Non-Toxic"
    except Exception as e:
        logging.critical(f"Critical error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    # Run the main function with the dataset
    main('train.csv')
