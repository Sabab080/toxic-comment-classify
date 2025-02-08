# Toxic Comment Classifier with DistilBERT

This project aims to classify whether a comment in a dataset is toxic or non-toxic using a fine-tuned DistilBERT model. The model is trained on the Jigsaw Toxic Comment Classification dataset, which is a widely used dataset for text classification.
Requirements

    Python 3.6+
    PyTorch
    Hugging Face Transformers
    scikit-learn
    pandas


Dataset
This model is trained on the Jigsaw Toxic Comment Classification dataset. To download the dataset:

    Go to Jigsaw Toxic Comment Classification Challenge.
    Download the train.csv file and place it in the same directory as this script.

Model
This model is based on DistilBERT and fine-tuned for binary classification (toxic vs. non-toxic comments). The model is enhanced with several techniques, including:

    Regularization with dropout and weight decay.
    Gradient Accumulation to simulate a larger batch size.
    Learning Rate Scheduling with warmup steps.
