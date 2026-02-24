# Local PII Detector (MiniLM-L6)

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20Weights-blue)](https://huggingface.co/Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection)

A highly efficient, lightweight (~90MB) Named Entity Recognition model designed to detect Personally Identifiable Information (PII) entirely locally.

Sending sensitive user data (like names, emails, and phone numbers) to third-party cloud LLM APIs is a massive security and compliance risk. This model allows you to scrub sensitive data **offline** and on a **CPU** in milliseconds before it ever leaves your secure environment.

This is a fine-tuned version of `sentence-transformers/all-MiniLM-L6-v2` trained on the `nvidia/Nemotron-PII` dataset.

## 🚀 Quick Start (Inference)

You can run this model locally in just a few lines of Python using the `transformers` library.

### 1. Install Dependencies
```bash
pip install transformers torch
```

### 2. Run the Prediction Script

```python
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

model_id = "Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection"

print("Downloading/Loading model locally...")
# Load the tokenizer and model locally
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)

# Initialize the pipeline
# aggregation_strategy="simple" merges B- and I- tags into single coherent words/phrases
pii_pipeline = pipeline(
    "token-classification", 
    model=model, 
    tokenizer=tokenizer, 
    aggregation_strategy="simple" 
)

# Text containing dummy PII for testing
sample_text = (
    "John Doe's bank routing number is 123456789. "
    "He is 45 years old and his email is john.doe@example.com."
)

print("\nRunning inference locally...")
results = pii_pipeline(sample_text)

# Display the detected PII entities
print("\nDetected PII Entities:")
for entity in results:
    print(f"- Entity: {entity['word']}")
    print(f"  Label:  {entity['entity_group']}")
    print(f"  Score:  {entity['score']:.4f}\n")
```

## Training Code
This model was trained using the Jupyter Notebook script MiniLM-L6-v2-pii-training-script.ipynb in this repository.

## Model Performance (Validation Set):
| Precision | Recall | F1 Score | Accuracy |
|:---:|:---:|:---:|:---:|
| 0.933025 | 0.947065 | **0.939993** | **0.992261** |

## ⚠️ Production Disclaimer
Please note that automated PII detection is not completely foolproof, and accuracy will vary depending on your specific data context and formatting. We strongly advise thoroughly validating the model on your own data and incorporating human oversight to ensure it meets your intended purpose before any production deployment.

## 📬 Contact
Please reach out if you have questions or feedback. We also do custom projects, consultating, freelance and collaboration.

**Email:** [thieves@negativestarinnovators.com](mailto:thieves@negativestarinnovators.com)

## 💖 Support This Project
If you find this PII detector useful for your projects or business, please support our work!

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png)](https://buymeacoffee.com/negativestarinnovators)
