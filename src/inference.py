import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------- Configuration -----------
model_dir = "models"  # Replace with your model directory name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- Load model and tokenizer -----------
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
model.eval()

# ----------- Define label mapping (update if needed) -----------
id2label = {
    0: "Backend Developer",
    1: "Computer Vision Engineer",
    2: "Data Analyst",
    3: "Data Scientist",
    4: "DevOps Engineer",
    5: "Digital Marketer",
    6: "Machine Learning Engineer",
    7: "Product Manager",
    8: "Software Engineer",
    9: "UI/UX Designer"
}

# ----------- Inference function -----------
def classify_resume(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

    label = id2label[predicted_class]
    return label, confidence

# ----------- CLI usage -----------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Please provide a resume text to classify.")
        print("Usage: python inference.py \"This is my resume text...\"")
        sys.exit(1)

    input_text = sys.argv[1]
    label, confidence = classify_resume(input_text)

    print(f"🧠 Predicted Label: {label}")
    print(f"🔒 Confidence: {confidence:.2f}")
