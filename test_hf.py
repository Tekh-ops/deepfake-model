import sys
try:
    from transformers import pipeline
    print("Transformers available.")
    pipe = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-Model")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error: {e}")
