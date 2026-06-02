# test_pipeline.py

from modules.metadata_analysis.metadata import create_metadata
from modules.core.legal import generate_legal_notice

# Dummy input
file_path = "sample.jpg"
result = "Fake"   # try "Real" also later
confidence = 0.85

# Step 1: Create metadata
metadata = create_metadata(file_path, result, confidence)

# Step 2: Generate legal notice
notice = generate_legal_notice(metadata)

# Print metadata
print("\n--- METADATA ---")
for key, value in metadata.items():
    print(f"{key}: {value}")

# Print legal notice (if exists)
if notice:
    print("\n--- LEGAL NOTICE ---")
    print(notice)
else:
    print("\nNo legal action required.")