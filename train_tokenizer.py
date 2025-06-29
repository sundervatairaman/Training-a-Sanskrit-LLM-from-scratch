import os
import re
import sentencepiece as spm

# Configuration
model_name = "alpindale/Qwen2.5-0.2B"
output_dir = "./sanskrit_finetuned_tokenizer"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Verify Input File
training_data_file = "training_data_with_filename.txt"
if not os.path.exists(training_data_file):
    raise FileNotFoundError(f"The training data file '{training_data_file}' was not found.")

# Step 2: Clean and Preprocess Data
def clean_text(text):
    # Preserve Sanskrit-specific characters and punctuation
    return re.sub(r"[^\u0900-\u097F\s\.,!?;:'\"()\-–॰॥]", "", text)

print("Preparing training data...")
with open(training_data_file, "r", encoding="utf-8") as f:
    text_data = [clean_text(line.strip()) for line in f.readlines() if line.strip()]

if not text_data:
    raise ValueError("The training data file is empty or contains no valid text.")

# Debug: Print the first few lines of cleaned data
print(f"Cleaned text (first 5 lines): {text_data[:5]}")

# Combine all lines into a single string and split into sentences
training_corpus = "\n".join(text_data)
corpus_file = "sanskrit_training_corpus.txt"

# Save the cleaned corpus to a temporary file
with open(corpus_file, "w", encoding="utf-8") as f:
    f.write(training_corpus)

# Step 3: Train a SentencePiece Tokenizer
print("Training SentencePiece tokenizer...")
try:
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix="sanskrit_tokenizer",
        vocab_size=8000,  # Adjust vocabulary size based on dataset complexity
        character_coverage=1.0,  # Ensure full coverage of characters
        model_type="bpe",  # Byte-Pair Encoding (BPE) works well for Sanskrit
        max_sentence_length=8192  # Increase the maximum sentence length
    )
except Exception as e:
    print(f"Tokenizer training failed: {e}")
    raise

# Step 4: Load and Save the Fine-Tuned Tokenizer
print("Loading and saving fine-tuned tokenizer...")
try:
    sp = spm.SentencePieceProcessor()
    sp.load("sanskrit_tokenizer.model")
except Exception as e:
    print(f"Failed to load tokenizer: {e}")
    raise

# Save the tokenizer files to the output directory
tokenizer_files = ["sanskrit_tokenizer.model", "sanskrit_tokenizer.vocab"]
for file in tokenizer_files:
    try:
        os.rename(file, os.path.join(output_dir, file))
    except Exception as e:
        print(f"Failed to move tokenizer file '{file}': {e}")
        raise

print(f"Fine-tuned tokenizer saved to {os.path.abspath(output_dir)}")

# Step 5: Verify the Fine-Tuned Tokenizer
print("Verifying fine-tuned tokenizer...")
sample_text = "ॐ नमः शिवाय । संस्कृतभाषा सुललिता ॥"
tokens = sp.encode_as_pieces(sample_text)
print(f"Sample text: {sample_text}")
print(f"Tokens: {tokens}")

# Cleanup temporary files
try:
    os.remove(corpus_file)
except Exception as e:
    print(f"Failed to clean up temporary file '{corpus_file}': {e}")