import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configuration
model_name_or_path ="./fine_tuned_model_output_1"  # Path to your fine-tuned model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

# Ensure pad_token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Chatbot loop
import re

# Function to keep only Devanagari (Sanskrit) characters and common punctuation/spaces
def filter_sanskrit(text):
 #import re

#import re

#def clean_sanskrit(text):
    # Keep only Devanagari (Sanskrit), spaces, punctuation (। ॥), and digits (for verse numbers)
    cleaned = re.sub(r'[^\u0900-\u097F\s।॥०-९]', '', text)

    # Normalize verse markers: turn ॥ ७ ११ २३ ॥ → ॥ ७.११.२३ ॥
    cleaned = re.sub(r'॥\s*(\d+)\s+(\d+)\s+(\d+)\s*॥', r'॥ \1.\2.\3 ॥', cleaned)

    # Fix common word splits (e.g., "राक्षसा श्चा" → "राक्षसाश्चा")
    cleaned = re.sub(r'(\u0900-\u0965)(\s+)(\u0900-\u0965)', r'\1\3', cleaned)

    # Remove extra spaces between words
    lines = []
    for line in cleaned.split('\n'):
        words = line.strip().split()
        fixed_line = ''.join(words) if len(words) > 0 else ''
        lines.append(fixed_line)
    cleaned = '\n'.join(lines)

    # Final cleanup: remove multiple spaces and fix line breaks
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()

    return cleaned

def chatbot():
    print("Chatbot initialized. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        # Filter only Sanskrit characters from input
        filtered_input = filter_sanskrit(user_input)

        if not filtered_input.strip():
            print("Chatbot: कृपया संस्कृत में प्रश्न दो।")  # Please ask a question in Sanskrit.
            continue

        # Tokenize the filtered input
        inputs = tokenizer(
            filtered_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                repetition_penalty=1.5,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.6
            )

        # Decode and filter only Sanskrit characters in output
        raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        filtered_response = filter_sanskrit(raw_response).strip()

        print(f"Chatbot: {filtered_response}")


# Run the chatbot
if __name__ == "__main__":
    chatbot()