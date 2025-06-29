# 🦙 Sanskrit NLP Toolkit: Tokenizer + Model Fine-Tuning + Chatbot

A complete toolkit to:

- ✅ Train a **custom SentencePiece tokenizer** for **Sanskrit** (Devanagari script).
- ✅ Fine-tune a **causal language model** (e.g., Qwen2.5) on Sanskrit text.
- ✅ Run an **interactive chatbot** using the fine-tuned model.

---

## 🔧 What’s Included?

| File | Description |
|------|-------------|
| `train_tokenizer.py` | Trains and saves a SentencePiece tokenizer tailored for Sanskrit text. |
| `fine_tune_model.py` | Adapts a pre-trained language model using your Sanskrit corpus with Hugging Face Transformers. |
| `run_chatbot.py` | Runs an interactive chat interface powered by your fine-tuned Sanskrit model. |

---

## 📌 Features

### Tokenizer
- Uses BPE tokenization via SentencePiece.
- Preserves Devanagari characters and common punctuation.
- Outputs `.model` and `.vocab` files for integration into NLP pipelines.

### Model Fine-Tuning
- Supports full/fine-tuning of models like `Qwen2.5`.
- Includes dynamic padding, GPU acceleration, and logging.
- Calculates **perplexity** during evaluation.
- Saves best model based on validation loss.

### Chatbot
- Filters input/output to only allow valid **Sanskrit (Devanagari)** characters.
- Handles normalization and cleanup of verse markers and spacing.
- Uses sampling with nucleus filtering (`top_k`, `top_p`) for creative responses.

---

## 🧾 Requirements

### For All Scripts:
```bash
pip install transformers datasets torch sentencepiece regex
