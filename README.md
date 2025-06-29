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
### Sample Chat Interaction
- You can interact with the chatbot in Devanagari script. Here's an example:
- You: श्रीदेवी खड्गमालास्तोत्ररत्नम्
- Chatbot: श्रीदेवीखड्गमालास्तोत्ररत्नम्।।रत्नसिंहासनस्थंचसर्वाभरणभूषितम्।ददौमन्त्रंविधिमयंब्रह्मविष्ण्वर्कभूषणम्।।८३२०।।महालक्ष्म्यायुतंदेवंसर्वसिद्धान्तसंयुतम्।नानावादित्रगीतैश्चदिव्यैरूपधरोपमैः॥२.४.५१॥एवंतुष्ण्मय्दसगङिमापूरयञ्जगत्क्लमस्तकश्शणतडित्पक्ष्मांगोरूलकम्।।३२२।।आभ्यन्तरनमोनमःशिवायैनमःश्रियेस्वाहा।ॐअङ्कायनमःशिवायशिवरूपायवैपद्मसौगन्धिकंतथास्वस्तिकाणीतिगायनमइतिशक्तिनिधनायनमःशिरःसम्मुखेन्दुग्रहणायतुभ्यम्।सहस्रसूर्यप्रकरप्रभामयनमोऽस्तुतेनमस्तेभगव

## 🧾 Requirements

### For All Scripts:
```bash
pip install transformers datasets torch sentencepiece regex
