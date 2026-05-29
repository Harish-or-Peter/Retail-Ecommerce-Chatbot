# Retail E-Commerce Customer-Support Chatbot (FLAN-T5)

Fine-tuning a compact instruction-tuned language model (`google/flan-t5-small`) for **retail e-commerce customer support**, with a full evaluation against the un-tuned baseline and a Gradio demo.

Companion code for the research paper *"Fine-Tuning a Compact Instruction-Tuned Language Model for Retail E-Commerce Customer-Support Conversational AI"* (Harish Chandra, Master's in CS: AI & ML, Woolf University, 2026).

## What this project does
- Fine-tunes `flan-t5-small` (~77M params) on the **Bitext retail e-commerce** support dataset (44,884 instruction–response pairs, 13 categories, 46 intents).
- Trains on a single **Google Colab T4 GPU** in full precision (FP32 — FLAN-T5 is unstable in FP16 and the T4 has no bf16).
- Evaluates fine-tuned vs. baseline with **ROUGE, BLEU, BERTScore**, plus per-category analysis and a qualitative test battery.
- Serves the model behind a **Gradio** chat interface.

## Results (held-out test set)

| Metric | Zero-shot baseline | Fine-tuned |
|---|---|---|
| ROUGE-1 | 0.067 | **0.617** |
| ROUGE-2 | 0.017 | **0.361** |
| ROUGE-L | 0.057 | **0.453** |
| BLEU | 0.00 | **29.27** |
| BERTScore-F1 | 0.818 | **0.909** |
| Avg. response length (words) | 7.1 | 133.5 |

Strongest categories: ACCOUNT, CART. Weakest: FEEDBACK. Healthy convergence, no overfitting.

## Repository layout
```
chatbot/
  Capstone_Project_CHAT_BOT_LLM_Deep_Learning_for_NLP_v2.ipynb   # end-to-end Colab notebook
  retail-ecommerce-chatbot-v2/                                    # tokenizer + config (model.safetensors not tracked — see below)
research paper/
  Retail_Ecommerce_Research_Paper_DRAFT.md                        # paper draft
  resources/figures/                                              # exported plots (fig1-6)
  resources/                                                      # metrics.json, qualitative_eval.csv, comparison_metrics.csv
```

## Reproducing
1. Open `chatbot/Capstone_Project_CHAT_BOT_LLM_Deep_Learning_for_NLP_v2.ipynb` in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. *Run all*. The notebook downloads the dataset, fine-tunes, evaluates, exports figures/metrics, and launches the Gradio demo.

## Notes on large files
- `model.safetensors` (~293 MB) is **not** tracked (exceeds GitHub's 100 MB limit). Regenerate it by running the notebook.
- The dataset (~40 MB) is **not** tracked; it is downloaded automatically from Hugging Face: [`bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset) (CDLA-Sharing-1.0).

## License / data
Dataset © Bitext, CDLA-Sharing-1.0 (synthetic, no PII). Base model: `google/flan-t5-small`.
