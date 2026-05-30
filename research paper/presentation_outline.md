# Presentation Outline — Retail E-Commerce Chatbot (13 slides)

> Two ways to build the deck:
> 1. **Auto (recommended):** `pip install python-pptx` then `python build_slides.py` → generates `presentation.pptx` with the figures already embedded.
> 2. **Manual:** copy the bullets below into PowerPoint / Google Slides and drop in the figures from `resources/figures/` and `resources/diagrams/`.

---

**Slide 1 — Title**
Fine-Tuning a Compact Instruction-Tuned Language Model for Retail E-Commerce Customer-Support Conversational AI · Harish Chandra · Master's in CS: AI & ML, Woolf University · 31 May 2026

**Slide 2 — Problem & Research Question**
- E-commerce support is high-volume and repetitive: orders, refunds, cancellations, returns, delivery, payments.
- Customers expect instant, 24/7 answers — costly with humans alone.
- Large LLMs work but are expensive; out of reach for smaller retailers.
- **RQ:** Can `flan-t5-small` be fine-tuned on retail support data to give accurate, relevant answers on a single free-tier GPU?

**Slide 3 — Why Retail E-Commerce?** (industry context)
- Multi-trillion-dollar, fast-growing sector; support volume scales with sales.
- Chatbot market ~$7.76B (2024) → ~$27.29B (2030); customer service = largest application.
- A few recurring intents dominate → highly automatable.
- Business value: lower cost-per-contact, 24/7 instant responses, consistency, scalability.

**Slide 4 — Literature & Research Gap**
- Foundations: Transformer (2017), T5 (2020), FLAN instruction tuning (2022).
- Applied: fine-tuned/generative support chatbots (Nandi 2024; Khennouche 2023).
- Gap 1: compact, low-resource models for retail support under-reported.
- Gap 2: most work isolates intent classification, not end-to-end generation.
- Contribution: reproducible single-GPU, end-to-end study + failure analysis.

**Slide 5 — Methodology / Pipeline** *(image: `resources/diagrams/pipeline_flow.png`)*
- Fine-tuning only (no RAG) · flan-t5-small (~77M) · FP32 on T4 · 5 epochs + early stopping · beam search (4 beams).

**Slide 6 — Dataset & EDA** *(image: `resources/figures/fig2_intent_distribution.png`)*
- Bitext retail e-commerce (HF, CDLA, synthetic — no PII) · 44,884 pairs · 13 categories · 46 intents.
- Split 35,907 / 4,488 / 4,489 (stratified) · token lengths → input 128 / target 256.

**Slide 7 — Fine-Tuning Setup**
- LR 3e-4, weight decay 0.01, 500-step linear warmup · batch 8 × grad-accum 2 = eff. 16.
- Labels padded with −100 · best checkpoint by val loss · ≤5 epochs (within 25 budget).

**Slide 8 — Results: Baseline vs Fine-Tuned** *(image: `resources/figures/fig5_metric_comparison.png`)*
- ROUGE-L 0.057 → 0.453 (~8×) · BLEU 0.0 → 29.27 · BERTScore-F1 0.818 → 0.909 · avg 7 → 133 words.

**Slide 9 — Results: Convergence & Per-Category** *(images: `fig4_training_loss.png`, `fig6_per_category_rougeL.png`)*
- Clean convergence, no overfitting · strongest: ACCOUNT/CART · weakest: FEEDBACK.

**Slide 10 — Demonstration & Qualitative Behaviour** *(image: `resources/diagrams/gradio_demo4.png`)*
- Gradio chat UI (live demo in the video).
- In-domain: structured step-by-step answers; baseline gave useless one-liners.
- Ambiguous words (e.g. "refund") → sensible inferred intent.
- Out-of-domain (weather/jokes) → answered in support style (over-generalisation).

**Slide 11 — Discussion & Limitations**
- Domain adaptation, not scale, was the limiting factor.
- Cheap to reproduce/retrain; automate high-volume intents first.
- No out-of-scope detection → needs intent filter + human fallback.
- Synthetic data; automatic metrics approximate human judgement.

**Slide 12 — Conclusion & Future Work**
- Compact, free-tier-trainable model handles routine retail support well (8× ROUGE-L gain).
- Contribution: reproducible blueprint for low-resource domain chatbots.
- Future: out-of-scope detection + human handoff; real logs + human eval; flan-t5-base comparison; multi-turn; multilingual.

**Slide 13 — Key References**
- Vaswani 2017 · Raffel 2020 (T5) · Chung 2022 (FLAN) · Nandi 2024; Khennouche 2023 · Bitext dataset (HF, CDLA).
