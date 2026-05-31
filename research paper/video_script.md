# Video Script — Retail E-Commerce Chatbot Research Paper

**Author:** Harish Chandra · Target length: **15–25 min** (aim ~16–18). Follows the official 8-part video rubric.

> `[SHOW: …]` = what to display on screen. Speak naturally in your own words — these are notes, not a teleprompter. Record the **live Gradio demo** in Part 5. Slides: `presentation.pptx`. Figures: `resources/figures/`.

---

### PART 1 — Introduction (≈1.5 min) `[SHOW: Title slide]`
"Hello, my name is Harish Chandra. In this video I'll walk through my Industry Immersion research paper: *Fine-Tuning a Compact Instruction-Tuned Language Model for Retail E-Commerce Customer-Support Conversational AI*.

This work extends my earlier Deep Learning for NLP project, where I fine-tuned a Hugging Face transformer to build a domain chatbot. Here I focus on the **Retail & E-commerce** industry and ask a clear research question: *can a small, efficient model — flan-t5-small — be fine-tuned on retail support data to give accurate, on-topic answers while running on a single free-tier GPU?* This matters because e-commerce support is high-volume, repetitive, and expensive to staff 24/7, so a lightweight automated agent has real practical and academic relevance."

### PART 2 — Problem Understanding & Industry Context (≈2 min) `[SHOW: Industry Analysis slide]`
"E-commerce has exploded, and with it the volume of customer queries — order status, refunds, cancellations, shipping options, returns, payments. These are highly repetitive yet customers expect instant, 24/7, accurate answers. Human-only support doesn't scale and is costly.

Conversational AI is a natural fit because most of these intents are well-structured. The gap I'm addressing: most production-grade solutions rely on large, expensive models or proprietary data. For smaller retailers, that's not viable. So the real problem is whether a **compact, cheaply-trainable** model can handle these industry-specific intents well enough to be useful — that's the need my research targets."

### PART 3 — Literature Review & Research Gap (≈2 min) `[SHOW: Literature slide]`
"I reviewed work across three threads. First, the transformer and text-to-text foundations — Vaswani's attention architecture, the T5 text-to-text framework, and instruction tuning with Flan-T5, which is the model family I use. Second, prior chatbot and customer-service NLP literature, which shows fine-tuning on domain data improves relevance. Third, evaluation methods for dialogue systems using ROUGE and BLEU.

The gap I found: much of the literature uses large models or generic datasets, with little focus on **compact models fine-tuned specifically for retail e-commerce support under constrained compute.** My contribution is to fill exactly that gap — a reproducible, low-resource fine-tuning study on a retail-specific dataset."

### PART 4 — Methodology & Research Design (≈2.5 min) `[SHOW: pipeline/flow diagram, then notebook]`
"My research workflow has five stages. **One — data:** I used the Bitext retail e-commerce dataset from Hugging Face — synthetic, openly licensed, no personal data — with instruction–response pairs across intents like place_order, track_order, get_refund, and delivery_options. **Two — preprocessing:** I cleaned and de-duplicated the pairs, mapped each instruction to its response, kept the placeholder slots, and split the data 80/10/10 stratified by category. **Three — tokenization** with the flan-t5 tokenizer at fixed max lengths (128 input, 256 target). **Four — fine-tuning:** flan-t5-small via Hugging Face Seq2SeqTrainer on a Colab T4, in full precision (FP32, because FLAN-T5 is numerically unstable in FP16), fixed seed, up to 5 epochs with early stopping — well within the 25-epoch cap. **Five — evaluation:** ROUGE-1/2/L, BLEU and BERTScore on a held-out test set, plus per-category analysis and a base-vs-fine-tuned comparison.

The design choices are deliberate: a small model and a single GPU keep the study reproducible and aligned with my research question about low-resource feasibility, rather than just maximizing scores with a huge model."

### PART 5 — Results, Demonstration & Evidence (≈2 min) `[SHOW: loss curve, metrics table, THEN live Gradio demo]`
"Here are the results. The training and validation loss curves show the model converging cleanly — training loss fell from about 2.5 to 0.7 and validation loss from about 1.08 to 0.62, with validation tracking below training, so no overfitting. On the held-out test set, the fine-tuned model scored a ROUGE-L of **0.45** and a BLEU of **29.3**, compared to just **0.06** and **0** for the un-tuned base flan-t5-small — that's roughly an **eight-fold** jump in ROUGE-L, and BERTScore rose from 0.82 to 0.91. Looking per category, it performed strongest on **Account and Cart** queries and weakest on **Feedback**. The average response also grew from a useless 7 words to a full, structured 133-word answer.

Now the live demo. `[SHOW: type into Gradio]` I'll ask: *'How can I track my order?'* … *'How do I get a refund?'* … *'I want to cancel my order.'* As you can see, the bot returns fluent, on-topic, step-by-step responses to genuine retail support queries — that's the practical evidence behind the metrics."

### PART 6 — Discussion, Implications & Originality (≈2 min) `[SHOW: Discussion slide]`
"What does this mean? It shows a compact model can be adapted to handle the bulk of routine retail support intents at very low cost — attractive for small and mid-size retailers who can't run large models. Practically, it could deflect a large share of repetitive tickets and free human agents for complex cases.

On originality: this analysis, the dataset pivot to retail, the experiments, and the write-up are my own work. I used the prior project only as a methodological starting point, and re-built everything around the retail e-commerce domain — the data, the evaluation, and the conclusions are specific to this study."

### PART 7 — Limitations, Challenges & Ethical Considerations (≈1.5 min) `[SHOW: Limitations slide]`
"Limitations: flan-t5-small is small, so it can produce generic or imperfect answers on rarer or more nuanced queries. A clear example I observed is **out-of-domain behaviour** — when I asked it about the weather or to tell a joke, it still replied in confident customer-support style instead of recognising the question was out of scope. That's expected from a model fine-tuned narrowly on support data, and it's exactly why a production system needs intent-scoping and a human fallback. The dataset is also synthetic, so it may not capture the full messiness of real customer language, and automatic metrics like ROUGE and BLEU approximate quality but don't fully reflect human judgment. Ethically, because the data is synthetic there's no personal-data exposure, but in real deployment you'd need guardrails and care around making refund or policy commitments. To strengthen the work, I'd add human evaluation, real anonymized logs, out-of-scope detection, and a larger-model comparison."

### PART 8 — Conclusion & Future Research (≈1.5 min) `[SHOW: Conclusion slide]`
"To conclude: I successfully fine-tuned a compact instruction-tuned model for retail e-commerce customer support, and showed it clearly outperforms the un-tuned baseline on standard metrics while running on free-tier compute. The contribution is a practical, reproducible blueprint for low-resource domain chatbots.

Future directions: larger or comparative models, retrieval augmentation over a live product/FAQ knowledge base, human-in-the-loop evaluation, multilingual support, and a production deployment with proper safeguards. Thank you for watching."

---

> **After recording:** trim to 15–25 min, confirm audio is clear, upload to Drive (viewer access), and paste the link in the dashboard's *Video link* field.
