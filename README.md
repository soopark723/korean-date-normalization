# Evaluating ReAct-Based LLM Reasoning for Korean Date and Schedule Processing

This repository contains the **datasets, prompts, agent implementations, and evaluation code** for the paper:

> **Evaluating ReAct-Based LLM Reasoning for Korean Date and Schedule Processing**

The project investigates whether **ReAct-style agentic reasoning**  
(*Thought → Action → Observation*) improves performance over a strong **few-shot Chain-of-Thought (CoT)** baseline for **Korean temporal reasoning**, evaluated across **accuracy, latency, and token cost**.

---

## Overview

Korean temporal expressions often combine:

- Relative references (e.g., *“다음 주 금요일”*)
- Contextual grounding
- Compound scheduling constraints  
  (weekday-only, holiday exclusion, minimum intervals)

We compare two reasoning paradigms:

- **CoT prompting** — single-pass reasoning, no tools  
- **ReAct agents** — iterative reasoning with tool invocation  

across three temporal reasoning tasks with increasing complexity.

---

## Tasks

We define three task settings to isolate different dimensions of temporal reasoning.

### **T1 — Date Normalization (Phrase-level)**
- **Input:** Standalone Korean temporal expressions  
  (e.g., *“다음 달 마지막 날 다음 평일”*)
- **Output:** Single normalized date (`YYYY-MM-DD`)
- **Focus:** Temporal arithmetic and boundary reasoning

### **T2 — Date Normalization with Sentences**
- **Input:** Full Korean sentences containing temporal expressions  
  (e.g., *“다음 주 금요일 날짜 알려줘”*)
- **Output:** Single normalized date (`YYYY-MM-DD`)
- **Focus:** Temporal span detection + normalization

### **T3 — Constraint-based Schedule Generation**
- **Input:** Korean scheduling instructions with multiple constraints  
  (weekday-only, interval rules, holiday exclusion, count limits)
- **Output:** Ordered list of dates (`YYYY-MM-DD`)
- **Focus:** Multi-event planning, constraint propagation, long-horizon reasoning

---

## Methodology

### **Baseline: Chain-of-Thought (CoT)**
- Few-shot prompting with explicit reasoning instructions
- Single forward generation
- No external tools or validation
- Same prompt structure across tasks

### **ReAct Agent**
- Iterative **Thought → Action → Observation** loop
- Tool primitives:
  - `calculator` — date arithmetic (offsets, intervals, weekdays)
  - `calendar_db` — holidays, solar terms, anniversaries
  - `search` — external event lookup
- Single-step reasoning for **T1 / T2**
- Multi-step planning loop for **T3** (max 10 turns)

---

## Models Evaluated

| Model | Reasoning |
|------|----------|
| GPT-4.1-mini | CoT / ReAct |
| SOLAR-pro2 (Upstage) | CoT / ReAct |

All experiments use **default decoding settings** from each API to ensure fair comparison.

---

## Evaluation Metrics

### **Accuracy**
- **T1 / T2:** Exact-match accuracy on normalized dates
- **T3:** Full constraint satisfaction (exact list + order)

### **Efficiency**
- **Latency:** Wall-clock time per query
- **Token usage:** Prompt + completion tokens

Failures include:
- Off-by-one date errors
- Constraint violations
- ReAct turn-limit collapse

---

## Key Findings

- **ReAct significantly improves GPT performance on T1 and T2**  
  (+4–6% accuracy, *p* < .05), but at very high token cost
- **Solar-ReAct provides the best cost–performance balance** for T1 and T2
- **All models fail to scale on T3**  
  (≤22% accuracy, no statistical difference between CoT and ReAct)
- **Tool-driven instability dominates ReAct failures** in multi-event reasoning
- **Ablation replacing hand-coded tools with LLM-driven reasoning improves T3 accuracy to 31.4%**, highlighting **orchestration—not tool availability—as the key factor**

---

## Limitations

- Evaluation relies primarily on accuracy and cost metrics
- Reasoning trace quality and failure propagation are not fully analyzed
- T1/T2 are simpler than real-world multi-event scheduling
- Tool interfaces are brittle under long-horizon planning

---

## Future Work

- Multi-event temporal pipelines with explicit event segmentation
- Adaptive tool-use policies and recovery mechanisms
- New metrics for reasoning stability
- Broader coverage of Korean temporal phenomena
