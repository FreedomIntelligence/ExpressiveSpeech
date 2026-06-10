# Supplementary Material for "Decoding the Ear: A Framework for Objectifying Expressiveness from Human Preference Through Efficient Alignment"

This repository contains the supplementary data, human annotation records, evaluation guidelines, and prompts supporting the experiments in our paper (Anonymous submission to Interspeech 2026).

## Directory Structure & Contents

### 1. prosody_prompt.txt
Contains the specific Chain-of-Thought (CoT) prompt used to instruct Gemini-2.5-Pro for Prosodic Richness Scoring (detailed in Section 2.2.2 of the paper).

### 2. Standardized protocols/
Contains the detailed guidelines and 5-point Mean Opinion Score (MOS) rating scales provided to our human expert annotators. 
- `expressiveness.md`: Guidelines for overall expressiveness.
- `prosody.md`: Guidelines for prosody evaluation.
- `Spontaneity.md`: Guidelines for spontaneity evaluation.

### 3. Experiment4.1/ (Validity: Alignment with Human Perception)
Contains the evaluation data used in Section 4.1 to validate the alignment between DeEAR scores and human perception.
- `emotion.jsonl`, `expressive.jsonl`, `prosody.jsonl`, `spontaneity.jsonl`: Each file contains the audio paths, the ground truth scores, the predicted objective scores from DeEAR, and the individual ratings from 10 human annotators for the respective dimensions.

### 4. Experiment4.2/ (Automated Benchmarking of SOTA Models)
Contains the data for the benchmarking of 7 SOTA Speech-to-Speech (S2S) models discussed in Section 4.2.
- `Experiment4.2_prompts.txt`: The 20 diverse conversational text prompts used as inputs for the SOTA models.
- `10annotators.jsonl`: Contains the detailed evaluation records for each generated audio, including the objective DeEAR scores (arousal, prosody, nature, expressive), the individual ratings from 10 human annotators, and the final averaged human MOS.

### 5. Experiment4.3/ (Evaluation-driven Data Curation)
Contains the evaluation results for the ablation study in Section 4.3, comparing our foundation model (Base) and the fine-tuned model (FT / exp).
- `objective.jsonl`: Contains the detailed DeEAR objective metrics (arousal, prosody, nature, expressive) scored for the synthesized audio samples.
- `subjective.jsonl`: Contains the blind A/B test results from 10 native speakers, recording their preferences ("base", "exp", or "none" for tie) for each test case.

## Data Format
All `.jsonl` (JSON Lines) files contain one JSON object per line, clearly mapping the audio samples to their respective metrics and human ratings.