#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import logging
import argparse
from pathlib import Path
import warnings

# --- Core Dependencies ---
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import xgboost as xgb
from pydub import AudioSegment
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2PreTrainedModel


# ==================== 1. Custom Model Definition ====================
# The following classes define the custom architecture of the deep learning model.

class RegressionHead(nn.Module):
    """A simple regression head for predicting a single score from pooled features."""
    def __init__(self, input_dim, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ParallelExpressivenessModel(Wav2Vec2PreTrainedModel):
    """
    A multi-task model built on Wav2Vec2 for predicting arousal, prosody, and
    naturalness scores in parallel.
    """
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        # Create a separate regression head for each perceptual dimension
        self.head_arousal = RegressionHead(config.hidden_size)
        self.head_prosody = RegressionHead(config.hidden_size)
        self.head_nature = RegressionHead(config.hidden_size)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        # Use mean pooling over the last hidden state
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        # Predict scores in parallel
        s_arousal_pred = self.head_arousal(pooled_output)
        s_prosody_pred = self.head_prosody(pooled_output)
        s_nature_pred = self.head_nature(pooled_output)

        return {
            "s_arousal": s_arousal_pred.squeeze(-1),
            "s_prosody": s_prosody_pred.squeeze(-1),
            "s_nature": s_nature_pred.squeeze(-1),
        }

# ==================== 2. Data Handling ====================

class InferenceDataset(Dataset):
    """Dataset for loading and preprocessing audio files for inference."""
    def __init__(self, audio_files, feature_extractor, max_duration_s=15.0):
        self.audio_files = audio_files
        self.feature_extractor = feature_extractor
        self.target_sample_rate = feature_extractor.sampling_rate
        self.max_samples = int(max_duration_s * self.target_sample_rate)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        try:
            # Try loading with torchaudio first (requires ffmpeg backend)
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
            # Fallback to pydub if torchaudio fails
            except (RuntimeError, OSError):
                audio = AudioSegment.from_file(audio_path)
                samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
                waveform = torch.from_numpy(samples).unsqueeze(0)
                if audio.channels > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                sample_rate = audio.frame_rate

            # Resample if necessary
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Truncate long audio
            if waveform.shape[1] > self.max_samples:
                waveform = waveform[:, :self.max_samples]

            return {"audio_path": audio_path, "input_values": waveform.squeeze(0)}
        except Exception as e:
            logging.warning(f"Failed to process {audio_path}: {e}. Skipping file.")
            return {"audio_path": audio_path, "input_values": None}


def collate_fn_inference(batch):
    """Custom collate function to handle failed audio loads and pad batches."""
    valid_items = [item for item in batch if item["input_values"] is not None]
    if not valid_items:
        return None, None
    audio_paths = [item["audio_path"] for item in valid_items]
    input_features = [{"input_values": item["input_values"]} for item in valid_items]
    # This is a bit of a trick to use the feature_extractor's padding logic
    # We need to find the feature_extractor instance.
    # A cleaner way would be to pass it as an argument.
    # For now, this dynamic lookup works if a global feature_extractor is set.
    padded_batch = feature_extractor_global.pad(input_features, return_tensors="pt", padding=True)
    return audio_paths, padded_batch

# ==================== 3. Inference Pipeline Stages ====================

def run_stage1_deep_learning(audio_files, model, feature_extractor, device, batch_size, num_workers):
    """Runs Stage 1: Extracts 3 perceptual scores using the deep learning model."""
    logging.info("--- Stage 1/2: Running Deep Learning Model Inference ---")
    
    # Make feature_extractor accessible to the collate function
    global feature_extractor_global
    feature_extractor_global = feature_extractor
    
    dataset = InferenceDataset(audio_files, feature_extractor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_inference,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )

    all_results = []
    model.eval()
    with torch.no_grad():
        for audio_paths, batch in tqdm(dataloader, desc="Stage 1: Extracting 3 scores"):
            if batch is None:
                continue

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # Enable AMP for performance
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                predictions = model(input_values=batch['input_values'], attention_mask=batch['attention_mask'])

            preds_cpu = {key: val.cpu().numpy() for key, val in predictions.items()}

            for i in range(len(audio_paths)):
                result = {
                    "audio_path": audio_paths[i],
                    "score_arousal": float(preds_cpu["s_arousal"][i]),
                    "score_prosody": float(preds_cpu["s_prosody"][i]),
                    "score_nature": float(preds_cpu["s_nature"][i]),
                }
                all_results.append(result)

    logging.info(f"--- Stage 1/2: Finished. Extracted scores for {len(all_results)} audio files. ---")
    return all_results


def run_stage2_xgboost(intermediate_results, xgb_model_path):
    """Runs Stage 2: Predicts the final expressiveness score using XGBoost."""
    logging.info("--- Stage 2/2: Running XGBoost Model Inference ---")

    if not intermediate_results:
        logging.warning("No valid results from Stage 1. Skipping XGBoost inference.")
        return []

    try:
        xgb_model = xgb.Booster()
        xgb_model.load_model(xgb_model_path)
        logging.info(f"XGBoost model loaded successfully from: {xgb_model_path}")
    except xgb.core.XGBoostError as e:
        logging.error(f"Failed to load XGBoost model! Please check the path. Error: {e}")
        raise

    df = pd.DataFrame(intermediate_results)
    # Ensure feature order matches the model's training order
    feature_columns = ['score_arousal', 'score_prosody', 'score_nature']
    X_test = df[feature_columns]
    dtest = xgb.DMatrix(X_test)
    predictions = xgb_model.predict(dtest)

    df['score_expressive'] = predictions

    logging.info("--- Stage 2/2: Finished. Final expressiveness scores generated. ---")
    return df.to_dict('records')

# ==================== 4. Main Execution Logic ====================

def main(args):
    """Main function to set up and run the entire inference pipeline."""
    start_time = time.time()

    # --- Setup ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    warnings.filterwarnings('ignore', category=UserWarning)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        logging.warning("No CUDA-enabled GPU found. Inference will run on CPU and may be slow.")

    try:
        torchaudio.set_audio_backend("ffmpeg")
        logging.info("Torchaudio backend set to 'ffmpeg'.")
    except RuntimeError:
        logging.warning("Could not set torchaudio backend to 'ffmpeg'. Ensure ffmpeg is installed for best compatibility.")

    # --- Model Loading ---
    dl_model_path = os.path.join(args.model_dir, "DeEAR_Base")
    xgb_model_path = os.path.join(args.model_dir, "XGBoost", "xgboost_model.json")

    for path in [dl_model_path, xgb_model_path]:
        if not os.path.exists(path):
            logging.error(f"Required model file or directory not found: {path}")
            logging.error("Please ensure you have downloaded the models and placed them in the correct directory.")
            return

    logging.info(f"Loading deep learning model from: {dl_model_path}")
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(dl_model_path)
        model = ParallelExpressivenessModel.from_pretrained(dl_model_path)
        model.to(device)
    except Exception as e:
        logging.error(f"Failed to load deep learning model. Error: {e}", exc_info=True)
        return
    logging.info("Deep learning model loaded successfully.")

    # --- File Discovery ---
    logging.info(f"Scanning for audio files in: {args.input_path}")
    audio_files = []
    if os.path.isdir(args.input_path):
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
        for p in Path(args.input_path).rglob('*'):
            if p.suffix.lower() in audio_extensions:
                audio_files.append(str(p.resolve()))
    elif os.path.isfile(args.input_path):
        with open(args.input_path, 'r') as f:
            audio_files = [line.strip() for line in f if line.strip()]
    else:
        logging.error(f"Input path is not a valid directory or file: {args.input_path}")
        return

    if not audio_files:
        logging.warning("No audio files found. Exiting.")
        return
    logging.info(f"Found {len(audio_files)} audio files to process.")

    # --- Run Pipeline ---
    num_workers = args.num_workers if args.num_workers is not None else os.cpu_count() // 2
    intermediate_results = run_stage1_deep_learning(audio_files, model, feature_extractor, device, args.batch_size, num_workers)
    final_results = run_stage2_xgboost(intermediate_results, xgb_model_path)

    # --- Save Results ---
    if final_results:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(args.output_file, 'w', encoding='utf-8') as f_out:
            for result in final_results:
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
        logging.info(f"Results successfully saved to: {args.output_file}")
    else:
        logging.warning("Pipeline produced no results.")


    # --- Final Report ---
    end_time = time.time()
    total_time = end_time - start_time
    processed_count = len(final_results)
    
    logging.info("=" * 60)
    logging.info("🚀 Pipeline Finished!")
    logging.info(f"  - Total Files Found: {len(audio_files)}")
    logging.info(f"  - Files Processed Successfully: {processed_count}")
    logging.info(f"  - Files Skipped/Failed: {len(audio_files) - processed_count}")
    logging.info(f"  - Total Time: {total_time:.2f} seconds")
    if processed_count > 0:
        logging.info(f"  - Average Speed: {processed_count / total_time:.2f} files/sec")
    logging.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expressive Speech Scoring Pipeline.")
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./model",
        help="Path to the root directory containing the exported models."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to a directory of audio files or a text file listing audio paths."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./results.jsonl",
        help="Path to the output .jsonl file where results will be saved."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference. Adjust based on your GPU memory."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes for data loading. Defaults to half of CPU cores."
    )

    cli_args = parser.parse_args()
    
    try:
        main(cli_args)
    except KeyboardInterrupt:
        logging.info("\nProgram interrupted by user. Exiting.")
    except Exception as e:
        logging.error(f"An unhandled error occurred: {e}", exc_info=True)