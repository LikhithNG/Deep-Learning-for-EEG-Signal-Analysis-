"""
Modified preprocessing script for seizure classification
Works with custom file structure without requiring file_markers_classification
"""
import os
import sys
import argparse
import numpy as np
import h5py
from tqdm import tqdm
import glob
import random

# Add the repository root to path for importing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import constants and utility functions
try:
    from constants import INCLUDED_CHANNELS, FREQUENCY
except ImportError:
    # Define constants if not available
    INCLUDED_CHANNELS = [
        'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
        'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FZ-CZ', 'CZ-PZ'
    ]
    FREQUENCY = 200

# Define seizure types mapping (for classification)
SEIZURE_TYPE_MAPPING = {
    'FNSZ': 0,  # Focal Non-specific
    'GNSZ': 1,  # Generalized Non-specific
    'ABSZ': 1,  # Absence (mapped to Generalized)
    'CPSZ': 0,  # Complex Partial (mapped to Focal)
    'SPSZ': 0,  # Simple Partial (mapped to Focal)
    'TCSZ': 3,  # Tonic-Clonic
    'TNSZ': 2,  # Tonic
    'MYSZ': 1,  # Myoclonic (mapped to Generalized)
}

def find_all_edf_files(data_dir):
    """Find all EDF files in the given directory recursively"""
    edf_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.edf'):
                edf_files.append(os.path.join(root, file))
    return edf_files

def find_all_annotation_files(data_dir):
    """Find all seizure annotation files (csv_bi) in the given directory recursively"""
    annotation_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv_bi'):
                annotation_files.append(os.path.join(root, file))
    return annotation_files

def read_annotation_file(file_path):
    """Read seizure annotations from .csv_bi file and determine seizure type"""
    seizures = []
    seizure_type = 'FNSZ'  # Default to focal if not specified
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            # Look for seizure type in the header comments
            for line in lines:
                if line.startswith('#'):
                    if 'seizure_type' in line.lower() or 'type' in line.lower():
                        parts = line.strip().split(':')
                        if len(parts) > 1:
                            possible_type = parts[1].strip().upper()
                            if possible_type in SEIZURE_TYPE_MAPPING:
                                seizure_type = possible_type
            
            # Extract seizure intervals
            for line in lines:
                if line.startswith('#') or line.startswith('channel'):
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 5 and parts[3] == 'seiz':
                    start_time = float(parts[1])
                    stop_time = float(parts[2])
                    seizures.append((start_time, stop_time))
    except Exception as e:
        print(f"Error reading annotation file {file_path}: {str(e)}")
    
    return seizures, seizure_type

def extract_windows(signal, window_size, step_size, is_fft=True):
    """Extract windows from signal with optional FFT transform"""
    num_channels, num_points = signal.shape
    windows = []
    
    for start in range(0, num_points - window_size + 1, step_size):
        end = start + window_size
        window_data = signal[:, start:end]
        
        if is_fft:
            # Apply Hamming window
            hamming = np.hamming(window_size)
            windowed_data = window_data * hamming[None, :]
            
            # Compute FFT
            fft_result = np.abs(np.fft.rfft(windowed_data, axis=1))
            
            # Keep only first 25 Hz
            freq_bins = int(25 * window_size / FREQUENCY)
            fft_result = fft_result[:, :freq_bins]
            
            windows.append(fft_result)
        else:
            windows.append(window_data)
    
    # Stack all windows
    if len(windows) > 0:
        return np.stack(windows, axis=1)
    else:
        return np.array([])

def process_eeg_file(h5_path, seizures, seizure_type, clip_len, time_step_size, is_fft):
    """Process an EEG file and extract clips"""
    try:
        # Load resampled EEG data
        with h5py.File(h5_path, 'r') as hf:
            signal = hf['resampled_signal'][:]
            fs = hf['resample_freq'][()]
            
        # Convert clip_len and time_step_size from seconds to samples
        clip_samples = int(clip_len * fs)
        step_samples = int(time_step_size * fs)
        
        # For each seizure interval, extract clips
        clips = []
        for start_time, stop_time in seizures:
            # Convert seizure times to sample indices
            start_idx = max(0, int(start_time * fs))
            stop_idx = min(int(stop_time * fs), signal.shape[1])
            
            # Ensure at least one clip_len worth of data
            if stop_idx - start_idx < clip_samples:
                # If seizure is shorter than clip_len, pad by centering
                pad_needed = clip_samples - (stop_idx - start_idx)
                start_idx = max(0, start_idx - pad_needed // 2)
                stop_idx = min(signal.shape[1], stop_idx + pad_needed // 2)
            
            # Extract seizure segment
            seizure_segment = signal[:, start_idx:stop_idx]
            
            # Extract overlapping windows if seizure is long enough
            if seizure_segment.shape[1] >= clip_samples:
                for window_start in range(0, seizure_segment.shape[1] - clip_samples + 1, 
                                         max(1, int(step_samples))):
                    window_end = window_start + clip_samples
                    clip_data = seizure_segment[:, window_start:window_end]
                    
                    # Process clip (extract features if needed)
                    if is_fft:
                        # Extract FFT features
                        window_size = int(fs)  # 1-second windows
                        window_step = int(fs * 0.5)  # 50% overlap
                        processed_clip = extract_windows(clip_data, window_size, window_step, is_fft=True)
                    else:
                        processed_clip = clip_data
                    
                    clips.append((processed_clip, SEIZURE_TYPE_MAPPING[seizure_type]))
        
        return clips
    except Exception as e:
        print(f"Error processing {h5_path}: {str(e)}")
        return []

def preprocess_for_classification(resampled_dir, raw_data_dir, output_dir, clip_len, time_step_size, is_fft):
    """Preprocess EEG data for seizure classification"""
    # Create output directory
    output_dir_full = os.path.join(output_dir, f'clipLen{clip_len}_timeStepSize{time_step_size}')
    os.makedirs(output_dir_full, exist_ok=True)
    
    # Find annotation files
    annotation_files = find_all_annotation_files(raw_data_dir)
    print(f"Found {len(annotation_files)} annotation files.")
    
    # Count by seizure type
    seizure_counts = {k: 0 for k in SEIZURE_TYPE_MAPPING.values()}
    processed_count = 0
    skipped_count = 0
    
    # Process each annotation file
    for ann_idx, ann_file in enumerate(tqdm(annotation_files, desc="Processing files")):
        # Get base name
        base_name = os.path.basename(ann_file).split('.csv_bi')[0]
        
        # Find corresponding h5 file
        h5_path = os.path.join(resampled_dir, f"{base_name}.h5")
        if not os.path.exists(h5_path):
            print(f"No resampled file found for {base_name}, skipping...")
            skipped_count += 1
            continue
        
        # Read annotations
        seizures, seizure_type = read_annotation_file(ann_file)
        if not seizures or seizure_type not in SEIZURE_TYPE_MAPPING:
            print(f"No valid seizures found in {ann_file}, skipping...")
            skipped_count += 1
            continue
        
        # Process the file
        clips = process_eeg_file(h5_path, seizures, seizure_type, clip_len, time_step_size, is_fft)
        
        # Save clips
        for clip_idx, (clip_data, class_label) in enumerate(clips):
            # Skip if empty
            if len(clip_data) == 0:
                continue
                
            # Save as NPZ file
            save_path = os.path.join(output_dir_full, f"{base_name}_seizure_{clip_idx}.npz")
            np.savez_compressed(
                save_path,
                data=clip_data,
                label=class_label,
                file_name=base_name,
                seizure_type=seizure_type
            )
            
            # Update counts
            seizure_counts[class_label] += 1
            processed_count += 1
    
    print("\nPreprocessing complete!")
    print(f"Processed {processed_count} clips across {len(annotation_files) - skipped_count} files.")
    print(f"Skipped {skipped_count} files.")
    print("\nClass distribution:")
    
    for class_idx, count in seizure_counts.items():
        class_name = next((k for k, v in SEIZURE_TYPE_MAPPING.items() if v == class_idx), "Unknown")
        print(f"  Class {class_idx} ({class_name}): {count} clips")
    
    # Save class distribution to file
    with open(os.path.join(output_dir_full, "class_distribution.txt"), 'w') as f:
        f.write("Class Distribution:\n")
        for class_idx, count in seizure_counts.items():
            class_name = next((k for k, v in SEIZURE_TYPE_MAPPING.items() if v == class_idx), "Unknown")
            f.write(f"Class {class_idx} ({class_name}): {count} clips\n")

def main():
    parser = argparse.ArgumentParser("Modified Preprocessing for Seizure Classification")
    
    parser.add_argument("--resampled_dir", type=str, required=True,
                       help="Directory containing resampled h5 files")
    parser.add_argument("--raw_data_dir", type=str, required=True,
                       help="Directory containing raw EDF and annotation files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for preprocessed data")
    parser.add_argument("--clip_len", type=int, default=12,
                       help="Clip length in seconds (default: 12)")
    parser.add_argument("--time_step_size", type=int, default=1,
                       help="Time step size in seconds")
    parser.add_argument("--is_fft", action="store_true", default=False,
                       help="Whether to perform FFT. If not set, use raw signals.")
    
    args = parser.parse_args()
    
    # Print settings
    print("\nPreprocessing Settings:")
    print(f"  Resampled directory: {args.resampled_dir}")
    print(f"  Raw data directory: {args.raw_data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Clip length: {args.clip_len} seconds")
    print(f"  Time step size: {args.time_step_size} seconds")
    print(f"  FFT enabled: {args.is_fft}")
    print()
    
    # Run preprocessing
    preprocess_for_classification(
        args.resampled_dir,
        args.raw_data_dir,
        args.output_dir,
        args.clip_len,
        args.time_step_size,
        args.is_fft
    )

if __name__ == "__main__":
    main()