import os
import numpy as np
from tqdm import tqdm
import random
import argparse

def load_sample_data(file_path):
    """Load an existing seizure data file as a template"""
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"Successfully loaded template file: {file_path}")
        print(f"Data shape: {data['data'].shape}")
        print(f"Original label: {data['label']}")
        return data
    except Exception as e:
        print(f"Error loading template file: {e}")
        return None

def apply_transformations(original_data, transformation_strength=0.5):
    """Apply various transformations to create a unique sample"""
    data = original_data.copy()
    
    # 1. Add random noise
    noise_level = np.random.uniform(0.1, transformation_strength * 0.5)
    noise = np.random.normal(0, noise_level, data.shape)
    data = data + noise
    
    # 2. Time shifting/warping (simulate different seizure temporal characteristics)
    if np.random.random() < 0.7:
        time_axis = 1 # Assuming time is the second dimension
        num_time_points = data.shape[time_axis]
        
        # Create warping grid
        warp_factor = np.random.uniform(0.1, transformation_strength)
        warp = np.random.normal(0, warp_factor, size=num_time_points)
        warp = np.cumsum(warp)  # Cumulative sum to create smooth warping
        warp = warp - warp.mean()  # Center around zero
        warp_scaled = warp / (warp.max() - warp.min() + 1e-8) * num_time_points / 4
        
        # Apply time warping
        orig_indices = np.arange(num_time_points)
        warped_indices = np.clip(orig_indices + warp_scaled, 0, num_time_points - 1)
        
        # Interpolate
        warped_data = np.zeros_like(data)
        for c in range(data.shape[0]):  # For each channel
            for t in range(num_time_points):
                idx = int(warped_indices[t])
                warped_data[c, t] = data[c, idx]
        data = warped_data
    
    # 3. Amplitude scaling (simulate different seizure intensities)
    if np.random.random() < 0.8:
        scale_factor = np.random.uniform(0.7, 1.0 + transformation_strength)
        data = data * scale_factor
    
    # 4. Channel-specific changes (simulate different spatial characteristics)
    if np.random.random() < 0.6:
        for c in range(data.shape[0]):
            if np.random.random() < 0.3:  # Only modify some channels
                channel_scale = np.random.uniform(0.5, 1.5)
                data[c, :] = data[c, :] * channel_scale
    
    # 5. Frequency domain transformations (if 2D data with time and frequency)
    if len(data.shape) == 3 and np.random.random() < 0.5:
        freq_axis = 2  # Assuming frequency is the third dimension
        for c in range(data.shape[0]):
            for t in range(data.shape[1]):
                # Apply frequency modulation
                freq_mod = np.random.uniform(0.8, 1.2, size=data.shape[freq_axis])
                data[c, t, :] = data[c, t, :] * freq_mod
    
    # 6. Add occasional spike artifacts
    if np.random.random() < 0.4:
        num_spikes = np.random.randint(1, 5)
        for _ in range(num_spikes):
            channel = np.random.randint(0, data.shape[0])
            time_point = np.random.randint(0, data.shape[1])
            spike_amp = np.random.uniform(1.5, 3.0)
            spike_width = np.random.randint(1, 5)
            
            # Create spike
            for t in range(max(0, time_point - spike_width), min(data.shape[1], time_point + spike_width + 1)):
                dist = abs(t - time_point)
                factor = spike_amp * (1 - dist/spike_width) if dist < spike_width else 0
                data[channel, t] = data[channel, t] * (1 + factor)
                
    return data

def generate_synthetic_data(template_data, class_distribution, output_dir, transformation_strength=0.5):
    """Generate synthetic data for multiple classes"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the original data for transformation
    original_data = template_data['data']
    
    total_generated = 0
    
    # Generate data for each class
    for class_idx, num_samples in enumerate(class_distribution):
        print(f"Generating {num_samples} samples for Class {class_idx}...")
        
        for i in tqdm(range(num_samples)):
            # Apply random transformations to create a unique sample
            synthetic_data = apply_transformations(original_data, transformation_strength)
            
            # Create file name
            file_name = f"synthetic_class{class_idx}_sample{i:04d}.npz"
            file_path = os.path.join(output_dir, file_name)
            
            # Determine seizure type (you can customize this based on class)
            seizure_types = ["CF", "GN", "AB", "CT"]  # Example seizure types
            if class_idx < len(seizure_types):
                seizure_type = seizure_types[class_idx]
            else:
                seizure_type = "OT"  # Other
                
            # Save the synthetic data with appropriate metadata
            np.savez(file_path, 
                    data=synthetic_data, 
                    label=class_idx,
                    file_name=file_name,
                    seizure_type=seizure_type)
            
            total_generated += 1
    
    print(f"Successfully generated {total_generated} synthetic samples across {len(class_distribution)} classes")
    print(f"Output directory: {output_dir}")
    
    # Create a visualization of the first few samples from each class for verification
    visualize_samples(output_dir, class_distribution)

def visualize_samples(data_dir, class_distribution):
    """Create a visualization of sample data from each class"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 10))
    
    for class_idx, num_samples in enumerate(class_distribution):
        if num_samples == 0:
            continue
            
        # Find files for this class
        class_files = [f for f in os.listdir(data_dir) if f.startswith(f"synthetic_class{class_idx}_")]
        
        if not class_files:
            continue
            
        # Pick a random sample to visualize
        sample_file = os.path.join(data_dir, random.choice(class_files))
        sample_data = np.load(sample_file)
        
        # Plot the first few channels
        data = sample_data['data']
        max_channels_to_plot = min(4, data.shape[0])
        
        for ch_idx in range(max_channels_to_plot):
            plt.subplot(len(class_distribution), max_channels_to_plot, 
                        class_idx * max_channels_to_plot + ch_idx + 1)
            
            # If 2D data (channel, time)
            if len(data.shape) == 2:
                plt.plot(data[ch_idx, :])
                
            # If 3D data (channel, time, freq)
            elif len(data.shape) == 3:
                plt.imshow(data[ch_idx, :, :], aspect='auto')
                plt.colorbar()
            
            plt.title(f"Class {class_idx}, Channel {ch_idx}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "sample_visualization.png"))
    plt.close()
    print(f"Sample visualization saved to {os.path.join(data_dir, 'sample_visualization.png')}")

def generate_class_distribution_file(output_dir, class_distribution):
    """Creates a summary file with class distribution information"""
    distribution_data = {
        'class_counts': {i: count for i, count in enumerate(class_distribution)},
        'total_samples': sum(class_distribution),
        'class_percentages': {
            i: (count / sum(class_distribution) * 100) 
            for i, count in enumerate(class_distribution)
        }
    }
    
    file_path = os.path.join(output_dir, "class_distribution.npz")
    np.savez(file_path, **distribution_data)
    print(f"Class distribution summary saved to {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic EEG seizure data with multiple classes")
    parser.add_argument("--template", type=str, required=True, help="Path to template .npz seizure file")
    parser.add_argument("--output_dir", type=str, default="synthetic_eeg_data", help="Output directory for synthetic data")
    parser.add_argument("--class_distribution", type=str, default="500,400,300,200,100", 
                       help="Comma-separated list of samples to generate per class")
    parser.add_argument("--transformation_strength", type=float, default=0.6,
                       help="Strength of transformations applied (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Parse class distribution
    class_distribution = [int(x) for x in args.class_distribution.split(',')]
    
    # Load template data
    template_data = load_sample_data(args.template)
    if template_data is None:
        print("Error: Could not load template data. Exiting.")
        return
    
    # Generate synthetic data
    generate_synthetic_data(
        template_data=template_data,
        class_distribution=class_distribution,
        output_dir=args.output_dir,
        transformation_strength=args.transformation_strength
    )
    
    # Generate class distribution summary
    generate_class_distribution_file(args.output_dir, class_distribution)

if __name__ == "__main__":
    main()