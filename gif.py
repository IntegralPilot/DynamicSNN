"""
SNN Simulation Animated Dashboard Generator

This script loads data from 'simulation_log.csv' and generates an animated GIF
dashboard ('simulation_dashboard.gif') visualizing the key dynamic aspects 
of the learning process.

The animation provides a much more intuitive understanding of how synaptic weights,
neuron dynamics, and network structure evolve over time.

Required new libraries:
pip install imageio imageio[ffmpeg] tqdm
(You should already have pandas, matplotlib, numpy)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
import io
from tqdm import tqdm

# --- Configuration ---
LOG_FILE = 'simulation_log.csv'
OUTPUT_GIF_FILE = 'simulation_dashboard.gif'

# Animation Settings
FRAME_INTERVAL_MS = 250  # Create a frame every 250ms of simulation time
GIF_FPS = 15             # Frames per second for the final GIF
DPI = 120                # Resolution of the output GIF
VM_WINDOW_S = 5          # Sliding window size for Vm plot, in seconds

# --- (The load_data_and_patterns and parse_and_classify_synapses functions are the same) ---

def load_data_and_patterns(filename):
    """Loads simulation data and dynamically parses pattern mappings from the CSV header."""
    print(f"Loading data from '{filename}'...")
    if not os.path.exists(filename):
        print(f"Error: Log file '{filename}' not found.")
        print("Please run the C++ simulation first.")
        return None, None, None

    patterns = {}
    print("Parsing pattern mappings from file header...")
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('# PATTERN_MAPPING'):
                    parts = line.strip('# \n').split(',')
                    pattern_name = parts[1]
                    neuron_ids = {int(id_str) for id_str in parts[2:]}
                    patterns[pattern_name] = neuron_ids
                    print(f"  - Found mapping for '{pattern_name}' with {len(neuron_ids)} neurons.")
                elif not line.startswith('#'):
                    break
        
        if not patterns:
            print("Warning: No pattern mappings found. Plots may be incorrect.")

        df = pd.read_csv(filename, comment='#')
        df['time_ms'] = pd.to_numeric(df['time_ms'])
        df['value1'] = pd.to_numeric(df['value1'], errors='coerce')
        df['value2'] = pd.to_numeric(df['value2'], errors='coerce')
        
        print("Data loaded successfully.")
        return df, patterns

    except Exception as e:
        print(f"An error occurred while loading or parsing the file: {e}")
        return None, None, None

def parse_and_classify_synapses(df, patterns):
    """Filters for synapse data and classifies them into A, B, or Distractor."""
    syn_df = df[df['event_type'] == 'synapse'].copy()
    if syn_df.empty:
        print("Warning: No 'synapse' event data found.")
        return syn_df

    syn_df['pre_id'] = syn_df['name'].str.extract(r'Sensory_(\d+)').astype(int)
    pattern_a_ids = patterns.get('PatternA', set())
    pattern_b_ids = patterns.get('PatternB', set())

    def classify_synapse(pre_id):
        if pre_id in pattern_a_ids: return 'Pattern A'
        elif pre_id in pattern_b_ids: return 'Pattern B'
        else: return 'Distractor'
            
    syn_df['type'] = syn_df['pre_id'].apply(classify_synapse)
    syn_df.rename(columns={'value1': 'weight', 'value2': 'eligibility'}, inplace=True)
    return syn_df

# --- New/Modified Plotting Functions for Animation ---

def plot_weight_evolution_animated(ax, full_synapse_df, current_time_ms):
    """Plots average weight evolution up to the current time."""
    ax.set_title('Synaptic Weight Evolution', fontsize=12, fontweight='bold')
    # Filter data up to the current frame's time
    data_to_plot = full_synapse_df[full_synapse_df['time_ms'] <= current_time_ms]
    if data_to_plot.empty: return

    avg_weights = data_to_plot.groupby(['time_ms', 'type'])['weight'].mean().unstack()
    
    if 'Pattern A' in avg_weights.columns:
        ax.plot(avg_weights.index / 1000, avg_weights['Pattern A'], label='Pattern A', color='royalblue', lw=2)
    if 'Pattern B' in avg_weights.columns:
        ax.plot(avg_weights.index / 1000, avg_weights['Pattern B'], label='Pattern B', color='darkorange', lw=2)
    
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_ylabel('Avg. Weight')
    ax.set_xlabel('Time (s)')

def plot_weight_distribution_animated(ax, full_synapse_df, current_time_ms):
    """Plots a histogram of weights at the current time."""
    ax.set_title('Live Weight Distribution', fontsize=12, fontweight='bold')
    
    # Find the single closest time point in the data to our current frame time
    time_point_data = full_synapse_df[full_synapse_df['time_ms'] <= current_time_ms]
    if time_point_data.empty: return
    
    closest_time = time_point_data['time_ms'].max()
    current_weights = time_point_data[time_point_data['time_ms'] == closest_time]
    
    weights_a = current_weights[current_weights['type'] == 'Pattern A']['weight']
    weights_b = current_weights[current_weights['type'] == 'Pattern B']['weight']
    
    bins = np.linspace(0, 20, 21)
    ax.hist(weights_a, bins=bins, alpha=0.7, label='Pattern A', color='royalblue')
    ax.hist(weights_b, bins=bins, alpha=0.7, label='Pattern B', color='darkorange')

    ax.set_ylabel('Synapse Count')
    ax.set_xlabel('Synaptic Weight')
    ax.set_xlim(0, 20)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_decision_neuron_dynamics_animated(ax, decision_df, current_time_ms):
    """Plots Vm and threshold in a sliding window."""
    ax.set_title('Decision Neuron Dynamics (Live)', fontsize=12, fontweight='bold')
    
    window_start_ms = max(0, current_time_ms - (VM_WINDOW_S * 1000))
    window_df = decision_df[
        (decision_df['time_ms'] >= window_start_ms) & 
        (decision_df['time_ms'] <= current_time_ms)
    ]
    if window_df.empty: return
    
    ax.plot(window_df['time_ms'] / 1000, window_df['value1'], label='Vm', color='seagreen', alpha=0.8)
    ax.plot(window_df['time_ms'] / 1000, window_df['value2'], label='Threshold', color='firebrick', lw=2, ls='--')

    ax.set_ylabel('Voltage (mV)')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(window_start_ms / 1000, current_time_ms / 1000)

def plot_synapse_count_evolution_animated(ax, count_df, current_time_ms):
    """Plots synapse count evolution up to the current time."""
    ax.set_title('Structural Plasticity', fontsize=12, fontweight='bold')
    data_to_plot = count_df[count_df['time_ms'] <= current_time_ms]
    if data_to_plot.empty: return

    count_pivot = data_to_plot.pivot(index='time_ms', columns='name', values='value1')

    if 'Pattern A' in count_pivot.columns:
        ax.plot(count_pivot.index / 1000, count_pivot['Pattern A'], label='Pattern A', color='royalblue', lw=2)
    if 'Pattern B' in count_pivot.columns:
        ax.plot(count_pivot.index / 1000, count_pivot['Pattern B'], label='Pattern B', color='darkorange', lw=2)

    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Synapse Count')
    ax.set_xlabel('Time (s)')

# --- Main Animation Function ---

def main():
    """Main function to load data and generate the animated GIF."""
    df, patterns = load_data_and_patterns(LOG_FILE)
    if df is None:
        print("\nExiting. Please generate the log file first.")
        return

    # Prepare data subsets once to speed up the loop
    print("Preparing data for animation...")
    synapse_df = parse_and_classify_synapses(df, patterns)
    decision_df = df[(df['event_type'] == 'vm') & (df['name'] == 'Decision')].copy()
    count_df = df[df['event_type'] == 'synapse_count'].copy()

    # Determine animation timeline
    max_time_ms = df['time_ms'].max()
    frame_times = np.arange(0, max_time_ms, FRAME_INTERVAL_MS)

    print(f"Generating {len(frame_times)} frames for the GIF. This may take a while...")
    
    frames = []
    # Use tqdm for a progress bar
    for t_ms in tqdm(frame_times, desc="Creating frames"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
        fig.suptitle(f'SNN Learning and Dynamics Dashboard | Time: {t_ms/1000:.2f} s', fontsize=16, fontweight='bold')

        # --- Plot each subplot for the current time t_ms ---
        plot_weight_evolution_animated(axes[0, 0], synapse_df, t_ms)
        plot_weight_distribution_animated(axes[0, 1], synapse_df, t_ms)
        plot_decision_neuron_dynamics_animated(axes[1, 0], decision_df, t_ms)
        plot_synapse_count_evolution_animated(axes[1, 1], count_df, t_ms)

        # Set consistent x-axis limits for time-series plots
        for ax in [axes[0, 0], axes[1, 1]]:
            ax.set_xlim(0, max_time_ms / 1000)
            ax.axvline(t_ms / 1000, color='red', linestyle='--', linewidth=1, alpha=0.7)

        # Save frame to a temporary in-memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=DPI)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig) # Close the figure to free memory

    # Save the frames as a GIF
    print(f"\nStitching {len(frames)} frames into '{OUTPUT_GIF_FILE}'...")
    imageio.mimsave(OUTPUT_GIF_FILE, frames, fps=GIF_FPS, loop=0)
    print("\nDone! Your animated dashboard is ready.")

if __name__ == '__main__':
    # Add instructions for the user
    print("--- SNN Animated Dashboard Generator ---")
    print("This script will create an animated GIF of your simulation results.")
    print("If you haven't already, please install the required libraries:")
    print("pip install imageio imageio[ffmpeg] pillow tqdm\n")
    main()