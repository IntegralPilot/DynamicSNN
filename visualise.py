"""
SNN Simulation Analysis Dashboard

This script loads data from 'simulation_log.csv' and generates a 2x2 dashboard
visualizing the key aspects of the learning process.

It dynamically reads the neuron-to-pattern mappings from the log file's header,
ensuring the analysis is accurate for any given simulation run.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
LOG_FILE = 'simulation_log.csv'
OUTPUT_IMAGE_FILE = 'simulation_dashboard.png'


def load_data_and_patterns(filename):
    """
    Loads simulation data and dynamically parses pattern mappings from the CSV header.

    The C++ simulation is expected to write commented lines like:
    # PATTERN_MAPPING,PatternA,0,15,22,3,...
    # PATTERN_MAPPING,PatternB,4,8,19,21,...
    """
    print(f"Loading data from '{filename}'...")
    if not os.path.exists(filename):
        print(f"Error: Log file '{filename}' not found.")
        print("Please run the C++ simulation first.")
        return None, None

    patterns = {}
    print("Parsing pattern mappings from file header...")
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('# PATTERN_MAPPING'):
                    parts = line.strip('# \n').split(',')
                    pattern_name = parts[1]
                    # Convert all subsequent parts to integers to get the neuron IDs
                    neuron_ids = {int(id_str) for id_str in parts[2:]}
                    patterns[pattern_name] = neuron_ids
                    print(f"  - Found mapping for '{pattern_name}' with {len(neuron_ids)} neurons.")
                elif not line.startswith('#'):
                    # Stop reading the header once we hit the actual data
                    break
        
        if not patterns:
            print("Warning: No pattern mappings found in the file header. Plots may be incorrect.")

        # Now, load the main data using pandas, which will skip all commented lines
        df = pd.read_csv(filename, comment='#')
        
        # Convert types for efficiency and correctness
        df['time_ms'] = pd.to_numeric(df['time_ms'])
        df['value1'] = pd.to_numeric(df['value1'], errors='coerce')
        df['value2'] = pd.to_numeric(df['value2'], errors='coerce')
        
        print("Data loaded successfully.")
        return df, patterns

    except Exception as e:
        print(f"An error occurred while loading or parsing the file: {e}")
        return None, None


def parse_and_classify_synapses(df, patterns):
    """Filters for synapse data and classifies them into A, B, or Distractor."""
    syn_df = df[df['event_type'] == 'synapse'].copy()
    
    # Extract the presynaptic neuron ID from the name like "Sensory_42->Decision"
    # This assumes the NeuronId matches the numeric suffix in the name, which is true for our setup.
    syn_df['pre_id'] = syn_df['name'].str.extract(r'Sensory_(\d+)').astype(int)
    
    # Get the sets of IDs from the loaded patterns dictionary
    pattern_a_ids = patterns.get('PatternA', set())
    pattern_b_ids = patterns.get('PatternB', set())

    # Classify each synapse based on the dynamically loaded IDs
    def classify_synapse(pre_id):
        if pre_id in pattern_a_ids:
            return 'Pattern A'
        elif pre_id in pattern_b_ids:
            return 'Pattern B'
        else:
            return 'Distractor'
            
    syn_df['type'] = syn_df['pre_id'].apply(classify_synapse)
    syn_df.rename(columns={'value1': 'weight', 'value2': 'eligibility'}, inplace=True)
    
    return syn_df

def plot_weight_evolution(ax, synapse_df):
    """Plots the average weight of synapse groups over time."""
    print("Plotting weight evolution...")
    # Group by time and synapse type, then calculate the mean weight
    avg_weights = synapse_df.groupby(['time_ms', 'type'])['weight'].mean().unstack()

    if 'Pattern A' in avg_weights.columns:
        ax.plot(avg_weights.index / 1000, avg_weights['Pattern A'], label='Pattern A Synapses', color='royalblue', lw=2)
    if 'Pattern B' in avg_weights.columns:
        ax.plot(avg_weights.index / 1000, avg_weights['Pattern B'], label='Pattern B Synapses', color='darkorange', lw=2)
    
    # Plot distractor weights to see if they are stable
    if 'Distractor' in avg_weights.columns:
         ax.plot(avg_weights.index / 1000, avg_weights['Distractor'], label='Distractor Synapses', color='gray', ls='--', alpha=0.7)

    ax.set_title('Synaptic Weight Evolution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Average Synaptic Weight')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', lw=0.5)


def plot_final_weight_distribution(ax, synapse_df):
    """Plots a histogram of the final weights for each synapse group."""
    print("Plotting final weight distribution...")
    if synapse_df.empty:
        print("  - No synapse data to plot.")
        return
        
    final_time = synapse_df['time_ms'].max()
    final_weights = synapse_df[synapse_df['time_ms'] == final_time]

    weights_a = final_weights[final_weights['type'] == 'Pattern A']['weight']
    weights_b = final_weights[final_weights['type'] == 'Pattern B']['weight']
    
    bins = np.linspace(0, 20, 21) # Bins from 0 to 20 (max weight)
    ax.hist(weights_a, bins=bins, alpha=0.7, label='Pattern A Synapses', color='royalblue')
    ax.hist(weights_b, bins=bins, alpha=0.7, label='Pattern B Synapses', color='darkorange')

    ax.set_title('Final Synaptic Weight Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Synaptic Weight')
    ax.set_ylabel('Number of Synapses')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_decision_neuron_dynamics(ax, df):
    """Plots the Decision neuron's membrane potential and firing threshold."""
    print("Plotting decision neuron dynamics...")
    decision_df = df[(df['event_type'] == 'vm') & (df['name'] == 'Decision')]
    
    ax.plot(decision_df['time_ms'] / 1000, decision_df['value1'], label='Membrane Potential (Vm)', color='seagreen', alpha=0.8)
    ax.plot(decision_df['time_ms'] / 1000, decision_df['value2'], label='Firing Threshold', color='firebrick', lw=2, ls='--')

    ax.set_title('Decision Neuron Dynamics', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (mV)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_dopamine_signal(ax, df):
    """Plots the global dopamine level over time."""
    print("Plotting dopamine signal...")
    dopamine_df = df[df['event_type'] == 'dopamine']
    
    ax.plot(dopamine_df['time_ms'] / 1000, dopamine_df['value1'], label='Dopamine Level', color='purple')
    
    # Find base level to draw a line
    if not dopamine_df.empty:
        # Estimate base level from early values, robust to initial spikes
        base_level = dopamine_df['value1'].iloc[1:10].mean()
        ax.axhline(base_level, color='gray', linestyle=':', label=f'Est. Base Level ({base_level:.3f})')

    ax.set_title('Global Dopamine Signal (Reward/Punishment)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Dopamine (arbitrary units)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_synapse_count_evolution(ax, df):
    """Plots the number of synapses in each group over time."""
    print("Plotting synapse count evolution...")
    count_df = df[df['event_type'] == 'synapse_count'].copy()
    
    if count_df.empty:
        print("  - No synapse_count data found to plot.")
        ax.text(0.5, 0.5, 'No synapse_count data found', ha='center', va='center')
        ax.set_title('Synapse Count Evolution', fontsize=14, fontweight='bold')
        return

    # Pivot the data so each group ('name' column) gets its own column
    count_pivot = count_df.pivot(index='time_ms', columns='name', values='value1')

    # Plot each group
    if 'Pattern A' in count_pivot.columns:
        ax.plot(count_pivot.index / 1000, count_pivot['Pattern A'], label='Pattern A Synapses', color='royalblue', lw=2)
    if 'Pattern B' in count_pivot.columns:
        ax.plot(count_pivot.index / 1000, count_pivot['Pattern B'], label='Pattern B Synapses', color='darkorange', lw=2)
    if 'Distractor' in count_pivot.columns:
        ax.plot(count_pivot.index / 1000, count_pivot['Distractor'], label='Distractor Synapses', color='gray', ls='--', alpha=0.7)

    ax.set_title('Synapse Count Evolution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Number of Synapses')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0) # Number of synapses can't be negative


def main():
    """Main function to run the analysis."""
    df, patterns = load_data_and_patterns(LOG_FILE)
    if df is None:
        return

    # Prepare data for each plot
    synapse_df = parse_and_classify_synapses(df, patterns)
    
    # Create the 2x2 dashboard
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('SNN Learning and Dynamics Dashboard', fontsize=20, fontweight='bold')

    # Generate each plot on its respective axis
    plot_weight_evolution(axes[0, 0], synapse_df)
    plot_final_weight_distribution(axes[0, 1], synapse_df)
    plot_decision_neuron_dynamics(axes[1, 0], df)
    plot_synapse_count_evolution(axes[1, 1], df) 

    # Final adjustments and saving
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    print(f"Saving dashboard to '{OUTPUT_IMAGE_FILE}'...")
    plt.savefig(OUTPUT_IMAGE_FILE, dpi=150)
    print("Done.")
    plt.show()


if __name__ == '__main__':
    main()