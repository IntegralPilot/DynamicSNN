// === experiment.h ===
#pragma once

#include "network.h"
#include "datalogger.h"
#include "config.h"

class Experiment {
private:
    Config _config;
    Network _network;
    DataLogger _logger;
    
    // --- Populations and Patterns ---
    std::vector<NeuronId> _sensory_population;
    NeuronId _decision_neuron_id;
    NeuronId _inhibitory_neuron_id;

    std::vector<NeuronId> _stimulus_A_pattern;
    std::vector<NeuronId> _stimulus_B_pattern;

    void setup_network();
    void run_trial(const std::vector<NeuronId>& stimulus_pattern, NeuronId target_id, bool should_be_rewarded);
    void log_all_states();
    void print_weight_statistics(); 

public:
    explicit Experiment(const Config& config);
    void run();
};