#include "experiment.h"
#include "learning_rules.h" // For make_unique
#include <iostream>
#include <iomanip>
#include <numeric>      // For std::accumulate
#include <algorithm>    // For std::min_element, std::max_element
#include <unordered_set> // For efficient lookups
#include <random>
#include <unordered_map>

void print_stats_for_group(const std::string& name, const std::vector<double>& weights) {
    if (weights.empty()) {
        std::cout << "  - " << name << ": No plastic synapses found.\n";
        return;
    }

    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    double avg = sum / weights.size();
    double min_w = *std::min_element(weights.begin(), weights.end());
    double max_w = *std::max_element(weights.begin(), weights.end());

    std::cout << "  - " << name << " Weights (avg/min/max): "
              << std::fixed << std::setprecision(3) << avg << " / "
              << min_w << " / " << max_w << "\n";
}

void Experiment::print_weight_statistics() {
    // Use sets for efficient O(1) average time lookups
    const std::unordered_set<NeuronId> pattern_A_ids(_stimulus_A_pattern.begin(), _stimulus_A_pattern.end());
    const std::unordered_set<NeuronId> pattern_B_ids(_stimulus_B_pattern.begin(), _stimulus_B_pattern.end());

    std::vector<double> weights_A, weights_B, weights_distractor;
    int valid_plastic_synapse_count = 0;

    for (const auto& syn : _network.get_synapses()) {
        if (!syn.is_valid) continue;

        if (syn.post_neuron_id == _decision_neuron_id && syn.learning_rule) {
            valid_plastic_synapse_count++;
            if (pattern_A_ids.count(syn.pre_neuron_id)) {
                weights_A.push_back(syn.weight);
            } else if (pattern_B_ids.count(syn.pre_neuron_id)) {
                weights_B.push_back(syn.weight);
            } else {
                weights_distractor.push_back(syn.weight);
            }
        }
    }
    

    double current_time = _network.get_current_time();
    _logger.log_synapse_counts(current_time, "Pattern A", weights_A.size());
    _logger.log_synapse_counts(current_time, "Pattern B", weights_B.size());
    _logger.log_synapse_counts(current_time, "Distractor", weights_distractor.size());

    std::cout << "  Weight Statistics (Total plastic synapses: " << valid_plastic_synapse_count << "):\n";
    print_stats_for_group("Pattern A -> Decision", weights_A);
    print_stats_for_group("Pattern B -> Decision", weights_B);
    print_stats_for_group("Distractor -> Decision", weights_distractor);
}

Experiment::Experiment(const Config& config) 
    : _config(config), 
      _network(config), 
      _logger("simulation_log.csv") 
{
    _logger.log_parameters(_config);
    std::cout << "\n--- Building Network for Experiment ---\n";
    setup_network();
}

void Experiment::setup_network() {
    const int SENSORY_POP_SIZE = 1000;
    const int STIMULUS_PATTERN_SIZE = 200;

    // 1. Create neuron populations with spatial layout
    std::cout << "Creating populations with spatial layout...\n";

    _sensory_population = _network.add_neuron_population(SENSORY_POP_SIZE, "Sensory", NeuronType::Excitatory, NeuronModel::Izhikevich, 
        false,      // is_homeostatic
        true,       // can_sprout
        false,      // can_be_target
        {0, 50, 0, 50, 0, 0});
    
    _decision_neuron_id = _network.add_neuron("Decision", NeuronType::Excitatory, NeuronModel::LIF, 
        true,       // is_homeostatic
        false,      // can_sprout
        true,       // can_be_target
        {25, 25, -5});

    _inhibitory_neuron_id = _network.add_neuron("Inhibitory", NeuronType::Inhibitory, NeuronModel::LIF, 
        false,      // is_homeostatic
        false,      // can_sprout
        false,      // can_be_target
        {25, 25, -10});

    // 2. Define the stimulus patterns by selecting random neurons from the sensory population
    std::cout << "Defining stimulus patterns...\n";
    auto& rng = _network.get_rng();
    std::vector<NeuronId> shuffled_ids = _sensory_population;
    std::shuffle(shuffled_ids.begin(), shuffled_ids.end(), rng);
    
    _stimulus_A_pattern.assign(shuffled_ids.begin(), shuffled_ids.begin() + STIMULUS_PATTERN_SIZE);
    _stimulus_B_pattern.assign(shuffled_ids.begin() + STIMULUS_PATTERN_SIZE, shuffled_ids.begin() + 2 * STIMULUS_PATTERN_SIZE);
    std::cout << "  - Stimulus A uses " << _stimulus_A_pattern.size() << " neurons.\n";
    std::cout << "  - Stimulus B uses " << _stimulus_B_pattern.size() << " neurons.\n";

    _logger.log_pattern_mapping("PatternA", _stimulus_A_pattern);
    _logger.log_pattern_mapping("PatternB", _stimulus_B_pattern);

    // 3. Connect the populations - STARTING SPARSELY
    std::cout << "Connecting populations (starting sparsely to test structural plasticity)...\n";
    std::vector<NeuronId> decision_pop = {_decision_neuron_id};
    
    // Connect Sensory -> Decision (Plastic synapses)
    _network.connect_populations(
        _sensory_population, decision_pop,
        0.2,
        {_config.INITIAL_WEIGHT_MIN, _config.INITIAL_WEIGHT_MAX},
        {1.5, 5.0},
        [](){ return std::make_unique<RewardModulatedSTDP>(); }
    );

    // Connect Decision -> Inhibitory (Fixed, strong synapse)
    _network.connect(
        "Decision", "Inhibitory", 
        15.0, 1.0, nullptr
    );

    // Connect Inhibitory -> Decision (Fixed inhibitory feedback)
    _network.connect(
        "Inhibitory", "Decision",
        8.0, 1.0, nullptr
    );
}


void Experiment::run_trial(const std::vector<NeuronId>& stimulus_pattern, NeuronId target_id, bool should_be_rewarded) {
    double target_spike_time_before = _network.get_neuron(target_id).get_last_spike_time();
    double next_log_time = _network.get_current_time();
    
    // Poisson event generator
    double rate_per_step = _config.STIMULUS_POISSON_RATE_HZ * (_config.TIME_STEP / 1000.0);
    std::bernoulli_distribution stim_dist(rate_per_step);
    auto& rng = _network.get_rng();

    // Map to track the remaining duration of the current pulse for each neuron
    std::unordered_map<NeuronId, double> active_pulses_ms;

    std::cout << "  Presenting stimulus for " << _config.STIMULUS_DURATION_MS << " ms...\n";

    // --- Main Trial Loop ---
    for (double t = 0; t < _config.TRIAL_DURATION_MS; t += _config.TIME_STEP) {
        
        // --- STIMULATION LOGIC ---
        // Only trigger new pulses during the stimulus window
        if (t < _config.STIMULUS_DURATION_MS) {
            for (const auto& nid : stimulus_pattern) {
                // If a Poisson event occurs, start a new pulse (or reset an existing one)
                if (stim_dist(rng)) {
                    active_pulses_ms[nid] = _config.STIMULUS_PULSE_DURATION_MS;
                }
            }
        }

        // Apply current to all neurons with an active pulse
        for (auto it = active_pulses_ms.begin(); it != active_pulses_ms.end(); /* manual increment */) {
            NeuronId nid = it->first;
            double& remaining_time = it->second;
            
            if (remaining_time > 0) {
                _network.get_neuron(nid).apply_external_current(_config.STIMULATION_CURRENT_NA);
                remaining_time -= _config.TIME_STEP;
                ++it;
            } else {
                // Remove expired pulses from the map
                it = active_pulses_ms.erase(it);
            }
        }

        // Run the network for one time step
        _network.run(_config.TIME_STEP);
        
        if (_network.get_current_time() >= next_log_time) {
            log_all_states();
            next_log_time += _config.LOG_INTERVAL_MS;
        }
    }
    
    bool target_fired = _network.get_neuron(target_id).get_last_spike_time() > target_spike_time_before;
    
    if (should_be_rewarded) { // This is a "Go" trial
        if (target_fired) {
            std::cout << "  ✅ SUCCESS: Decision neuron fired. REWARDING.\n";
            _network.reward(_config.REWARD_AMOUNT_SUCCESS);
        } else {
            std::cout << "  - Miss: Decision neuron did not fire. REINFORCING.\n";
            // Manually create eligibility traces for the synapses that should have fired.
            _network.reinforce_silent_contributors(target_id, stimulus_pattern, _config.REINFORCE_AMOUNT_OMISSION);
            // Now apply a reward so the positive trace is multiplied by positive dopamine, causing LTP.
            _network.reward(_config.REWARD_AMOUNT_SUCCESS);
        }
    } else { // This is a "No-Go" trial
        if (target_fired) {
            std::cout << "  - Mistake: Decision neuron fired. PUNISHING.\n";
            _network.tag_synapses_for_punishment(
                target_id, 
                _config.PUNISHMENT_TAG_AMOUNT, 
                _config.PUNISHMENT_TAG_WINDOW_MS
            );
            _network.punish(_config.PUNISH_AMOUNT_MISTAKE);
        } else {
            std::cout << "  ✅ CORRECT INHIBITION: Decision neuron did not fire. (Unlearning contributors)\n";
            // To prevent weights from getting "frozen" high, we must gently punish
            // the synapses that contributed to the (correctly) silenced output.
            _network.punish_silent_contributors(target_id, stimulus_pattern, _config.UNLEARNING_TAG_AMOUNT);
            // Now apply a punishment so the punishment trace is multiplied by negative dopamine, causing LTD.
            _network.punish(_config.PUNISH_AMOUNT_MISTAKE); 
        }
    }
    _network.apply_and_reset_learning();
}

void Experiment::run() {
    std::cout << "\n--- Starting Training ---\n"
              << "Task: Reward network if 'Decision' fires for pattern 'A', but not for 'B'.\n"
              << "Logging data to simulation_log.csv\n";

    if (_config.ENABLE_STRUCTURAL_PLASTICITY) {
        std::cout << "💡 Structural plasticity is ENABLED. Network will attempt to grow/prune synapses.\n";
    } else {
        std::cout << "💡 Structural plasticity is DISABLED.\n";
    }

    _logger.write_data_header();

    // --- PHASE 1: INITIAL TRAINING ---
    std::cout << "\n\n--- PHASE 1: Initial Training (A=Go, B=No-Go) ---\n";
    for (int i = 0; i < _config.TRAINING_EPOCHS; ++i) {
        std::cout << "\n--- Epoch " << i + 1 << "/" << _config.TRAINING_EPOCHS << " ---\n";

        // run_trial(stimulus_pattern, target_id, should_be_rewarded)
        run_trial(_stimulus_A_pattern, _decision_neuron_id, true); // A is rewarded
        _network.run(_config.ITI_MS); // Inter-trial interval

        run_trial(_stimulus_B_pattern, _decision_neuron_id, false); // B is not
        _network.run(_config.ITI_MS);

        print_weight_statistics();
    }

    // --- PHASE 2: TASK REVERSAL ---
    if (_config.REVERSAL_EPOCHS && *_config.REVERSAL_EPOCHS > 0) {
        std::cout << "\n\n"
                  << "***************************************************\n"
                  << "*               !!! TASK REVERSAL !!!               *\n"
                  << "***************************************************\n"
                  << "New Task: Reward network if 'Decision' fires for pattern 'B', but not for 'A'.\n";
        
        int reversal_epochs = *_config.REVERSAL_EPOCHS;
        for (int i = 0; i < reversal_epochs; ++i) {
            // We use the total epoch count for clarity in logs
            int total_epoch = _config.TRAINING_EPOCHS + i + 1;
            std::cout << "\n--- Reversal Epoch " << i + 1 << "/" << reversal_epochs
                      << " (Total Epoch " << total_epoch << ") ---\n";
            
            // The only change is here: the boolean flags are flipped.
            run_trial(_stimulus_A_pattern, _decision_neuron_id, false); // A is now punished
            _network.run(_config.ITI_MS);

            run_trial(_stimulus_B_pattern, _decision_neuron_id, true); // B is now rewarded
            _network.run(_config.ITI_MS);

            print_weight_statistics();
        }
    }


    std::cout << "\n\n--- Training Complete ---\n";
}

void Experiment::log_all_states() {
    double time = _network.get_current_time();

    // Log a few representative sensory neurons from each pattern + the decision/inhibitory neurons
    // This ensures we log something meaningful without flooding the CSV file.
    _logger.log_neuron_vm(time, _network.get_neuron(_stimulus_A_pattern[0]));
    _logger.log_neuron_vm(time, _network.get_neuron(_stimulus_B_pattern[0]));
    _logger.log_neuron_vm(time, _network.get_neuron(_decision_neuron_id));
    _logger.log_neuron_vm(time, _network.get_neuron(_inhibitory_neuron_id));

    // Logging all plastic synapses is still useful to see weight distribution
    const auto& synapses = _network.get_synapses();
    for(size_t i = 0; i < synapses.size(); ++i) {
        if (synapses[i].learning_rule) {
            const auto& pre_n = _network.get_neuron(synapses[i].pre_neuron_id);
            const auto& post_n = _network.get_neuron(synapses[i].post_neuron_id);
            _logger.log_synapse_weight(time, i, pre_n.name(), post_n.name(), synapses[i]);
        }
    }
    _logger.log_dopamine(time, _network.get_dopamine_level());
}