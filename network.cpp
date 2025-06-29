// network.cpp
#include "network.h"
#include "config.h"
#include "learning_rules.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>    // Include OpenMP for parallelization
#include <thread>   // For std::hash<std::thread::id> for seeding RNGs
#include <numeric>
#include <algorithm>
#include <unordered_set>

// Helper function for calculating distance
double distance_sq(const std::array<double, 3>& p1, const std::array<double, 3>& p2) {
    double dx = p1[0] - p2[0];
    double dy = p1[1] - p2[1];
    double dz = p1[2] - p2[2];
    return dx * dx + dy * dy + dz * dz;
}

Network::Network(const Config& config) 
    : _config(config), _noise_dist(0.0, config.NOISE_STD_DEV_NA) 
{
    if (_config.SIMULATION_SEED) {
        std::cout << "🌱 Seeding simulation with: " << _config.SIMULATION_SEED.value() << "\n";
        _rng.seed(_config.SIMULATION_SEED.value());
    } else {
        std::cout << "🎲 Seeding simulation with random device.\n";
        _rng.seed(std::random_device{}());
    }
    _dopamine_level = _config.DOPAMINE_BASE_LEVEL;
    _dopamine_decay_lambda = std::log(2.0) / _config.DOPAMINE_DECAY_HALF_LIFE_MS;
}

NeuronId Network::add_neuron(const std::string& name, NeuronType type, NeuronModel model, bool is_homeostatic, bool can_sprout, bool can_be_target, const std::array<double, 3>& pos) {
    if (_neuron_name_map.count(name)) return _neuron_name_map[name];
    NeuronId id = _neurons.size();
    // Pass the new parameter to the Neuron constructor
    _neurons.emplace_back(id, name, type, model, _config, is_homeostatic, can_sprout, can_be_target, pos); 
    _neuron_name_map[name] = id;
    std::cout << " Neuron added: " << name << " (ID: " << id
              << ", Type: " << (type == NeuronType::Excitatory ? "Excitatory" : "Inhibitory")
              << ", Model: " << (model == NeuronModel::LIF ? "LIF" : "Izhikevich")
              << ", Homeostatic: " << (is_homeostatic ? "Yes" : "No") 
              << ", Can Sprout: " << (can_sprout ? "Yes" : "No") 
              << ", Is Target: " << (can_be_target ? "Yes" : "No") << ")\n"; // <-- Add to log
    return id;
}

void Network::connect(const std::string& pre_name, const std::string& post_name, double weight, double delay_ms, std::unique_ptr<ILearningRule> rule) {
    auto pre_id_opt = get_neuron_id_by_name(pre_name);
    auto post_id_opt = get_neuron_id_by_name(post_name);
    if (!pre_id_opt || !post_id_opt) return;
    
    NeuronId pre_id = *pre_id_opt;
    NeuronId post_id = *post_id_opt;
    
    SynapseId s_id = _synapses.size();
    _synapses.emplace_back(pre_id, post_id, _neurons[pre_id].type(), weight, delay_ms, _config, std::move(rule));
    _neurons[pre_id].add_outgoing_synapse(s_id);
    _neurons[post_id].add_incoming_synapse(s_id);
    std::cout << "🔗 Synapse created: " << pre_name << " -> " << post_name
              << " (Weight: " << std::fixed << std::setprecision(2) << weight
              << ", Delay: " << std::setprecision(1) << delay_ms << "ms, Plastic: " << (_synapses.back().learning_rule ? "Yes" : "No") << ")\n";
}

void Network::reward(double amount) { _dopamine_level += amount; }
void Network::punish(double amount) { _dopamine_level -= amount; }

void Network::run(double duration_ms) {
    double end_time = _current_time_ms + duration_ms;
    
    std::vector<SpikeEvent> new_spikes_this_step;

    while (_current_time_ms < end_time) {
        // --- EVENT PROCESSING (Pre-synaptic side of STDP) ---
        while (!_event_queue.empty() && _event_queue.top().delivery_time <= _current_time_ms) {
            SpikeEvent event = _event_queue.top(); _event_queue.pop();
            Synapse& syn = _synapses[event.origin_synapse_id];
            Neuron& post_neuron = _neurons[syn.post_neuron_id];
            post_neuron.receive_spike(syn.transmit(event.delivery_time, _config), syn.pre_type);
            syn.last_pre_spike_time = event.delivery_time;

            // This is the pre-synaptic spike event. The post-synaptic neuron hasn't fired yet in response.
            LearningContext ctx = { event.delivery_time, post_neuron.get_last_spike_time(), _dopamine_level };
            if (syn.learning_rule) {
                syn.learning_rule->process_spike_timing(syn, ctx, _config);
            }
        }
        
        std::vector<NeuronId> fired_neuron_ids;

        // --- NEURON UPDATE (Parallelized) ---
        // This loop is safe to parallelize because each neuron's update depends only on its own state.
        #pragma omp parallel
        {
            std::vector<SpikeEvent> local_new_spikes;
            std::vector<NeuronId> local_fired_neuron_ids;
            
            thread_local static std::mt19937 thread_rng(
                _config.SIMULATION_SEED ? (*_config.SIMULATION_SEED) ^ std::hash<std::thread::id>{}(std::this_thread::get_id()) 
                                        : std::random_device{}() ^ std::hash<std::thread::id>{}(std::this_thread::get_id())
            );
            std::normal_distribution<> thread_noise_dist(0.0, _config.NOISE_STD_DEV_NA);

            #pragma omp for
            for (NeuronId i = 0; i < _neurons.size(); ++i) {
                double noise = thread_noise_dist(thread_rng);
                if(_neurons[i].update(_current_time_ms, _config.TIME_STEP, noise)) {
                    local_fired_neuron_ids.push_back(i);
                    // Generate outgoing spikes but store them locally for now.
                    for (const auto& s_id : _neurons[i].get_outgoing_synapses()) {
                        const auto& syn = _synapses[s_id];
                        local_new_spikes.push_back({_current_time_ms + syn.delay_ms, s_id});
                    }
                }
            }

            #pragma omp critical
            {
                new_spikes_this_step.insert(new_spikes_this_step.end(), local_new_spikes.begin(), local_new_spikes.end());
                fired_neuron_ids.insert(fired_neuron_ids.end(), local_fired_neuron_ids.begin(), local_fired_neuron_ids.end());
            }
        } // End of parallel neuron update

        // --- POST-SPIKE PROCESSING (Post-synaptic side of STDP) ---
        // This must be done serially after all neurons have been updated.
        for(NeuronId nid : fired_neuron_ids) {
            LearningContext ctx = { -1.0, _current_time_ms, _dopamine_level };
            // For each incoming synapse to the neuron that just fired...
            for (const auto& s_id : _neurons[nid].get_incoming_synapses()) {
                Synapse& syn = _synapses[s_id];
                ctx.pre_spike_time = syn.last_pre_spike_time; // Get the pre-spike time from the synapse
                if (syn.learning_rule) {
                    syn.learning_rule->process_spike_timing(syn, ctx, _config);
                }
            }
        }

        // Add newly generated spikes to the main event queue.
        for(const auto& spike : new_spikes_this_step) {
            _event_queue.push(spike);
        }
        new_spikes_this_step.clear();

        // --- SYNAPSE UPDATE (Parallelized) ---
        #pragma omp parallel for
        for (size_t i = 0; i < _synapses.size(); ++i) {
            _synapses[i].update(_config.TIME_STEP, _config);
        }

        // --- GLOBAL STATE UPDATE ---
        double base = _config.DOPAMINE_BASE_LEVEL;
        double decay_factor = std::exp(-_dopamine_decay_lambda * _config.TIME_STEP);
        _dopamine_level = base + (_dopamine_level - base) * decay_factor;

        _current_time_ms += _config.TIME_STEP;

        // --- STRUCTURAL PLASTICITY UPDATE ---
        if (_config.ENABLE_STRUCTURAL_PLASTICITY &&
            (_current_time_ms - _time_of_last_structure_update >= _config.STRUCTURAL_UPDATE_INTERVAL_MS))
        {
            std::cout << "Running structural update at t=" << _current_time_ms << "ms...\n";
            _update_structure();
            _time_of_last_structure_update = _current_time_ms;
        }
    }
}


void Network::stimulate_neuron_by_name(const std::string& name, double strength_nA) {
    auto id_opt = get_neuron_id_by_name(name);
    if (id_opt) {
        _neurons[*id_opt].apply_external_current(strength_nA);
    }
}

std::vector<NeuronId> Network::add_neuron_population(int count, const std::string& name_prefix, NeuronType type, NeuronModel model, bool is_homeostatic, bool can_sprout, bool can_be_target, const std::array<double, 6>& volume_bounds) {
    std::vector<NeuronId> ids;
    ids.reserve(count);
    std::uniform_real_distribution<> x_dist(volume_bounds[0], volume_bounds[1]);
    std::uniform_real_distribution<> y_dist(volume_bounds[2], volume_bounds[3]);
    std::uniform_real_distribution<> z_dist(volume_bounds[4], volume_bounds[5]);

    for (int i = 0; i < count; ++i) {
        std::string name = name_prefix + "_" + std::to_string(i);
        std::array<double, 3> pos = {x_dist(_rng), y_dist(_rng), z_dist(_rng)};
        // Pass the new parameter here
        ids.push_back(add_neuron(name, type, model, is_homeostatic, can_sprout, can_be_target, pos)); 
    }
    return ids;
}

void Network::connect_populations(
    const std::vector<NeuronId>& pre_pop,
    const std::vector<NeuronId>& post_pop,
    double connection_probability,
    const std::pair<double, double>& weight_range,
    const std::pair<double, double>& delay_range,
    std::function<std::unique_ptr<ILearningRule>()> rule_factory) 
{
    std::uniform_real_distribution<> weight_dist(weight_range.first, weight_range.second);
    std::uniform_real_distribution<> delay_dist(delay_range.first, delay_range.second);
    std::bernoulli_distribution connect_dist(connection_probability);

    for (const auto& pre_id : pre_pop) {
        for (const auto& post_id : post_pop) {
            if (pre_id == post_id) continue;
            if (connect_dist(_rng)) {
                std::unique_ptr<ILearningRule> rule = rule_factory ? rule_factory() : nullptr;
                connect(
                    _neurons[pre_id].name(),
                    _neurons[post_id].name(),
                    weight_dist(_rng),
                    delay_dist(_rng),
                    std::move(rule)
                );
            }
        }
    }
}

// --- Getters ---
double Network::get_current_time() const { return _current_time_ms; }
double Network::get_dopamine_level() const { return _dopamine_level; }
std::optional<NeuronId> Network::get_neuron_id_by_name(const std::string& name) const {
    auto it = _neuron_name_map.find(name);
    if (it != _neuron_name_map.end()) {
        return it->second;
    }
    return std::nullopt;
}
const Neuron& Network::get_neuron(NeuronId id) const { return _neurons[id]; }
const std::vector<Neuron>& Network::get_neurons() const { return _neurons; }
Neuron& Network::get_neuron(NeuronId id) { return _neurons[id]; }
const std::vector<Synapse>& Network::get_synapses() const { return _synapses; }
std::mt19937& Network::get_rng() { return _rng; }

void Network::apply_and_reset_learning() {
    for (auto& syn : _synapses) {
        syn.apply_learning(_config, _dopamine_level);
        syn.eligibility_trace = 0.0; 
        syn.punishment_trace = 0.0;
    }
}

void Network::_update_structure() {
    // --- PHASE 1: PRUNING ---
    std::vector<SynapseId> synapses_to_prune;
    for (size_t i = 0; i < _synapses.size(); ++i) {
        if (_synapses[i].is_valid && 
           (_synapses[i].weight < _config.PRUNING_WEIGHT_THRESHOLD || 
            _synapses[i].punishment_trace > _config.PRUNING_PUNISHMENT_THRESHOLD)
        ) {            
            synapses_to_prune.push_back(i);
        }
    }
    if (!synapses_to_prune.empty()) {
        std::cout << "  - Pruning " << synapses_to_prune.size() << " weak synapses.\n";
        for (SynapseId s_id : synapses_to_prune) {
            Synapse& syn = _synapses[s_id];
            syn.is_valid = false;
            Neuron& pre_n = _neurons[syn.pre_neuron_id];
            Neuron& post_n = _neurons[syn.post_neuron_id];
            auto& out_vec = pre_n.get_outgoing_synapses_for_modification();
            auto& in_vec = post_n.get_incoming_synapses_for_modification();
            out_vec.erase(std::remove(out_vec.begin(), out_vec.end(), s_id), out_vec.end());
            in_vec.erase(std::remove(in_vec.begin(), in_vec.end(), s_id), in_vec.end());
        }
    }
    
    // --- PHASE 2: SPROUTING / SYNAPTOGENESIS (Generalized Logic) ---
    
    // 1. Identify candidates based on activity and their `can_sprout` property
    std::vector<NeuronId> sprouting_candidates;
    double max_sprouter_firing_rate = 0.0;

    for (const auto& neuron : _neurons) {
        // A neuron is a potential sprouter if it is flagged and of the correct type (optional, but good practice)
        if (neuron.can_sprout() && neuron.type() == NeuronType::Excitatory) {
            max_sprouter_firing_rate = std::max(max_sprouter_firing_rate, neuron.get_avg_firing_rate());
            if (neuron.get_avg_firing_rate() > _config.SPROUTING_ACTIVITY_THRESHOLD_HZ) {
                sprouting_candidates.push_back(neuron.id());
            }
        }
    }

    std::cout << "  - Sprouting Diagnostics:\n";
    std::cout << "    - Max Avg Rate of any Sprouting-Capable Neuron: " << std::fixed << std::setprecision(4) << max_sprouter_firing_rate << " Hz\n";
    std::cout << "    - Sprouting Threshold:                          " << _config.SPROUTING_ACTIVITY_THRESHOLD_HZ << " Hz\n";

    if (sprouting_candidates.empty()) {
        std::cout << "  - Result: No sprouting-capable neurons met the activity threshold.\n";
        return; 
    }

    std::cout << "  - Result: Found " << sprouting_candidates.size() << " candidates for sprouting.\n";

    // 2. Build a set of existing connections for fast O(1) lookups (Unchanged)
    std::unordered_set<uint64_t> existing_connections;
    for (const auto& syn : _synapses) {
        if (syn.is_valid) {
            existing_connections.insert((uint64_t)syn.pre_neuron_id << 32 | syn.post_neuron_id);
        }
    }

    // 3. Attempt to form new connections by searching for nearby partners
    int new_synapses_formed = 0;
    std::bernoulli_distribution connect_dist(_config.NEW_CONNECTION_PROBABILITY);
    double max_dist_sq = _config.SPROUTING_MAX_DISTANCE * _config.SPROUTING_MAX_DISTANCE;

    for (NeuronId pre_id : sprouting_candidates) {
        const Neuron& pre_neuron = _neurons[pre_id];
        
        // Loop through ALL neurons to find potential post-synaptic partners
        for (const auto& post_neuron : _neurons) {
            NeuronId post_id = post_neuron.id();

            // --- Apply Connection Rules ---
            // Rule 1: Don't connect to yourself
            if (pre_id == post_id) continue;

            // Rule 2: Don't connect if the post-synaptic neuron is not sprouting-capable
            if (!post_neuron.can_be_target()) continue;

            // Rule 3: Don't connect if a synapse already exists
            if (existing_connections.count((uint64_t)pre_id << 32 | post_id)) continue;
            
            // Rule 4: Only connect to neurons within the maximum sprouting distance
            if (distance_sq(pre_neuron.position(), post_neuron.position()) > max_dist_sq) continue;
            
            // Rule 5: Probabilistically form the connection
            if (connect_dist(_rng)) {
                connect(pre_neuron.name(), post_neuron.name(), 
                        _config.NEW_SYNAPSE_INITIAL_WEIGHT, 
                        1.0, // Fixed delay for simplicity, could be randomized
                        std::make_unique<RewardModulatedSTDP>()); 

                new_synapses_formed++;
                // Add to set to prevent reconnecting in the same update step
                existing_connections.insert((uint64_t)pre_id << 32 | post_id);
            }
        }
    }
    
    if (new_synapses_formed > 0) {
        std::cout << "  - Sprouted " << new_synapses_formed << " new synapses.\n";
    }
}

void Network::tag_synapses_for_punishment(NeuronId post_neuron_id, double tag_amount, double time_window_ms) {
    // Get the neuron that fired by mistake
    const Neuron& post_neuron = _neurons[post_neuron_id];

    // Look at all synapses that connect to this neuron
    for (const auto& s_id : post_neuron.get_incoming_synapses()) {
        Synapse& syn = _synapses[s_id];

        // Check if the presynaptic neuron fired within the recent time window
        if ((_current_time_ms - syn.last_pre_spike_time) <= time_window_ms) {
            // This synapse is a likely contributor to the mistaken spike. Tag it.
            syn.punishment_trace += tag_amount;
            
            std::cout << "  - Tagging synapse " << syn.pre_neuron_id << "->" << syn.post_neuron_id 
                      << " for punishment. Trace is now " << syn.punishment_trace << "\n";
        }
    }
}

void Network::reinforce_silent_contributors(NeuronId target_id, const std::vector<NeuronId>& contributing_pattern, double amount) {
    // For fast lookup of which neurons were part of the stimulus
    std::unordered_set<NeuronId> pattern_set(contributing_pattern.begin(), contributing_pattern.end());

    // Get the neuron that should have fired
    const Neuron& target_neuron = _neurons[target_id];

    // Iterate through all incoming synapses to the target neuron
    for (const auto& s_id : target_neuron.get_incoming_synapses()) {
        Synapse& syn = _synapses[s_id];
        
        // If the presynaptic neuron was part of the stimulus pattern...
        if (pattern_set.count(syn.pre_neuron_id)) {
            // ... and the synapse is plastic...
            if (syn.learning_rule) {
                // ...manually create a positive eligibility trace. This synapse "gets credit" for trying.
                syn.eligibility_trace += amount;
            }
        }
    }
}

void Network::punish_silent_contributors(NeuronId target_id, const std::vector<NeuronId>& contributing_pattern, double amount) {
    // For fast lookup of which neurons were part of the stimulus
    std::unordered_set<NeuronId> pattern_set(contributing_pattern.begin(), contributing_pattern.end());

    // Get the neuron that correctly did not fire
    const Neuron& target_neuron = _neurons[target_id];

    // Iterate through all incoming synapses to the target neuron
    for (const auto& s_id : target_neuron.get_incoming_synapses()) {
        Synapse& syn = _synapses[s_id];
        
        // If the presynaptic neuron was part of the stimulus pattern...
        if (pattern_set.count(syn.pre_neuron_id)) {
            // ... and the synapse is plastic...
            if (syn.learning_rule) {
                // ...manually create a PUNISHMENT trace. This synapse "gets blamed" for trying.
                syn.punishment_trace += amount;
            }
        }
    }
}
