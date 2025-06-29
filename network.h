// network.h
#pragma once

#include "core_types.h"
#include "config.h"
#include "neuron.h"
#include "synapse.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <random>
#include <memory>
#include <optional>

class Network {
private:
    Config _config;
    std::vector<Neuron> _neurons;
    std::vector<Synapse> _synapses;
    std::unordered_map<std::string, NeuronId> _neuron_name_map;
    std::priority_queue<SpikeEvent, std::vector<SpikeEvent>, std::greater<SpikeEvent>> _event_queue;
    double _current_time_ms = 0.0, _dopamine_level, _dopamine_decay_lambda;
    double _time_of_last_structure_update = 0.0;
    
    std::mt19937 _rng;
    std::normal_distribution<> _noise_dist;

    void _update_structure();

public:
    explicit Network(const Config& config);

    NeuronId add_neuron(const std::string& name, NeuronType type, NeuronModel model, bool is_homeostatic, bool can_sprout, bool can_be_target, const std::array<double, 3>& pos = {0,0,0});
    std::vector<NeuronId> add_neuron_population(
        int count,
        const std::string& name_prefix,
        NeuronType type,
        NeuronModel model,
        bool is_homeostatic,
        bool can_sprout,
        bool can_be_target,
        const std::array<double, 6>& volume_bounds
    );

    void connect(const std::string& pre_name, const std::string& post_name, double weight, double delay_ms, std::unique_ptr<ILearningRule> rule);

    void reward(double amount);
    void punish(double amount);
    void reinforce_silent_contributors(NeuronId target_id, const std::vector<NeuronId>& contributing_pattern, double amount);
    void punish_silent_contributors(NeuronId target_id, const std::vector<NeuronId>& contributing_pattern, double amount);
    void tag_synapses_for_punishment(NeuronId post_neuron_id, double tag_amount, double time_window_ms);

    void run(double duration_ms);
    void stimulate_neuron_by_name(const std::string& name, double strength_nA);
    
    // Getters
    double get_current_time() const;
    double get_dopamine_level() const;
    std::optional<NeuronId> get_neuron_id_by_name(const std::string& name) const;
    const Neuron& get_neuron(NeuronId id) const;
    Neuron& get_neuron(NeuronId id); 
    const std::vector<Neuron>& get_neurons() const;
    const std::vector<Synapse>& get_synapses() const;
    std::mt19937& get_rng();

    void connect_populations(
        const std::vector<NeuronId>& pre_pop,
        const std::vector<NeuronId>& post_pop,
        double connection_probability,
        const std::pair<double, double>& weight_range,
        const std::pair<double, double>& delay_range,
        std::function<std::unique_ptr<ILearningRule>()> rule_factory
    );

    void apply_and_reset_learning();
};