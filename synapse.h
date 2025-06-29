#pragma once

#include "core_types.h"
#include <memory>

// Forward declarations to reduce include dependencies
class Config;
class ILearningRule;

struct Synapse {
    NeuronId pre_neuron_id, post_neuron_id;
    NeuronType pre_type;
    double weight, delay_ms;
    double last_pre_spike_time = -1.0;
    std::unique_ptr<ILearningRule> learning_rule;
    double stp_x = 1.0, stp_u;
    double eligibility_trace = 0.0;
    double punishment_trace = 0.0;
    bool is_valid = true;

    Synapse(NeuronId pre, NeuronId post, NeuronType type, double w, double d, const Config& config, std::unique_ptr<ILearningRule> rule);

    // Rule of Five implemented directly in the class definition
    Synapse(const Synapse&) = delete;
    Synapse& operator=(const Synapse&) = delete;
    Synapse(Synapse&&) noexcept = default;
    Synapse& operator=(Synapse&&) noexcept = default;
    ~Synapse();

    double transmit(double current_time, const Config& config);
    void update(double dt, const Config& config); // No longer takes dopamine
    void apply_learning(const Config& config, double current_dopamine_level);
};