#pragma once

#include <cstddef>

// Forward declarations to break circular dependencies
class ILearningRule;
struct Synapse;
struct Config;

// Type aliases
using NeuronId = size_t;
using SynapseId = size_t;

// Enums
enum class NeuronType { Excitatory, Inhibitory };
enum class NeuronModel { LIF, Izhikevich };

// Simple data structures
struct SpikeEvent {
    double delivery_time;
    SynapseId origin_synapse_id;
    bool operator>(const SpikeEvent& other) const { return delivery_time > other.delivery_time; }
};

struct LearningContext {
    double pre_spike_time;
    double post_spike_time;
    double dopamine_level;
};