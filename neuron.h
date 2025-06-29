// neuron.h
#pragma once

#include "core_types.h"
#include <string>
#include <vector>
#include <array>

// Forward declarations
struct Config;
struct Synapse;

class Neuron {
private:
    NeuronId _id;
    std::string _name;
    NeuronType _type;
    NeuronModel _model;
    const Config& _config;
    double _membrane_vm, _last_spike_time = -1000.0, _spike_threshold_mv;
    double _g_excitatory_ns = 0.0, _g_inhibitory_ns = 0.0;
    double _izh_u = 0.0;
    double _avg_firing_rate_hz = 0.0;
    double _external_current_nA = 0.0;
    bool _is_homeostatic;
    bool _can_sprout;
    bool _can_be_target;
    std::vector<SynapseId> _incoming_synapses, _outgoing_synapses;
    std::array<double, 3> _position;

    void update_lif(double dt, double I_external);
    void update_izhikevich(double dt, double I_external);
    void fire(double current_time); 
    void update_homeostasis(double dt);

public:
    Neuron(NeuronId id, std::string name, NeuronType type, NeuronModel model, const Config& config, bool is_homeostatic, bool can_sprout, bool can_be_target, const std::array<double, 3>& pos);
    
    // Getters
    NeuronId id() const;
    const std::string& name() const;
    double vm() const;
    double threshold() const;
    NeuronType type() const;
    const std::vector<SynapseId>& get_outgoing_synapses() const;
    const std::vector<SynapseId>& get_incoming_synapses() const;
    std::vector<SynapseId>& get_outgoing_synapses_for_modification();
    std::vector<SynapseId>& get_incoming_synapses_for_modification();
    double get_last_spike_time() const;
    double get_avg_firing_rate() const;
    bool can_sprout() const;
    bool can_be_target() const;
    const std::array<double, 3>& position() const;

    // Methods
    void add_incoming_synapse(SynapseId s_id);
    void add_outgoing_synapse(SynapseId s_id);
    void receive_spike(double delta_g, NeuronType pre_type);
    void apply_external_current(double strength_nA);
    bool update(double current_time, double dt, double noise_current_nA);
};