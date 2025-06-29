// neuron.cpp
#include "neuron.h"
#include "synapse.h"
#include "config.h"
#include "learning_rules.h"
#include <cmath>
#include <algorithm>
#include <utility>

Neuron::Neuron(NeuronId id, std::string name, NeuronType type, NeuronModel model, const Config& config, bool is_homeostatic, bool can_sprout, bool can_be_target, const std::array<double, 3>& pos)
    : _id(id), _name(std::move(name)), _type(type), _model(model), _config(config),
      _membrane_vm(config.RESTING_POTENTIAL_MV), _spike_threshold_mv(config.SPIKE_THRESHOLD_MV),
      _is_homeostatic(is_homeostatic), _can_sprout(can_sprout), _can_be_target(can_be_target), _position(pos)
{
    _izh_u = _config.IZH_B * _membrane_vm;
}

// --- Getters ---
NeuronId Neuron::id() const { return _id; }
const std::string& Neuron::name() const { return _name; }
double Neuron::vm() const { return _membrane_vm; }
double Neuron::threshold() const { return _spike_threshold_mv; }
NeuronType Neuron::type() const { return _type; }
const std::vector<SynapseId>& Neuron::get_outgoing_synapses() const { return _outgoing_synapses; }
const std::vector<SynapseId>& Neuron::get_incoming_synapses() const { return _incoming_synapses; }
std::vector<SynapseId>& Neuron::get_outgoing_synapses_for_modification() { return _outgoing_synapses; }
std::vector<SynapseId>& Neuron::get_incoming_synapses_for_modification() { return _incoming_synapses; }
double Neuron::get_last_spike_time() const { return _last_spike_time; }
double Neuron::get_avg_firing_rate() const { return _avg_firing_rate_hz; }
bool Neuron::can_sprout() const { return _can_sprout; }
bool Neuron::can_be_target() const { return _can_be_target; } 
const std::array<double, 3>& Neuron::position() const { return _position; }

// --- Methods ---
void Neuron::add_incoming_synapse(SynapseId s_id) { _incoming_synapses.push_back(s_id); }
void Neuron::add_outgoing_synapse(SynapseId s_id) { _outgoing_synapses.push_back(s_id); }

void Neuron::receive_spike(double delta_g, NeuronType pre_type) {
    if (pre_type == NeuronType::Excitatory) _g_excitatory_ns += delta_g;
    else _g_inhibitory_ns += delta_g;
}

void Neuron::apply_external_current(double strength_nA) {
    _external_current_nA += strength_nA;
}

bool Neuron::update(double current_time, double dt, double noise_current_nA) {
    bool has_fired = false;
    double total_external_current = _external_current_nA + noise_current_nA;

    if (current_time - _last_spike_time < _config.REFRACTORY_PERIOD_MS) {
        _membrane_vm = _config.RESTING_POTENTIAL_MV;
    } else {
        if ((_model == NeuronModel::LIF && _membrane_vm >= _spike_threshold_mv) ||
            (_model == NeuronModel::Izhikevich && _membrane_vm >= _config.IZH_SPIKE_PEAK_MV)) {
            fire(current_time);
            has_fired = true;
        } else {
            if (_model == NeuronModel::LIF) update_lif(dt, total_external_current);
            else update_izhikevich(dt, total_external_current);
        }
    }
    
    _external_current_nA = 0.0;

    double synaptic_decay_factor = std::exp(-dt / _config.SYNAPTIC_TIME_CONSTANT_MS);
    _g_excitatory_ns *= synaptic_decay_factor;
    _g_inhibitory_ns *= synaptic_decay_factor;
    
    double decay = std::exp(-dt / _config.HOMEOSTASIS_TAU_MS);
    _avg_firing_rate_hz *= decay;
    if (has_fired) {
        _avg_firing_rate_hz += (1.0 / (_config.HOMEOSTASIS_TAU_MS / 1000.0));
    }
    
    if (_is_homeostatic) {
        update_homeostasis(dt);
    }
    
    return has_fired;
}

void Neuron::update_lif(double dt, double I_external) {
    double I_syn = _g_excitatory_ns * (_config.E_EXCITATORY_MV - _membrane_vm) + _g_inhibitory_ns * (_config.E_INHIBITORY_MV - _membrane_vm);
    double I_total = I_syn + I_external;
    double dV = (dt / _config.MEMBRANE_TIME_CONSTANT_MS) * ((_config.RESTING_POTENTIAL_MV - _membrane_vm) + I_total);
    _membrane_vm += dV;
}

void Neuron::update_izhikevich(double dt, double I_external) {
    double I_syn = _g_excitatory_ns * (_config.E_EXCITATORY_MV - _membrane_vm) + _g_inhibitory_ns * (_config.E_INHIBITORY_MV - _membrane_vm);
    double I_total = I_syn + I_external;
    _membrane_vm += dt * (0.04 * _membrane_vm * _membrane_vm + 5 * _membrane_vm + 140 - _izh_u + I_total);
    _izh_u += dt * (_config.IZH_A * (_config.IZH_B * _membrane_vm - _izh_u));
}

// The responsibility for STDP updates is now in the Network class.
void Neuron::fire(double current_time) {
    _last_spike_time = current_time;

    if (_model == NeuronModel::LIF) {
        _membrane_vm = _config.RESTING_POTENTIAL_MV;
    } else { 
        _membrane_vm = _config.IZH_C; 
        _izh_u += _config.IZH_D; 
    }
}

void Neuron::update_homeostasis(double dt) {
    double error = _avg_firing_rate_hz - _config.TARGET_FIRING_RATE_HZ;
    _spike_threshold_mv += error * dt * _config.HOMEOSTASIS_ADAPTATION_RATE;

    double minimum_threshold = _config.RESTING_POTENTIAL_MV + _config.HOMEOSTASIS_MIN_THRESHOLD_MARGIN_MV;
    _spike_threshold_mv = std::max(_spike_threshold_mv, minimum_threshold);
}
