#include "synapse.h"
#include "config.h"
#include "learning_rules.h"
#include <cmath>
#include <algorithm>

Synapse::Synapse(NeuronId pre, NeuronId post, NeuronType type, double w, double d, const Config& config, std::unique_ptr<ILearningRule> rule)
    : pre_neuron_id(pre), post_neuron_id(post), pre_type(type), weight(w),
      delay_ms(d), stp_u(config.STP_U), learning_rule(std::move(rule)) {}

Synapse::~Synapse() = default;

double Synapse::transmit(double current_time, const Config& config) {
    double last_spike_interval = current_time - last_pre_spike_time;

    // 1. RECOVERY: Recover resources based on time since the last spike.
    // This check is important for the very first spike of the simulation.
    if (last_pre_spike_time > 0) {
        // Recover neurotransmitter resources (x)
        stp_x = 1.0 - (1.0 - stp_x) * std::exp(-last_spike_interval / config.STP_TAU_DECAY_MS);
        // Recover facilitation factor (u) back towards its baseline U
        stp_u = config.STP_U + (stp_u - config.STP_U) * std::exp(-last_spike_interval / config.STP_TAU_FACIL_MS);
    }

    // 2. USAGE: Use resources for the current spike.
    // The strength of this spike's transmission
    double transmission_strength = stp_u * stp_x;
    
    // Update the facilitation variable for the *next* spike
    stp_u = stp_u + config.STP_U * (1.0 - stp_u);
    // Deplete the neurotransmitter resources
    stp_x = stp_x - transmission_strength;
    
    double conductance_scale = (pre_type == NeuronType::Excitatory) ? config.EXCITATORY_CONDUCTANCE_SCALE_NS : config.INHIBITORY_CONDUCTANCE_SCALE_NS;
    // The final conductance change is modulated by the base weight and the dynamic STP factor.
    return weight * transmission_strength * conductance_scale;
}


void Synapse::apply_learning(const Config& config, double current_dopamine_level) {
    if (!learning_rule) return;

    double dopamine_modulation = current_dopamine_level - config.DOPAMINE_BASE_LEVEL;
    double delta_w = 0.0;

    // Use an if/else if structure to prioritize punishment
    if (dopamine_modulation < 0 && std::abs(punishment_trace) > 1e-9) {
       // PUNISHMENT: Use the specific punishment trace.
       // Note: dopamine_modulation is negative, punishment_trace is positive, so the result is negative (LTD).
        delta_w = punishment_trace * dopamine_modulation * config.PUNISHMENT_LEARNING_RATE;
        
        eligibility_trace = 0;

    } else if (dopamine_modulation > 0 && std::abs(eligibility_trace) > 1e-9) {
        // REWARD: Only apply if dopamine is positive
        delta_w = eligibility_trace * dopamine_modulation * config.LEARNING_RATE;
    }

    if (std::abs(delta_w) > 0) {
         weight += delta_w;
         weight = std::max(0.0, std::min(weight, config.MAX_WEIGHT));
     }
}

void Synapse::update(double dt, const Config& config) {
    eligibility_trace *= std::exp(-dt / config.ELIGIBILITY_TRACE_DECAY_TAU_MS);
    punishment_trace *= std::exp(-dt / config.PUNISHMENT_TRACE_DECAY_TAU_MS);
}