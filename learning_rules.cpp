// === learning_rules.cpp ===
#include "learning_rules.h"
#include "synapse.h" // Needed for the Synapse& parameter
#include "config.h"  // Needed for the Config& parameter
#include <cmath>     // For std::exp

void RewardModulatedSTDP::process_spike_timing(Synapse& synapse, const LearningContext& ctx, const Config& config) {
    // This function is called from two places:
    // 1. From Network::run after a PRE-synaptic spike. ctx.post_spike_time is the last known post-spike.
    // 2. From Neuron::fire after a POST-synaptic spike. ctx.pre_spike_time is the last known pre-spike.
    
    // We only care about the case where we have a new pair to evaluate.
    // The context times will be < 0 if one side of the pair hasn't happened yet.
    if (ctx.pre_spike_time < 0 || ctx.post_spike_time < 0) return;

    double delta_t = ctx.post_spike_time - ctx.pre_spike_time;
    double eligibility_change = 0.0;

    if (delta_t > 0 && delta_t < config.STDP_WINDOW_MS) {
        // LTP: Pre-synaptic spike happened BEFORE post-synaptic spike (causal)
        eligibility_change = config.LTP_RATE * std::exp(-delta_t / config.STDP_WINDOW_MS);
    } else if (delta_t < 0 && delta_t > -config.STDP_WINDOW_MS) {
        // LTD: Post-synaptic spike happened BEFORE pre-synaptic spike (anti-causal)
        // Note: config.LTD_RATE is negative, so this adds a negative value.
        eligibility_change = config.LTD_RATE * std::exp(delta_t / config.STDP_WINDOW_MS);
    }

    if (std::abs(eligibility_change) > 0) {
        synapse.eligibility_trace += eligibility_change;
    }
}