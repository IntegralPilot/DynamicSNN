// FILE: config.h
#pragma once

#include <optional>
#include <string>

// Include the json header here, as its type is used in the function signatures
#include "json.hpp"
using json = nlohmann::json;

struct Config {
    // --- Neuron Properties ---
    double RESTING_POTENTIAL_MV;
    double SPIKE_THRESHOLD_MV;
    double MEMBRANE_TIME_CONSTANT_MS;
    double REFRACTORY_PERIOD_MS;

    // --- Izhikevich Model Parameters ---
    double IZH_A, IZH_B, IZH_C, IZH_D;
    double IZH_SPIKE_PEAK_MV;

    // --- Synapse Properties ---
    double E_EXCITATORY_MV, E_INHIBITORY_MV;
    double EXCITATORY_CONDUCTANCE_SCALE_NS, INHIBITORY_CONDUCTANCE_SCALE_NS;
    double SYNAPTIC_TIME_CONSTANT_MS;

    // --- Short-Term Plasticity (STP) Properties ---
    double STP_U, STP_TAU_FACIL_MS, STP_TAU_DECAY_MS;

    // --- Long-Term Plasticity (STDP) Properties ---
    double STDP_WINDOW_MS, LTP_RATE, LTD_RATE;

    // --- Reward-Modulated STDP Properties ---
    double DOPAMINE_BASE_LEVEL, DOPAMINE_DECAY_HALF_LIFE_MS;
    double ELIGIBILITY_TRACE_DECAY_TAU_MS;
    double PUNISHMENT_TRACE_DECAY_TAU_MS;

    // --- Learning & Weight Properties ---
    double LEARNING_RATE;
    double PUNISHMENT_LEARNING_RATE;
    double MAX_WEIGHT;
    double INITIAL_WEIGHT_MIN, INITIAL_WEIGHT_MAX;

    // --- Homeostatic Plasticity Properties ---
    double HOMEOSTASIS_TAU_MS, TARGET_FIRING_RATE_HZ;
    double HOMEOSTASIS_ADAPTATION_RATE;
    double HOMEOSTASIS_MIN_THRESHOLD_MARGIN_MV;

    // --- Simulation Properties ---
    double NOISE_STD_DEV_NA;
    std::optional<unsigned int> SIMULATION_SEED;

    // --- Experiment-specific Properties ---
    int TRAINING_EPOCHS;
    std::optional<int> REVERSAL_EPOCHS;
    double TIME_STEP;
    double TRIAL_DURATION_MS;
    double STIMULUS_DURATION_MS;
    double ITI_MS;
    double STIMULATION_CURRENT_NA;
    double STIMULUS_POISSON_RATE_HZ;
    double STIMULUS_PULSE_DURATION_MS;
    double LOG_INTERVAL_MS;
    double REWARD_AMOUNT_SUCCESS;
    double REINFORCE_AMOUNT_OMISSION;
    double PUNISH_AMOUNT_MISTAKE;
    double PUNISHMENT_TAG_AMOUNT;
    double PUNISHMENT_TAG_WINDOW_MS;
    double UNLEARNING_TAG_AMOUNT;

    // --- Structural plasticity ---
    bool ENABLE_STRUCTURAL_PLASTICITY;
    double STRUCTURAL_UPDATE_INTERVAL_MS;
    double PRUNING_WEIGHT_THRESHOLD;
    double SPROUTING_ACTIVITY_THRESHOLD_HZ;
    double SPROUTING_MAX_DISTANCE;
    double NEW_CONNECTION_PROBABILITY;
    double NEW_SYNAPSE_INITIAL_WEIGHT;
    double PRUNING_PUNISHMENT_THRESHOLD;

    // Default constructor to initialize with default values
    Config();

    // Function declarations
    static Config from_json(const json& j);
    json to_json() const;
};