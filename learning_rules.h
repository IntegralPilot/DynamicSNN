#pragma once

#include "core_types.h"

// Forward declare to avoid including the full synapse.h in this header
struct Synapse;

class ILearningRule {
public:
    virtual ~ILearningRule() = default;
    virtual void process_spike_timing(Synapse& synapse, const LearningContext& ctx, const Config& config) = 0;
};

class RewardModulatedSTDP : public ILearningRule {
public:
    void process_spike_timing(Synapse& synapse, const LearningContext& ctx, const Config& config) override;
};