#pragma once

#include "core_types.h"
#include <string>
#include <fstream>
#include <sstream>

// Forward declarations
class Neuron;
struct Synapse;
struct Config;

class DataLogger {
private:
    std::ofstream _log_file;

    std::stringstream _buffer;
    size_t _buffer_line_count = 0;
    const size_t _flush_threshold_lines;

    void flush();

public:
    explicit DataLogger(const std::string& filename, size_t flush_threshold = 2000);
    ~DataLogger();

    void log_parameters(const Config& config);
    void write_data_header();

    void log_spike(double time, NeuronId id, const std::string& name);
    void log_neuron_vm(double time, const Neuron& n);
    void log_synapse_weight(double time, SynapseId id, const std::string& pre_name, const std::string& post_name, const Synapse& s);
    void log_dopamine(double time, double level);
    void log_pattern_mapping(const std::string& pattern_name, const std::vector<NeuronId>& neuron_ids);
    void log_synapse_counts(double time, const std::string& group_name, int plastic_synapse_count);
};