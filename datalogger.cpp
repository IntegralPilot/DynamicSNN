#include "datalogger.h"
#include "config.h" // For json and Config
#include "neuron.h"
#include "synapse.h"
#include "json.hpp" // For parameter logging

// Constructor initializes the flush threshold and opens the file.
DataLogger::DataLogger(const std::string& filename, size_t flush_threshold)
    : _flush_threshold_lines(flush_threshold) 
{
    _log_file.open(filename);
}

// Destructor ensures any remaining data in the buffer is written to the file.
DataLogger::~DataLogger() {
    flush(); // CRITICAL: Write any remaining lines before closing.
    if (_log_file.is_open()) {
        _log_file.close();
    }
}

// Private helper to flush the in-memory buffer to the disk.
void DataLogger::flush() {
    // No need to do anything if the file isn't open or the buffer is empty.
    if (!_log_file.is_open() || _buffer_line_count == 0) return;

    // Write the entire buffer's content to the file in one operation.
    // Using rdbuf() is often more efficient than .str() as it can avoid a string copy.
    _log_file << _buffer.rdbuf();
    
    // Clear the buffer for the next batch of logs.
    _buffer.str("");
    _buffer.clear(); // Important to clear error states (like eof) on the stream.
    _buffer_line_count = 0;
}

// Config parameters are not buffered. They are written immediately to the top of the file.
void DataLogger::log_parameters(const Config& config) {
    if (!_log_file.is_open()) return;

    flush(); // Flush any existing data (should be none, but good practice).
    
    json j = config.to_json();
    _log_file << "# --- SIMULATION CONFIGURATION ---\n";
    std::string dump = j.dump(4);
    std::istringstream iss(dump);
    std::string line;
    while(std::getline(iss, line)) {
        _log_file << "# " << line << "\n";
    }
    _log_file << "# --------------------------------\n";
}

void DataLogger::log_pattern_mapping(const std::string& pattern_name, const std::vector<NeuronId>& neuron_ids) {
    if (!_log_file.is_open()) return;

    // This data is small and part of the metadata, so we can write it directly after a flush.
    flush(); 
    _log_file << "# PATTERN_MAPPING," << pattern_name;
    for (NeuronId id : neuron_ids) {
        _log_file << "," << id;
    }
    _log_file << "\n";
}

void DataLogger::write_data_header() {
    if (!_log_file.is_open()) return;
    flush(); // Ensure buffer is flushed before writing the header.
    _log_file << "time_ms,event_type,id,name,value1,value2\n";
}


void DataLogger::log_spike(double time, NeuronId id, const std::string& name) {
    _buffer << time << ",spike," << id << "," << name << ",,\n";
    if (++_buffer_line_count >= _flush_threshold_lines) {
        flush();
    }
}

void DataLogger::log_neuron_vm(double time, const Neuron& n) {
    _buffer << time << ",vm," << n.id() << "," << n.name() << "," << n.vm() << "," << n.threshold() << "\n";
    if (++_buffer_line_count >= _flush_threshold_lines) {
        flush();
    }
}

void DataLogger::log_synapse_weight(double time, SynapseId id, const std::string& pre_name, const std::string& post_name, const Synapse& s) {
    _buffer << time << ",synapse," << id << "," << pre_name << "->" << post_name << "," << s.weight << "," << s.eligibility_trace << "\n";
    if (++_buffer_line_count >= _flush_threshold_lines) {
        flush();
    }
}
    
void DataLogger::log_dopamine(double time, double level) {
    _buffer << time << ",dopamine,0,global," << level << ",\n";
    if (++_buffer_line_count >= _flush_threshold_lines) {
        flush();
    }
}

void DataLogger::log_synapse_counts(double time, const std::string& group_name, int plastic_synapse_count) {
    // Format: time,event_type,id,name,value1,value2
    // We can leave id and value2 blank.
    _buffer << time << ",synapse_count,," << group_name << "," << plastic_synapse_count << ",\n";
    if (++_buffer_line_count >= _flush_threshold_lines) {
        flush();
    }
}