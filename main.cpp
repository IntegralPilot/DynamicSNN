#include "experiment.h"
#include "config.h"
#include "json.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

int main() {
    const std::string config_filename = "config.json";
    Config config;

    std::ifstream config_file(config_filename);
    if (config_file.is_open()) {
        std::cout << "📄 Loading configuration from " << config_filename << "...\n";
        json j;
        config_file >> j;
        config = Config::from_json(j);
    } else {
        std::cout << "⚠️ " << config_filename << " not found. Creating a default config file.\n";
        std::ofstream out_config_file(config_filename);
        out_config_file << std::setw(4) << config.to_json() << std::endl;
    }

    Experiment experiment(config);
    experiment.run();
    
    std::cout << "\nSimulation finished. Data saved to 'simulation_log.csv'.\n";
    std::cout << "You can now analyze this file, for example, using Python with pandas and matplotlib.\n";
    std::cout << "To change parameters, edit '" << config_filename << "' and run again.\n";

    return 0;
}