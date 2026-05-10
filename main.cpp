#include"core_engine.h"
#include"pipeline.h"
#include"interface.h"
#include<cstdlib>
#include<ctime>
#include<iostream>
#include<memory>

int main() {
    try {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));

        // load heart disease dataset: 16 features, 1 binary label (0=healthy, 1=disease)
        TrainingData data;
        data.load_csv("heart_disease_normalized.csv", 16, 1, true);
        std::cout << "Loaded samples: " << data.size() << std::endl;

        // build the network: 
        NeuralNetwork network;
        network.add_layer(std::unique_ptr<Layer>(new DenseLayer(16, 16)));
        network.add_layer(std::unique_ptr<Layer>(new SigmoidLayer()));
        network.add_layer(std::unique_ptr<Layer>(new DenseLayer(16, 8)));
        network.add_layer(std::unique_ptr<Layer>(new SigmoidLayer()));
        network.add_layer(std::unique_ptr<Layer>(new DenseLayer(8, 1)));
        network.add_layer(std::unique_ptr<Layer>(new SigmoidLayer()));

        WeightInitializer::xavier_network(network, 42);
        std::cout << "Network ready, launching UI" << std::endl;

        SimulationUI ui(network, data);
        ui.run();

        std::cout << "\nSimulation closed\n";

    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
