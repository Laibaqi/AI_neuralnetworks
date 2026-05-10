#include "pipeline.h"
#include<cctype>
#include<cmath>
#include<iomanip>
#include<iostream>
#include<queue>
#include<random>
#include<sstream>
#include<stdexcept>
#include<utility>
#include<vector>

TrainingSample::TrainingSample(const Matrix& in, const Matrix& out) : input(in), target(out) {}

std::string TrainingData::trim(const std::string& value) {
    int start = 0;
    int end = static_cast<int>(value.size()) - 1;

    while (start <= end && std::isspace(static_cast<unsigned char>(value[start]))) start++;
    while (end >= start && std::isspace(static_cast<unsigned char>(value[end]))) end--;

    if (start > end) return "";
    return value.substr(start, end - start + 1);
}

std::vector<double> TrainingData::parse_csv_row(const std::string& line, int row_number) {
    std::vector<double> values;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
        cell = trim(cell);
        if (cell.empty())
            throw std::runtime_error("Empty value found in CSV row " + std::to_string(row_number));

        try {
            size_t used_chars = 0;
            double number = std::stod(cell, &used_chars);
            if (used_chars != cell.size())
                throw std::runtime_error("extra characters");
            values.push_back(number);
        } catch (...) {
            throw std::runtime_error("Non numeric value found in CSV row " + std::to_string(row_number));
        }
    }

    return values;
}

TrainingData::TrainingData() : input_cols(0), target_cols(0) {}

void TrainingData::load_csv(const std::string& filename, int input_count, int target_count, bool has_header) {
    if (input_count <= 0 || target_count <= 0)
        throw std::invalid_argument("Input and target column counts must be positive");

    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open dataset file: " + filename);

    samples.clear();
    input_cols  = input_count;
    target_cols = target_count;

    std::string line;
    int row_number = 0;
    const int expected_cols = input_cols + target_cols;

    while (std::getline(file, line)) {
        row_number++;
        if (has_header && row_number == 1) continue;  // skip header row
        if (trim(line).empty()) continue;

        std::vector<double> values = parse_csv_row(line, row_number);
        if (static_cast<int>(values.size()) != expected_cols)
            throw std::runtime_error(
                "CSV row " + std::to_string(row_number) + " has " +
                std::to_string(values.size()) + " columns, expected " +
                std::to_string(expected_cols)
            );

        Matrix input(1, input_cols);
        Matrix target(1, target_cols);

        for (int i = 0; i < input_cols; i++)
            input.data[0][i] = values[i];
        for (int j = 0; j < target_cols; j++)
            target.data[0][j] = values[input_cols + j];

        samples.push_back(TrainingSample(input, target));
    }

    if (samples.empty())
        throw std::runtime_error("Dataset is empty after loading: " + filename);
}

int TrainingData::size()         const { return static_cast<int>(samples.size()); }
int TrainingData::input_count()  const { return input_cols; }
int TrainingData::target_count() const { return target_cols; }

const std::vector<TrainingSample>& TrainingData::get_samples() const { return samples; }

double TrainingPipeline::sample_loss(NeuralNetwork& network, const TrainingSample& sample) {
    Matrix prediction = network.predict(sample.input);

    if (prediction.cols != sample.target.cols)
        throw std::runtime_error("Prediction size does not match target size");

    double total = 0.0;
    for (int j = 0; j < prediction.cols; j++) {
        double diff = prediction.data[0][j] - sample.target.data[0][j];
        total += diff * diff;
    }
    return total / prediction.cols;
}

std::vector<double> TrainingPipeline::train(
    NeuralNetwork& network,
    const TrainingData& data,
    int epochs,
    double learning_rate
) {
    if (epochs <= 0)
        throw std::invalid_argument("Epoch count must be positive");
    if (learning_rate <= 0.0)
        throw std::invalid_argument("Learning rate must be positive");

    std::vector<double> loss_history;
    const std::vector<TrainingSample>& samples = data.get_samples();

    // max heap: worst of the top 3 is at the top, pop it when we go over 3
    using EpochEntry = std::pair<double, int>;
    std::priority_queue<EpochEntry> best_epochs;
    const int K = 3;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (const TrainingSample& sample : samples) {
            network.train(sample.input, sample.target, learning_rate);
            total_loss += sample_loss(network, sample);
        }

        double average_loss = total_loss / samples.size();
        loss_history.push_back(average_loss);

        best_epochs.push(EpochEntry(average_loss, epoch + 1));
        if (static_cast<int>(best_epochs.size()) > K)
            best_epochs.pop();  // drop the worst of the current best 3

        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                  << " - loss: " << average_loss << std::endl;
    }

    // only print the summary for real batch runs
    if (epochs > 1) {
        std::vector<EpochEntry> ranked;
        while (!best_epochs.empty()) {
            ranked.push_back(best_epochs.top());
            best_epochs.pop();
        }
        std::cout << "\n=== Top " << ranked.size() << " Best Epochs (lowest loss) ===" << std::endl;
        for (int i = static_cast<int>(ranked.size()) - 1; i >= 0; i--)
            std::cout << "  Epoch " << ranked[i].second
                      << " | loss: " << std::fixed << std::setprecision(6)
                      << ranked[i].first << std::endl;
    }

    return loss_history;
}

DenseLayer* WeightInitializer::as_dense_layer(Layer* layer, int layer_number) {
    DenseLayer* dense = dynamic_cast<DenseLayer*>(layer);
    if (dense == nullptr)
        throw std::runtime_error("Layer " + std::to_string(layer_number) + " is not a DenseLayer");
    return dense;
}

void WeightInitializer::fill_uniform(Matrix& matrix, double min_value, double max_value, unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(min_value, max_value);
    for (int i = 0; i < matrix.rows; i++)
        for (int j = 0; j < matrix.cols; j++)
            matrix.data[i][j] = distribution(generator);
}

void WeightInitializer::uniform(DenseLayer& layer, double min_value, double max_value, unsigned int seed) {
    if (min_value >= max_value)
        throw std::invalid_argument("Minimum weight value must be less than maximum weight value");
    fill_uniform(layer.weights, min_value, max_value, seed);
    fill_uniform(layer.bias,    min_value, max_value, seed + 1);
}

// Xavier: limit = sqrt(6 / (fan_in + fan_out)), keeps gradients from blowing up early on
void WeightInitializer::xavier(DenseLayer& layer, unsigned int seed) {
    if (layer.weights.rows <= 0 || layer.weights.cols <= 0)
        throw std::invalid_argument("DenseLayer weight size must be positive for Xavier initialization");
    double limit = std::sqrt(6.0 / (layer.weights.rows + layer.weights.cols));
    uniform(layer, -limit, limit, seed);
}

void WeightInitializer::xavier_network(NeuralNetwork& network, unsigned int seed) {
    const std::vector<std::unique_ptr<Layer>>& layers = network.get_layers();
    int dense_index = 0;
    for (const auto& layer : layers) {
        if (layer->get_type() != "Dense") continue;  // skip sigmoid layers
        DenseLayer* dense = as_dense_layer(layer.get(), dense_index);
        xavier(*dense, seed + dense_index);
        dense_index++;
    }
}

DenseLayer* ModelIO::as_dense_layer(Layer* layer, int layer_number) {
    DenseLayer* dense = dynamic_cast<DenseLayer*>(layer);
    if (dense == nullptr)
        throw std::runtime_error("Layer " + std::to_string(layer_number) + " is not a DenseLayer");
    return dense;
}

void ModelIO::write_matrix(std::ofstream& file, const Matrix& matrix) {
    file << matrix.rows << " " << matrix.cols << "\n";
    file << std::setprecision(17);
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            file << matrix.data[i][j];
            if (j + 1 < matrix.cols) file << " ";
        }
        file << "\n";
    }
}

void ModelIO::read_matrix(std::ifstream& file, Matrix& matrix, const std::string& label) {
    int rows = 0, cols = 0;
    if (!(file >> rows >> cols))
        throw std::runtime_error("Could not read " + label + " size");

    if (rows != matrix.rows || cols != matrix.cols)
        throw std::runtime_error(
            label + " size mismatch. File has " +
            std::to_string(rows) + "x" + std::to_string(cols) +
            ", network expects " +
            std::to_string(matrix.rows) + "x" + std::to_string(matrix.cols)
        );

    for (int i = 0; i < matrix.rows; i++)
        for (int j = 0; j < matrix.cols; j++)
            if (!(file >> matrix.data[i][j]))
                throw std::runtime_error("Could not read values for " + label);
}

void ModelIO::save_model(const NeuralNetwork& network, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not create model file: " + filename);

    const std::vector<std::unique_ptr<Layer>>& layers = network.get_layers();
    int dense_count = 0;
    for (const auto& layer : layers)
        if (layer->get_type() == "Dense") dense_count++;

    file << "PERSON_B_MODEL_V1\n";
    file << dense_count << "\n";

    int dense_index = 0;
    for (const auto& layer : layers) {
        if (layer->get_type() != "Dense") continue;
        DenseLayer* dense = as_dense_layer(layer.get(), dense_index);
        file << "DENSE " << dense_index << "\n";
        write_matrix(file, dense->weights);
        write_matrix(file, dense->bias);
        dense_index++;
    }
}

void ModelIO::load_model(NeuralNetwork& network, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open model file: " + filename);

    std::string header;
    int dense_count = 0;

    if (!(file >> header) || header != "PERSON_B_MODEL_V1")
        throw std::runtime_error("Invalid model file format");
    if (!(file >> dense_count) || dense_count < 0)
        throw std::runtime_error("Invalid dense layer count in model file");

    const std::vector<std::unique_ptr<Layer>>& layers = network.get_layers();
    int network_dense_count = 0;
    for (const auto& layer : layers)
        if (layer->get_type() == "Dense") network_dense_count++;

    if (network_dense_count != dense_count)
        throw std::runtime_error("Model dense layer count does not match the network");

    int dense_index = 0;
    for (const auto& layer : layers) {
        if (layer->get_type() != "Dense") continue;

        std::string tag;
        int file_dense_index = -1;
        if (!(file >> tag >> file_dense_index) || tag != "DENSE" || file_dense_index != dense_index)
            throw std::runtime_error("Invalid dense layer entry in model file");

        DenseLayer* dense = as_dense_layer(layer.get(), dense_index);
        read_matrix(file, dense->weights, "weights for Dense layer " + std::to_string(dense_index));
        read_matrix(file, dense->bias,    "bias for Dense layer "    + std::to_string(dense_index));
        dense_index++;
    }
}
