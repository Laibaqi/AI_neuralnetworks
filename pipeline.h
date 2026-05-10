#ifndef PIPELINE_H
#define PIPELINE_H
#include"core_engine.h"
#include<fstream>
#include<string>
#include<vector>

// one row from the dataset: 16 inputs + 1 target label
struct TrainingSample {
    Matrix input;
    Matrix target;

    TrainingSample(const Matrix& in, const Matrix& out);
};

// loads and stores the dataset from a CSV file
class TrainingData {
private:
    std::vector<TrainingSample> samples;
    int input_cols;
    int target_cols;

    static std::string trim(const std::string& value);
    static std::vector<double> parse_csv_row(const std::string& line, int row_number);

public:
    TrainingData();

    void load_csv(const std::string& filename, int input_count, int target_count, bool has_header = false);
    int size() const;
    int input_count() const;
    int target_count() const;
    const std::vector<TrainingSample>& get_samples() const;
};

// runs training epochs and tracks per epoch loss
class TrainingPipeline {
private:
    static double sample_loss(NeuralNetwork& network, const TrainingSample& sample);

public:
    static std::vector<double> train(
        NeuralNetwork& network,
        const TrainingData& data,
        int epochs,
        double learning_rate
    );
};

// initializes network weights, Xavier keeps gradients from vanishing early on
class WeightInitializer {
private:
    static DenseLayer* as_dense_layer(Layer* layer, int layer_number);
    static void fill_uniform(Matrix& matrix, double min_value, double max_value, unsigned int seed);

public:
    static void uniform(DenseLayer& layer, double min_value, double max_value, unsigned int seed = 1);
    static void xavier(DenseLayer& layer, unsigned int seed = 1);
    static void xavier_network(NeuralNetwork& network, unsigned int seed = 1);
};

// saves and loads model weights to a file so training isn't lost on exit
class ModelIO {
private:
    static DenseLayer* as_dense_layer(Layer* layer, int layer_number);
    static void write_matrix(std::ofstream& file, const Matrix& matrix);
    static void read_matrix(std::ifstream& file, Matrix& matrix, const std::string& label);

public:
    static void save_model(const NeuralNetwork& network, const std::string& filename);
    static void load_model(NeuralNetwork& network, const std::string& filename);
};

#endif
