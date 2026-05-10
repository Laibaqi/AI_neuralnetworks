#ifndef CORE_ENGINE_H
#define CORE_ENGINE_H
#include<iosfwd>
#include<memory>
#include<string>
#include<vector>

// 2D matrix, everything passing through the network is a Matrix
class Matrix {
public:
    int rows;
    int cols;
    std::vector<std::vector<double>> data;

    Matrix(int r, int c);

    void randomize();
    Matrix transpose() const;

    static Matrix multiply(const Matrix& a, const Matrix& b);

    // operator* does full matrix multiply, +/- are element wise
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    bool   operator==(const Matrix& other) const;  
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);
};

// abstract base, can't use this directly, need DenseLayer or SigmoidLayer
class Layer {
public:
    Matrix last_input;  // saved during forward pass, needed for backpropagation

    Layer();
    virtual ~Layer();

    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& grad, double lr) = 0;
    virtual std::string get_type() = 0;
};

// fully connected layer: output = input * weights + bias
class DenseLayer : public Layer {
public:
    Matrix weights;  // shape: (in, out) — rows=inputs, cols=outputs
    Matrix bias;     // shape: (1, out)

    DenseLayer(int in, int out);

    std::string get_type() override;
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad, double lr) override;
    void set_weights(const std::vector<std::vector<double>>& new_weights);
};

// applies sigmoid element wise, no learnable parameters
class SigmoidLayer : public Layer {
public:
    std::string get_type() override;
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad, double lr) override;
};

// holds the layer stack, runs forward and backward passes
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    void add_layer(std::unique_ptr<Layer> layer);
    const std::vector<std::unique_ptr<Layer>>& get_layers() const;
    Matrix predict(Matrix input);
    void train(const Matrix& input, const Matrix& target, double lr);
};

#endif
