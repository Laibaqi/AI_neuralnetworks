#include"core_engine.h"
#include<cmath>
#include<cstdlib>
#include<ostream>
#include<stdexcept>
#include<utility>

Matrix::Matrix(int r, int c) : rows(r), cols(c), data(r, std::vector<double>(c, 0.0)) {}

void Matrix::randomize() {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
}

Matrix Matrix::multiply(const Matrix& a, const Matrix& b) {
    Matrix result(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < b.cols; j++)
            for (int k = 0; k < a.cols; k++)
                result.data[i][j] += a.data[i][k] * b.data[k][j];
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[j][i] = data[i][j];
    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols)
        throw std::runtime_error("Matrix::operator+ size mismatch");
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[i][j] = data[i][j] + other.data[i][j];
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols)
        throw std::runtime_error("Matrix::operator- size mismatch");
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[i][j] = data[i][j] - other.data[i][j];
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows)
        throw std::runtime_error("Matrix::operator* inner dimension mismatch");
    return Matrix::multiply(*this, other);
}

// epsilon comparison so floating point rounding doesn't cause false mismatches
bool Matrix::operator==(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) return false;
    const double epsilon = 1e-9;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (std::fabs(data[i][j] - other.data[i][j]) > epsilon) return false;
    return true;
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    os << "Matrix(" << m.rows << "x" << m.cols << ")\n";
    for (int i = 0; i < m.rows; i++) {
        os << "[ ";
        for (int j = 0; j < m.cols; j++)
            os << m.data[i][j] << " ";
        os << "]\n";
    }
    return os;
}

Layer::Layer() : last_input(0, 0) {}
Layer::~Layer() {}

DenseLayer::DenseLayer(int in, int out) : weights(in, out), bias(1, out) {
    weights.randomize();
    bias.randomize();
}

std::string DenseLayer::get_type() { return "Dense"; }

Matrix DenseLayer::forward(const Matrix& input) {
    last_input = input;
    Matrix output = Matrix::multiply(input, weights);
    // add bias to every output node
    for (int j = 0; j < output.cols; j++)
        output.data[0][j] += bias.data[0][j];
    return output;
}

Matrix DenseLayer::backward(const Matrix& grad, double lr) {
    Matrix input_T    = last_input.transpose();
    Matrix weights_grad = Matrix::multiply(input_T, grad);
    Matrix weights_T  = weights.transpose();
    Matrix input_grad = Matrix::multiply(grad, weights_T);

    // gradient descent update
    for (int i = 0; i < weights.rows; i++)
        for (int j = 0; j < weights.cols; j++)
            weights.data[i][j] -= lr * weights_grad.data[i][j];

    for (int j = 0; j < bias.cols; j++)
        bias.data[0][j] -= lr * grad.data[0][j];

    return input_grad;
}

void DenseLayer::set_weights(const std::vector<std::vector<double>>& new_weights) {
    weights.data = new_weights;
}

std::string SigmoidLayer::get_type() { return "Sigmoid"; }

// sigmoid: squashes any value into (0, 1)
Matrix SigmoidLayer::forward(const Matrix& input) {
    last_input = input;
    Matrix result(input.rows, input.cols);
    for (int i = 0; i < input.rows; i++)
        for (int j = 0; j < input.cols; j++)
            result.data[i][j] = 1.0 / (1.0 + exp(-input.data[i][j]));
    return result;
}

// sigmoid derivative: s * (1 - s)
Matrix SigmoidLayer::backward(const Matrix& grad, double lr) {
    Matrix result(grad.rows, grad.cols);
    for (int i = 0; i < grad.rows; i++)
        for (int j = 0; j < grad.cols; j++) {
            double s = 1.0 / (1.0 + exp(-last_input.data[i][j]));
            result.data[i][j] = grad.data[i][j] * (s * (1.0 - s));
        }
    return result;
}

void NeuralNetwork::add_layer(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

const std::vector<std::unique_ptr<Layer>>& NeuralNetwork::get_layers() const {
    return layers;
}

Matrix NeuralNetwork::predict(Matrix input) {
    for (auto& layer : layers)
        input = layer->forward(input);
    return input;
}

void NeuralNetwork::train(const Matrix& input, const Matrix& target, double lr) {
    Matrix output = predict(input);

    // initial gradient, 2 * (prediction - target) from MSE derivative
    Matrix grad = output - target;
    for (int j = 0; j < grad.cols; j++)
        grad.data[0][j] *= 2.0;

    // backpropagation through layers in reverse
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--)
        grad = layers[i]->backward(grad, lr);
}
