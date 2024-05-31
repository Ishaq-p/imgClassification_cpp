#include "Matrix.cpp"
#include "ConLayer.cpp"
#include "FClayer.cpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include<cstdlib>


void backwardConvLayer(const std::vector<Matrix>& input, const std::vector<Matrix>& d_L_d_out, ConLayer& conv_layer, float learning_rate) {
    int filter_size = conv_layer.filterSize;
    int out_filters_num = conv_layer.outFiltersNum;
    int in_filters_num = conv_layer.inFiltersNum;

    std::vector<std::vector<Matrix>> d_L_d_filters(out_filters_num, std::vector<Matrix>(in_filters_num, Matrix(filter_size, filter_size)));
    Matrix d_L_d_biases(out_filters_num, 1);

    for (int out_f = 0; out_f < out_filters_num; ++out_f) {
        for (int in_f = 0; in_f < in_filters_num; ++in_f) {
            for (int i = 0; i < d_L_d_out[0].rows; ++i) {
                for (int j = 0; j < d_L_d_out[0].cols; ++j) {
                    for (int fi = 0; fi < filter_size; ++fi) {
                        for (int fj = 0; fj < filter_size; ++fj) {
                            d_L_d_filters[out_f][in_f].data[fi][fj] += d_L_d_out[out_f].data[i][j] * input[in_f].data[i + fi][j + fj];
                        }
                    }
                }
            }
        }
        d_L_d_biases.data[out_f][0] = d_L_d_out[out_f].rows * d_L_d_out[out_f].cols;
    }

    for (int out_f = 0; out_f < out_filters_num; ++out_f) {
        for (int in_f = 0; in_f < in_filters_num; ++in_f) {
            for (int i = 0; i < filter_size; ++i) {
                for (int j = 0; j < filter_size; ++j) {
                    conv_layer.filters[out_f][in_f].data[i][j] -= learning_rate * d_L_d_filters[out_f][in_f].data[i][j];
                }
            }
        }
        conv_layer.cnn_bias.data[out_f][0] -= learning_rate * d_L_d_biases.data[out_f][0];
    }
}


class CNN {
public:
    ConLayer conv_layer;
    FClayer fc_layer;

    CNN(ConLayer convLayer, FClayer fcLayer) 
        : conv_layer(convLayer), fc_layer(fcLayer) {}

    std::vector<float> forward(const std::vector<Matrix>& input) {
        std::vector<Matrix> conv_output = conv_layer.forward(input);

        std::vector<float> fc_input;
        for (const Matrix& mat : conv_output) {
            for (const auto& row : mat.data) {
                fc_input.insert(fc_input.end(), row.begin(), row.end());
            }
        }

        std::vector<float> fc_output(fc_layer.biases.size());
        return fc_layer.forward(fc_input, fc_output);
    }

    void train(const std::vector<std::vector<Matrix>>& inputs, const std::vector<std::vector<float>>& labels, int epochs, float learning_rate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            for (size_t i = 0; i < inputs.size(); ++i) {
                // Forward pass
                std::vector<float> predicted = forward(inputs[i]);

                // Compute loss
                float loss = crossEntropyLoss(predicted, labels[i]);
                total_loss += loss;

                // Backward pass
                std::vector<float> d_L_d_out(predicted.size());
                for (size_t j = 0; j < predicted.size(); ++j) {
                    d_L_d_out[j] = predicted[j] - labels[i][j];
                }

                // Gradients for fully connected layer
                Matrix fc_weight_gradients(fc_layer.weights.rows, fc_layer.weights.cols);
                std::vector<float> fc_bias_gradients(fc_layer.biases.size());

                for (int j = 0; j < fc_layer.weights.rows; ++j) {
                    fc_bias_gradients[j] = d_L_d_out[j];
                    for (int k = 0; k < fc_layer.weights.cols; ++k) {
                        fc_weight_gradients.data[j][k] = d_L_d_out[j] * fc_layer.weights.data[j][k];
                    }
                }

                // Update weights and biases
                updateWeights(fc_layer.weights, fc_weight_gradients, learning_rate);
                updateBiases(fc_layer.biases, fc_bias_gradients, learning_rate);

                // Backward pass for convolutional layer
                std::vector<Matrix> conv_input = inputs[i];
                std::vector<Matrix> conv_output = conv_layer.forward(conv_input);

                std::vector<Matrix> d_L_d_conv_out(conv_layer.outFiltersNum, Matrix(conv_output[0].rows, conv_output[0].cols));
                for (int j = 0; j < conv_layer.outFiltersNum; ++j) {
                    for (int r = 0; r < conv_output[0].rows; ++r) {
                        for (int c = 0; c < conv_output[0].cols; ++c) {
                            d_L_d_conv_out[j].data[r][c] = d_L_d_out[j] * (conv_output[j].data[r][c] > 0 ? 1 : 0); // ReLU derivative
                        }
                    }
                }

                backwardConvLayer(conv_input, d_L_d_conv_out, conv_layer, learning_rate);
            }
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / inputs.size() << std::endl;
        }
    }

    float crossEntropyLoss(const std::vector<float>& predictions, const std::vector<float>& labels) {
        float loss = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            loss -= labels[i] * log(predictions[i]);
        }
        return loss;
    }

    void updateWeights(Matrix& weights, const Matrix& gradients, float learning_rate) {
        for (int i = 0; i < weights.rows; ++i) {
            for (int j = 0; j < weights.cols; ++j) {
                weights.data[i][j] -= learning_rate * gradients.data[i][j];
            }
        }
    }

    void updateBiases(std::vector<float>& biases, const std::vector<float>& gradients, float learning_rate) {
        for (size_t i = 0; i < biases.size(); ++i) {
            biases[i] -= learning_rate * gradients[i];
        }
    }
};


