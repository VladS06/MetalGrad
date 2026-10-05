#include <fstream>
#include <iostream>
#include <numbers>
#include <random>

typedef unsigned int uint;

struct Layer {
    uint input_size;
    float* input;
    float* d_input;
    float* activation;
    Layer(uint input_size): input_size(input_size) {
        input = new float[input_size]();
        d_input = new float[input_size]();
        activation = new float[input_size]();
    }
    virtual void forward(float* input_data) = 0;
    virtual void backward_pass(float* true_labels) = 0;
    virtual void update_parameters(float learning_rate, uint batch_size) = 0;
    virtual ~Layer() {
        delete[] input;
        delete[] d_input;
        delete[] activation;
    }
};

struct DenseLayer:  Layer {
    uint output_size;
    float* biases;
    float* weights;
    float* output;
    float* d_weights;
    float* d_output;
    float* d_biases;
    bool is_hidden;
    DenseLayer(uint input_size, uint output_size,bool is_hidden): Layer(input_size), output_size(output_size), is_hidden(is_hidden) {
        biases = new float[output_size];
        for (int i = 0; i < output_size; i++)
            biases[i] = 0.1f;
        weights = new float[input_size * output_size]();
        output = new float[output_size]();
        d_weights = new float[input_size * output_size]();
        d_output = new float[output_size]();
        d_biases = new float[output_size]();
    }
    ~DenseLayer() override{

        delete[] biases;
        delete[] weights;
        delete[] output;
        delete[] d_weights;
        delete[] d_output;
        delete[] d_biases;
    }

    void initialize_weights() const {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution <float>nd(0.0f, 0.1f);
        for (uint i = 0; i < input_size * output_size; i++) {
            weights[i] = nd(generator);
        }
    }

    void forward(float* input_data) override {
        for (uint i = 0; i < input_size; i++)
            input[i] = input_data[i];
        for (uint j = 0; j < output_size; j++) {
            activation[j] = 0.0f;
            output[j] = biases[j];
            for (uint i = 0; i < input_size; i++) {
                output[j] += input[i] * weights[i* output_size+ j];
            }
            if (is_hidden)
                activation[j] = output[j] > 0 ? output[j] : 0;
            else
                activation[j] = output[j];
        }
    }

    float compute_loss(float* true_labels) {
        float loss = 0.0f;
        for (uint i = 0; i < output_size; i++) {
            loss += (activation[i] - true_labels[i])*(activation[i] - true_labels[i]);
        }
        return loss / output_size;
    }
    void backward_pass(float* incoming_gradients) override {
        for (unsigned int i = 0; i < output_size; i++) {
            if (is_hidden)
                d_output[i] = output[i] > 0.0f ? incoming_gradients[i] : 0.0f;
            else
                d_output[i] = incoming_gradients[i];
        }
        for (uint i = 0; i < input_size; i++) {
            for (uint j = 0; j < output_size; j++) {
                d_weights[i * output_size + j] += d_output[j] * input[i];
            }
        }
        for (uint j = 0; j < output_size; j++) {
            d_biases[j] += d_output[j];
        }
        for (uint i = 0; i < input_size; i++) {
            d_input[i] = 0.0f;
            for (uint j = 0; j < output_size; j++) {
                d_input[i] += d_output[j] * weights[i * output_size + j];
            }
        }
    }
    void update_parameters(float learning_rate, uint batch_size) override {
        float step_size = 1.0f / batch_size * learning_rate;
        for (uint i = 0; i < input_size; i++) {
            for (uint j = 0; j < output_size; j++) {
                weights[i * output_size + j] -= step_size * d_weights[i * output_size + j];
                d_weights[i*output_size + j] = 0.0f;
            }
        }
        for (uint j = 0; j < output_size; j++) {
            biases[j] -= step_size * d_biases[j];
            d_biases[j] = 0.0f;
        }
    }
};

struct SoftmaxLayer: Layer{
    SoftmaxLayer(uint input_size): Layer(input_size){}

    void forward(float* input_data) override {
        float maxInput = input_data[0];
        for (uint i = 0; i < input_size; i++) {
            input[i] = input_data[i];
            if (input[i] > maxInput)
                maxInput = input[i];
        }
        float sum = 0.0f;
        for (uint i = 0; i < input_size; i++) {
            float exp_val = std::exp(input[i]-maxInput);
            sum += exp_val;
            activation[i] = exp_val;
        }
        float norm = 1.0f / sum;
        for (uint i = 0; i < input_size; i++) {
            activation[i] *= norm;
        }
    }
    void backward_pass(float* true_labels) override {
        for (uint i = 0; i < input_size; i++) {
            d_input[i] = activation[i] - true_labels[i];
        }
    }
    void update_parameters(float learning_rate, uint batch_size) override {}
};

struct NeuralNetwork {
    std::vector<Layer*> layers;
    void forward(float* initial_input) {
        float* current_input = initial_input;
        for (auto layer: layers) {
            layer->forward(current_input);
            current_input = layer->activation;
        }
    }
    void backward(float* true_labels) {
        float* curr_input = true_labels;
        for(auto it = layers.rbegin(); it != layers.rend(); it++) {
            (*it)->backward_pass(curr_input);
            curr_input = (*it)->d_input;
        }
    }
    void train_on_batch(float** batch_inputs, float ** batch_labels, uint batch_size, float learning_rate) {
        for (uint i = 0; i < batch_size; i++) {
            forward(batch_inputs[i]);
            backward(batch_labels[i]);
        }
        for (auto layer: layers) {
            layer->update_parameters(learning_rate, batch_size);
        }
    }
    float* predict(float* single_input) {
        forward(single_input);
        return layers[layers.size()-1]->activation;
    }
};

uint32_t flip_endian(unsigned char* bytes) {
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) | ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3];
}

float** load_mnist_images(const char* filename, uint32_t& out_num_images) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "FATAL: Could not open file " << filename << '\n';
        exit(1);
    }
    unsigned char buffer[4];
    file.read((char*)buffer, 4);
    file.read((char*)buffer, 4);
    out_num_images = flip_endian(buffer);
    file.read((char*)buffer, 4);
    file.read((char*)buffer, 4);
    float** images = new float*[out_num_images];
    for (uint i = 0; i < out_num_images; i++) {
        images[i] = new float[784];
        unsigned char pixel_buffer[784];
        file.read((char*)pixel_buffer, 784);
        float div = 1.0f / 255.0f;
        for (int j = 0; j < 784; j++)
            images[i][j] = pixel_buffer[j] * div;
    }
    return images;
}

float** load_mnist_labels(const char* filename, uint32_t& out_num_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "FATAL: Could not open file " << filename << '\n';
        exit(1);
    }
    unsigned char buffer[4];
    file.read((char*)buffer, 4);
    file.read((char*)buffer, 4);
    out_num_labels = flip_endian(buffer);
    float** labels = new float*[out_num_labels];
    unsigned char digit;
    for (unsigned int i = 0; i < out_num_labels; i++) {
        labels[i] = new float[10]();
        file.read((char*)&digit, 1);
        labels[i][digit] = 1.0f;
    }
    return labels;
}

float calculate_accuracy(NeuralNetwork& nn, float** images, float ** labels, uint32_t num_items) {
    uint correct = 0;
    for (uint i = 0; i < num_items;i++) {
        float* pred = nn.predict(images[i]);
        float max = pred[0];
        uint prediction_index = 0;
        uint correct_index = 0;
        for (uint j = 0; j < 10; j++) {
            if (max < pred[j]) {
                max = pred[j];
                prediction_index = j;
            }
            if (labels[i][j] == 1.0f)
                correct_index = j;
        }
        if (prediction_index == correct_index) {
            correct++;
        }
    }
    return (float)correct / (float)num_items;
}

int main() {
    uint32_t num_images;
    float** images = load_mnist_images("/Users/vlad/PersonalProjects/AI/NeuralNetwork/train-images-idx3-ubyte", num_images);
    uint32_t num_labels;
    float** labels = load_mnist_labels("/Users/vlad/PersonalProjects/AI/NeuralNetwork/train-labels-idx1-ubyte", num_labels);
    if (num_images != num_labels) {
        std::cerr << "FATAL: Data mismatch! Images: " << num_images << " Labels: " << num_labels << '\n';
        return 1;
    }
    std::cout << "Successfully loaded " << num_labels << " pairs.\n";
    NeuralNetwork nn;
    DenseLayer* l1 = new DenseLayer(784, 128, true);
    l1->initialize_weights();
    nn.layers.push_back(l1);

    DenseLayer* l2 = new DenseLayer(128, 10, false);
    l2->initialize_weights();
    nn.layers.push_back(l2);

    SoftmaxLayer* l3 = new SoftmaxLayer(10);
    nn.layers.push_back(l3);
    for (uint i = 0; i < 10; i++) {
        for (uint j = 0; j < num_images; j = j + 128) {
            nn.train_on_batch(images + j, labels + j, std::min((uint)128, (num_images - j)), 0.01f);
        }
        float acc = calculate_accuracy(nn, images, labels, num_images);
        std::cout << "--- Epoch " << i << " ---\nAccuracy: " << acc * 100.0f << "%\n\n";
    }

    return 0;
}
