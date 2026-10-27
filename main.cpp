#include <fstream>
#include <iostream>
#include <thread>
#include <random>
#include <valarray>
#include <vector>

typedef unsigned int uint;

constexpr uint IMAGE_SIZE = 784;
constexpr uint NUM_CLASSES = 10;
constexpr uint BATCH_SIZE = 128;
constexpr float LEARNING_RATE = 0.1f;

constexpr uint align_to_cache(uint size) {
    return (size + 15) & ~15;
}

struct Layer {
    uint input_size;
    uint output_size;
    uint max_batch_size;
    std::vector<float> input;
    std::vector<float> output;
    std::vector<float> d_input;
    std::vector<float> activation;

    Layer(uint input_size, uint max_batch_size, uint output_size) :
        max_batch_size(max_batch_size),
        input_size(input_size),
        output_size(output_size),
        input(max_batch_size * input_size, 0.0f),
        output(max_batch_size * align_to_cache(output_size), 0.0f),
        d_input(max_batch_size * input_size, 0.0f),
        activation(max_batch_size * align_to_cache(output_size), 0.0f) {
    }
    virtual void forward(float* input_data, uint batch_index) = 0;
    virtual void backward_pass(float* true_labels, uint batch_index) = 0;
    virtual void update_parameters(float learning_rate, uint batch_size) = 0;
    virtual ~Layer() {}
};

struct DenseLayer : Layer {
    std::vector<float> biases;
    std::vector<float> weights;
    std::vector<float> d_weights;
    std::vector<float> d_output;
    std::vector<float> d_biases;
    uint stride_output;
    bool is_hidden;
    std::vector<float> grad_weights_sum;
    std::vector<float> grad_biases_sum;

    DenseLayer(uint input_size, uint output_size, bool is_hidden, uint max_batch_size) :
        Layer(input_size, max_batch_size, output_size),
        is_hidden(is_hidden),
        stride_output(align_to_cache(output_size)),
        biases(output_size, 0.1f),
        weights(input_size * output_size, 0.0f),
        d_weights(max_batch_size * input_size * output_size, 0.0f),
        d_output(max_batch_size * align_to_cache(output_size), 0.0f),
        d_biases(max_batch_size * align_to_cache(output_size), 0.0f),
        grad_weights_sum(input_size * output_size, 0.0f),
        grad_biases_sum(output_size, 0.0f) {}

    ~DenseLayer() override {}

    void initialize_weights() {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<float> nd(0.0f, 0.1f);
        for (uint i = 0; i < input_size * output_size; i++) {
            weights[i] = nd(generator);
        }
    }

    void forward(float* input_data, uint batch_index) override {
        for (uint i = 0; i < input_size; i++)
            input[batch_index * input_size + i] = input_data[i];

        for (uint j = 0; j < output_size; j++) {
            activation[batch_index * stride_output + j] = 0.0f;
            output[batch_index * stride_output + j] = biases[j];
        }
        for (uint i = 0; i < input_size; i++) {
            for (uint j = 0; j < output_size; j++) {
                output[batch_index * stride_output + j] += input[batch_index * input_size + i] * weights[i * output_size + j];
            }
        }
        for (uint j = 0; j < output_size; j++) {
            if (is_hidden)
                activation[batch_index * stride_output + j] = output[batch_index * stride_output + j] > 0 ? output[batch_index * stride_output + j] : 0;
            else
                activation[batch_index * stride_output + j] = output[batch_index * stride_output + j];
        }
    }

    void backward_pass(float* incoming_gradients, uint batch_index) override {
        for (unsigned int i = 0; i < output_size; i++) {
            if (is_hidden)
                d_output[batch_index * stride_output + i] = output[batch_index * stride_output + i] > 0.0f ? incoming_gradients[i] : 0.0f;
            else
                d_output[batch_index * stride_output + i] = incoming_gradients[i];
        }
        for (uint i = 0; i < input_size; i++) {
            for (uint j = 0; j < output_size; j++) {
                d_weights[batch_index * input_size * output_size + i * output_size + j] = d_output[batch_index * stride_output + j] * input[batch_index * input_size + i];
            }
        }
        for (uint j = 0; j < output_size; j++) {
            d_biases[batch_index * stride_output + j] = d_output[batch_index * stride_output + j];
        }
        for (uint i = 0; i < input_size; i++) {
            d_input[batch_index * input_size + i] = 0.0f;
            for (uint j = 0; j < output_size; j++) {
                d_input[batch_index * input_size + i] += d_output[batch_index * stride_output + j] * weights[i * output_size + j];
            }
        }
    }

    void update_parameters(float learning_rate, uint batch_size) override {
        std::fill_n(grad_weights_sum.begin(), input_size * output_size, 0.0f);
        std::fill_n(grad_biases_sum.begin(), output_size, 0.0f);
        float step_size = learning_rate / batch_size;

        for (uint batch = 0; batch < batch_size; batch++) {
            for (uint i = 0; i < input_size * output_size; i++) {
                grad_weights_sum[i] += d_weights[batch * input_size * output_size + i];
            }
        }
        for (uint batch = 0; batch < batch_size; batch++) {
            for (uint j = 0; j < output_size; j++) {
                grad_biases_sum[j] += d_biases[batch * stride_output + j]; // FIXED
            }
        }
        for (uint i = 0; i < input_size * output_size; i++) {
            weights[i] -= step_size * grad_weights_sum[i];
        }
        for (uint j = 0; j < output_size; j++) {
            biases[j] -= step_size * grad_biases_sum[j];
        }
    }
};

void apply_softmax(float* logits, uint size) {
    float maxInput = logits[0];
    for (uint i = 0; i < size; i++) {
        if (logits[i] > maxInput)
            maxInput = logits[i];
    }
    float sum = 0.0f;
    for (uint i = 0; i < size; i++) {
        float exp_val = std::exp(logits[i] - maxInput);
        sum += exp_val;
        logits[i] = exp_val;
    }
    float norm = 1.0f / sum;
    for (uint i = 0; i < size; i++) {
        logits[i] *= norm;
    }
}

struct NeuralNetwork {
    std::vector<Layer*> layers;
    float* current_batch_input;
    float* current_batch_labels;
    uint batch_size;
    std::vector<std::thread> threads;
    std::mutex padlock;
    std::condition_variable thread_bell;
    std::condition_variable manager_bell;
    std::atomic<bool> terminate_pool;
    std::atomic<uint> active_threads;
    std::atomic<uint> current_index;
    std::vector<float> initial_gradients;
    uint input_size;
    uint output_size;
    uint job_id = 0;

    NeuralNetwork(uint batch_size) : batch_size(batch_size) {
        terminate_pool = false;
        active_threads = 0;
        job_id = 0;

        for (uint i = 0; i < std::thread::hardware_concurrency(); i++) {
            threads.push_back(std::thread([this, i]() {
                uint my_last_job = 0;
                while (true) {
                    std::unique_lock<std::mutex> lock(padlock);
                    thread_bell.wait(lock, [&]() {
                        return my_last_job != job_id || terminate_pool;
                    });
                    if (terminate_pool) break;
                    my_last_job = job_id;
                    lock.unlock();

                    while (current_index < this->batch_size) {
                        uint j = current_index.fetch_add(1);
                        if (j >= this->batch_size) break;
                        forward(current_batch_input + j * input_size, j);
                        backward(current_batch_labels + j * output_size, j);
                    }
                    --active_threads;
                    if (active_threads == 0) {
                        manager_bell.notify_one();
                    }
                }
            }));
        }
    }

    void forward(float* initial_input, uint batch_index) {
        float* current_input = initial_input;
        for (auto layer : layers) {
            layer->forward(current_input, batch_index);
            // FIXED: pointer now offsets by aligned stride to match physical memory
            current_input = &(layer->activation[batch_index * align_to_cache(layer->output_size)]);
        }
        apply_softmax(current_input, output_size);
    }

    void backward(float* true_labels, uint batch_index) {
        uint final_stride = align_to_cache(output_size);
        for (uint i = 0; i < output_size; i++) {
            // FIXED: index into the padded activation array using final_stride
            layers.back()->activation[batch_index * final_stride + i] -= true_labels[i];
            initial_gradients[output_size * batch_index + i] = layers.back()->activation[batch_index * final_stride + i];
        }
        float* current_gradients = &initial_gradients[output_size * batch_index];
        for (auto it = layers.rbegin(); it != layers.rend(); it++) {
            (*it)->backward_pass(current_gradients, batch_index);
            current_gradients = &(*it)->d_input[(*it)->input_size * batch_index];
        }
    }

    void train_on_batch(float* batch_inputs, float* batch_labels, uint current_batch_size, float learning_rate) {
        std::unique_lock<std::mutex> lock(padlock);
        current_batch_input = batch_inputs;
        current_batch_labels = batch_labels;
        batch_size = current_batch_size;
        active_threads = std::thread::hardware_concurrency();
        job_id++;
        current_index = 0;
        thread_bell.notify_all();
        manager_bell.wait(lock, [&]() { return active_threads == 0; });
        for (auto layer : layers) {
            layer->update_parameters(learning_rate, current_batch_size);
        }
    }

    float* predict(float* single_input) {
        forward(single_input, 0);
        return &layers.back()->activation[0];
    }

    ~NeuralNetwork() {
        std::unique_lock<std::mutex> lock(padlock);
        terminate_pool = true;
        thread_bell.notify_all();
        lock.unlock();
        for (std::thread& thread : threads) {
            thread.join();
        }
    }

    void compile() {
        input_size = layers[0]->input_size;
        output_size = layers.back()->output_size;
        initial_gradients.resize(output_size * batch_size);
    }
};

uint32_t flip_endian(unsigned char* bytes) {
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) | ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3];
}

float* load_mnist_images(const char* filename, uint32_t& out_num_images) {
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
    float* images = new float[out_num_images * 784];
    unsigned char pixel_buffer[784];
    for (int i = 0; i < out_num_images; i++) {
        file.read((char*)pixel_buffer, 784);
        float div = 1.0f / 255.0f;
        for (int j = 0; j < 784; j++)
            images[i * 784 + j] = pixel_buffer[j] * div;
    }
    return images;
}

float* load_mnist_labels(const char* filename, uint32_t& out_num_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "FATAL: Could not open file " << filename << '\n';
        exit(1);
    }
    unsigned char buffer[4];
    file.read((char*)buffer, 4);
    file.read((char*)buffer, 4);
    out_num_labels = flip_endian(buffer);
    float* labels = new float[out_num_labels * 10];
    unsigned char digit;
    for (int i = 0; i < out_num_labels; i++) {
        file.read((char*)&digit, 1);
        for (int j = 0; j < 10; j++) {
            labels[i * 10 + j] = (j == digit) ? 1.0f : 0.0f;
        }
    }
    return labels;
}

float calculate_accuracy(NeuralNetwork& nn, float* images, float* labels, uint32_t num_items) {
    uint correct = 0;
    for (uint i = 0; i < num_items; i++) {
        float* pred = nn.predict(images + i * IMAGE_SIZE);
        float max = pred[0];
        uint prediction_index = 0;
        uint correct_index = 0;
        for (uint j = 0; j < 10; j++) {
            if (max < pred[j]) {
                max = pred[j];
                prediction_index = j;
            }
            if (labels[i * 10 + j] == 1.0f)
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
    float* images = load_mnist_images("/Users/vlad/PersonalProjects/AI/NeuralNetwork/train-images-idx3-ubyte", num_images);
    uint32_t num_labels;
    float* labels = load_mnist_labels("/Users/vlad/PersonalProjects/AI/NeuralNetwork/train-labels-idx1-ubyte", num_labels);

    uint32_t num_test_images;
    uint32_t num_test_labels;
    float* test_images = load_mnist_images("/Users/vlad/PersonalProjects/AI/NeuralNetwork/t10k-images-idx3-ubyte", num_test_images);
    float* test_labels = load_mnist_labels("/Users/vlad/PersonalProjects/AI/NeuralNetwork/t10k-labels-idx1-ubyte", num_test_labels);

    if (num_images != num_labels) {
        std::cerr << "FATAL: Data mismatch! Images: " << num_images << " Labels: " << num_labels << '\n';
        return 1;
    }
    std::cout << "Successfully loaded " << num_labels << " pairs.\n";

    NeuralNetwork nn(BATCH_SIZE);

    DenseLayer* l1 = new DenseLayer(IMAGE_SIZE, 128, true, BATCH_SIZE);
    l1->initialize_weights();
    nn.layers.push_back(l1);

    DenseLayer* l2 = new DenseLayer(128, NUM_CLASSES, false, BATCH_SIZE);
    l2->initialize_weights();
    nn.layers.push_back(l2);

    nn.compile();

    for (uint i = 0; i < 50; i++) {
        for (uint j = 0; j < num_images; j += BATCH_SIZE) {
            nn.train_on_batch(images + j * IMAGE_SIZE, labels + j * NUM_CLASSES, std::min(BATCH_SIZE, (num_images - j)), LEARNING_RATE);
        }
        float acc = calculate_accuracy(nn, test_images, test_labels, num_test_images);
        std::cout << "--- Epoch " << i << " ---\nAccuracy: " << acc * 100.0f << "%\n\n";
    }

    delete[] images;
    delete[] labels;
    delete[] test_images;
    delete[] test_labels;
    return 0;
}
