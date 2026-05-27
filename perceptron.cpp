#include "perceptron.hpp"


Perceptron::Perceptron(int weight_count) : train_loss_log(), val_loss_log(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.01, 0.01);

    weights.resize(weight_count);

    int weight_index = 0;

    for (weight_index = 0; weight_index < weight_count; weight_index++) {
        weights[weight_index] = dis(gen);
    }

    bias = 0.;

    velocity_w.resize(weight_count, 0.0);
    velocity_b = 0.0;
}


void Perceptron::reinitialize(InitType type) {
    std::random_device rd;
    std::mt19937 gen(rd());
    if (type == SMALL_RANDOM) {
        std::uniform_real_distribution<> dis(-0.01, 0.01);
        for (auto& w : weights) w = dis(gen);
        bias = 0.0;
    } else if (type == ZERO) {
        std::fill(weights.begin(), weights.end(), 0.0);
        bias = 0.0;
    } else if (type == LARGE_RANDOM) {
        std::normal_distribution<> dis(0.0, 10.0);
        for (auto& w : weights) w = dis(gen);
        bias = dis(gen);
    }

    std::fill(velocity_w.begin(), velocity_w.end(), 0.0);
    velocity_b = 0.0;
}


double Perceptron::sigmoid(double z) const {
    return 1. / (1. + std::exp(-z));
}

std::vector<double> Perceptron::forward(const std::vector<std::vector<double>>& X) const {
    int result_size = X.size(), result_index = 0;
    std::vector<double> result(result_size);

    for (result_index = 0; result_index < result_size; result_index++) {
        result[result_index] = sigmoid(std::inner_product(weights.begin(), weights.end(), X[result_index].begin(), bias));
    }

    return result;
}

double Perceptron::compute_loss(const std::vector<double>& y_true,
                                const std::vector<double>& y_pred,
                                double lambda_reg) const {
    double loss = 0.;
    double reg = 0.;

    int size = y_true.size();
    int y_index = 0;

    const double eps = 1e-9;

    for (y_index = 0; y_index < size; y_index++) {
        loss += y_true[y_index] * std::log(y_pred[y_index] + eps) + (1. - y_true[y_index]) * std::log(1. - y_pred[y_index] + eps);
    }

    for (double w : weights) reg += w * w;

    loss += 0.5 * lambda_reg * reg;

    return -loss / size;

}

void Perceptron::fit(const std::vector<std::vector<double>> &X_train, const std::vector<double> &y_train,
                     const std::vector<std::vector<double> > &X_val, const std::vector<double> &y_val,
                     int epochs, double lr, int batch_size, double lambda_reg, double beta) {
    int train_size = X_train.size(), weights_size = weights.size(), weights_index = 0;
    int epoch_number = 0, batch_end_index = 0, batch_start_index = 0, batch_now_index = 0, batch_now_size = batch_size;

    std::vector<std::vector<double>> X_batch(batch_size);
    std::vector<double> y_batch(batch_size);
    std::vector<double> batch_fw;

    std::vector<double> dw(weights_size, 0.0);
    double db, error = 0.;

    std::vector<int> indexes(train_size);
    std::iota(indexes.begin(), indexes.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());

    for (epoch_number = 0; epoch_number < epochs; epoch_number++) {

        std::shuffle(indexes.begin(),indexes.end(), gen);

        for (batch_start_index = 0; batch_start_index < train_size; batch_start_index += batch_size) {
            batch_end_index = std::min(batch_start_index + batch_size, train_size);
            batch_now_size = batch_end_index - batch_start_index;

            X_batch.resize(batch_now_size);
            y_batch.resize(batch_now_size);

            for (batch_now_index = 0; batch_now_index < batch_now_size; batch_now_index++) {
                X_batch[batch_now_index] = X_train[indexes[batch_now_index + batch_start_index]];
                y_batch[batch_now_index] = y_train[indexes[batch_now_index + batch_start_index]];
            }


            batch_fw = forward(X_batch);

            std::fill(dw.begin(), dw.end(), 0.0);
            db = 0.;

            for (batch_now_index = 0; batch_now_index < batch_now_size; batch_now_index++) {
                error = batch_fw[batch_now_index] - y_batch[batch_now_index];

                db += error;
                for (weights_index = 0; weights_index < weights_size; weights_index++) {
                    dw[weights_index] += error * X_batch[batch_now_index][weights_index] + lambda_reg * weights[weights_index];
                }
            }

            db /= batch_now_size;
            for (weights_index = 0; weights_index < weights_size; weights_index++) {
                dw[weights_index] /= batch_now_size;
            }

            velocity_b = beta * velocity_b - lr * db;
            bias += velocity_b;
            for (weights_index = 0; weights_index < weights_size; weights_index++) {
                velocity_w[weights_index] = beta * velocity_w[weights_index] - lr * dw[weights_index];
                weights[weights_index] += velocity_w[weights_index];
            }
        }


        std::vector<double> train_fw = forward(X_train);
        train_loss_log.push_back(compute_loss(y_train, train_fw, lambda_reg));

        std::vector<double> val_fw = forward(X_val);
        val_loss_log.push_back(compute_loss(y_val, val_fw, lambda_reg));
    }
}

std::vector<int> Perceptron::predict(const std::vector<std::vector<double>>& X, double threshold) const {
    std::vector<double> forw = forward(X);
    std::vector<int> result(forw.size());

    int result_size = result.size(), result_index = 0;

    for (result_index = 0; result_index < result_size; result_index++) {
        if (forw[result_index] >= threshold)
            result[result_index] = 1;
        else
            result[result_index] = 0;
    }

    return result;
}


const std::vector<double> &Perceptron::get_weights() const {
    return weights;
}

double Perceptron::get_bias() const {
    return bias;
}

const std::vector<double> &Perceptron::get_train_loss_log() const {
    return train_loss_log;
}

const std::vector<double> &Perceptron::get_val_loss_log() const {
    return val_loss_log;
}

