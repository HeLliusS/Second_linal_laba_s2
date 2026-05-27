#ifndef LINALSECONDLABA_PERCEPTRON_HPP
#define LINALSECONDLABA_PERCEPTRON_HPP

#include <algorithm>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>


enum InitType {
    SMALL_RANDOM,   // по умолчанию (-0.01, 0.01)
    ZERO,           // все веса 0, bias 0
    LARGE_RANDOM    // например N(0,10)
};


class Perceptron {
private:
    std::vector<double> weights;
    double bias;

    std::vector<double> train_loss_log;
    std::vector<double> val_loss_log;

    std::vector<double> velocity_w;
    double velocity_b;

public:

    Perceptron(int weight_count);

    void reinitialize(InitType type);

    double sigmoid(double z) const;

    std::vector<double> forward(const std::vector<std::vector<double>>& X) const;

    double compute_loss(const std::vector<double>& y_true,
                        const std::vector<double>& y_pred,
                        double lambda_reg = 0.0) const;

    void fit(const std::vector<std::vector<double>>& X_train,
             const std::vector<double>& y_train,
             const std::vector<std::vector<double>>& X_val,
             const std::vector<double>& y_val,
             int epochs, double lr, int batch_size, double lambda_reg = 0.0, double beta = 0.0);

    std::vector<int> predict(const std::vector<std::vector<double>>& X, double threshold = 0.5) const;

    const std::vector<double>& get_weights() const;
    double get_bias() const;

    const std::vector<double>& get_train_loss_log() const;
    const std::vector<double>& get_val_loss_log() const;

};


#endif