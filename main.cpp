#include "perceptron.hpp"
#include "data_generator.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>


void save_visualization(const std::vector<std::vector<double>>& X,
                        const std::vector<double>& y,
                        const std::vector<double>& weights,
                        double bias,
                        const std::string& points_file = "results/points.csv",
                        const std::string& line_file = "results/line.csv") {
    std::ofstream pout(points_file);
    pout << "x1,x2,label\n";
    for (size_t i = 0; i < X.size(); ++i) {
        pout << X[i][0] << "," << X[i][1] << "," << y[i] << "\n";
    }

    double x1_min = X[0][0], x1_max = X[0][0];
    for (auto& row : X) {
        if (row[0] < x1_min) x1_min = row[0];
        if (row[0] > x1_max) x1_max = row[0];
    }

    double padding = (x1_max - x1_min) * 0.2;
    x1_min -= padding;
    x1_max += padding;
    double x2_at_min = -(weights[0] * x1_min + bias) / weights[1];
    double x2_at_max = -(weights[0] * x1_max + bias) / weights[1];
    std::ofstream lout(line_file);
    lout << "x1,x2\n";
    lout << x1_min << "," << x2_at_min << "\n";
    lout << x1_max << "," << x2_at_max << "\n";
}


int main() {
    std::filesystem::create_directories("results");

    std::vector<std::vector<double>> X;
    std::vector<double> y;
    int n_samples = 500, n_features = 2;

    int epochs = 100, batch_size = 32;
    double lr = 0.1;

    generate_linear_data(X, y, n_samples, n_features);


    std::vector<std::vector<double>> X_train, X_val;
    std::vector<double> y_train, y_val;

    stratified_split(X, y, X_train, y_train, X_val, y_val);

    std::cout << "Train size: " << X_train.size() << ", Validation size: " << X_val.size() << std::endl;


    std::vector<double> mean, stddev;
    standardize(X_train, mean, stddev);
    standardize(X_val, mean, stddev, false);

    Perceptron perc(2);
    perc.fit(X_train, y_train, X_val, y_val, epochs, lr, batch_size);

    std::vector<int> pred_train = perc.predict(X_train);
    std::vector<int> pred_val = perc.predict(X_val);

    double acc_train = accuracy(pred_train, y_train);
    double acc_val = accuracy(pred_val, y_val);

    std::cout << "Train accuracy: " << acc_train << std::endl;
    std::cout << "Validation accuracy: " << acc_val << std::endl;


    int epoch_index = 0;

    for (epoch_index = 0; epoch_index < epochs; epoch_index++) {
        std::cout << perc.get_train_loss_log()[epoch_index] << " ";

        if ((epoch_index + 1) % 10 == 0) std::cout << std::endl;
    }

    std::ofstream loss_file("results/loss.csv");
    loss_file << "epoch,train_loss,val_loss\n";
    for (epoch_index = 0; epoch_index < epochs; ++epoch_index) {
        loss_file << epoch_index << "," << perc.get_train_loss_log()[epoch_index] << "," << perc.get_val_loss_log()[epoch_index] << "\n";
    }


    std::cout << std::endl;


    for (epoch_index = 0; epoch_index < epochs; epoch_index++) {
        std::cout << perc.get_val_loss_log()[epoch_index] << " ";

        if ((epoch_index + 1) % 10 == 0) std::cout << std::endl;
    }

    save_visualization(X_val, y_val, perc.get_weights(), perc.get_bias());

    return 0;
}






