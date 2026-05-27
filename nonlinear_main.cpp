#include "perceptron.hpp"
#include "data_generator.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

int main() {
    std::filesystem::create_directories("nonlinear_results");

    std::vector<std::vector<double>> X_xor;
    std::vector<double> y_xor;
    generate_xor_data(X_xor, y_xor, 400, 0.1);
    
    std::vector<std::vector<double>> X_train_x, X_val_x;
    std::vector<double> y_train_x, y_val_x;
    stratified_split(X_xor, y_xor, X_train_x, y_train_x, X_val_x, y_val_x);
    
    std::vector<double> mean_x, stddev_x;
    standardize(X_train_x, mean_x, stddev_x, true);
    standardize(X_val_x, mean_x, stddev_x, false);
    
    Perceptron perc_xor(2);
    perc_xor.fit(X_train_x, y_train_x, X_val_x, y_val_x, 200, 0.1, 32);
    
    auto pred_xor = perc_xor.predict(X_val_x);
    double acc_xor = accuracy(pred_xor, y_val_x);
    std::cout << "XOR validation accuracy: " << acc_xor << std::endl;
    
    std::vector<std::vector<double>> X_circ;
    std::vector<double> y_circ;
    generate_circle_data(X_circ, y_circ, 500, 2.4);
    
    std::vector<std::vector<double>> X_train_c, X_val_c;
    std::vector<double> y_train_c, y_val_c;
    stratified_split(X_circ, y_circ, X_train_c, y_train_c, X_val_c, y_val_c);
    
    std::vector<double> mean_c, stddev_c;
    standardize(X_train_c, mean_c, stddev_c, true);
    standardize(X_val_c, mean_c, stddev_c, false);
    
    Perceptron perc_circ(2);
    perc_circ.fit(X_train_c, y_train_c, X_val_c, y_val_c, 200, 0.1, 32);
    
    auto pred_circ = perc_circ.predict(X_val_c);
    double acc_circ = accuracy(pred_circ, y_val_c);
    std::cout << "Circle validation accuracy: " << acc_circ << std::endl;

    std::ofstream points_xor("nonlinear_results/xor_points.csv");
    points_xor << "x1,x2,label\n";
    for (size_t i = 0; i < X_val_x.size(); ++i) {
        points_xor << X_val_x[i][0] << "," << X_val_x[i][1] << "," << y_val_x[i] << "\n";
    }
    
    std::ofstream points_circ("nonlinear_results/circle_points.csv");
    points_circ << "x1,x2,label\n";
    for (size_t i = 0; i < X_val_c.size(); ++i) {
        points_circ << X_val_c[i][0] << "," << X_val_c[i][1] << "," << y_val_c[i] << "\n";
    }
    

    auto w = perc_circ.get_weights();
    double b = perc_circ.get_bias();
    std::ofstream line("nonlinear_results/circle_line.csv");
    line << "x1,x2\n";
    double x1_min = -2.0, x1_max = 2.0;
    double x2_min = -(w[0]*x1_min + b) / w[1];
    double x2_max = -(w[0]*x1_max + b) / w[1];
    line << x1_min << "," << x2_min << "\n";
    line << x1_max << "," << x2_max << "\n";
    
    return 0;
}