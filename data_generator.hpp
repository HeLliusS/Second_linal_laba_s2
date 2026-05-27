#ifndef LINALSECONDLABA_DATA_GENERATOR_HPP
#define LINALSECONDLABA_DATA_GENERATOR_HPP

#include <algorithm>
#include <vector>
#include <random>
#include <cmath>

void generate_linear_data(std::vector<std::vector<double>>& X, std::vector<double>& y,
                          int n_samples, int n_features);


void generate_xor_data(std::vector<std::vector<double>>& X, std::vector<double>& y,
                       int n_samples, double sigma = 0.1);

void generate_circle_data(std::vector<std::vector<double>>& X, std::vector<double>& y,
                          int n_samples, double radius = 2.4);


void stratified_split(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                      std::vector<std::vector<double>>& X_train, std::vector<double>& y_train,
                      std::vector<std::vector<double>>& X_val, std::vector<double>& y_val,
                      double train_ratio = 0.7);

void standardize(std::vector<std::vector<double>>& data,
                 std::vector<double>& mean,
                 std::vector<double>& stddev,
                 bool compute = true);

double accuracy(const std::vector<int>& pred, const std::vector<double>& real);

#endif