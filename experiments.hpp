#ifndef LINALSECONDLABA_EXPERIMENTS_HPP
#define LINALSECONDLABA_EXPERIMENTS_HPP

#include <string>
#include <vector>
#include "perceptron.hpp"

void run_experiment(double lr, int batch_size, InitType init_type,
                    const std::string& output_dir,
                    double lambda_reg = 0.0, double beta = 0.0,
                    int n_samples = 500, int n_features = 2, int epochs = 100);


#endif