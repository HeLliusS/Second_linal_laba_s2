#include "experiments.hpp"
#include <iostream>
#include <vector>

int main() {
    std::vector<double> lr_list = {0.001, 0.01, 0.5, 1.0};
    std::vector<int> batch_list = {1, 16, 64, 256};
    std::vector<InitType> init_list = {SMALL_RANDOM, ZERO, LARGE_RANDOM};
    std::vector<double> lambda_list = {0.0, 0.001, 0.01, 0.1};
    std::vector<double> beta_list = {0.0, 0.5, 0.9, 0.99};

    // 1) Эксперименты с lr (batch_size=32, init=SMALL_RANDOM, lambda_reg = 0.0)
    for (double lr : lr_list) {
        std::string out_dir = "experiments_results/lr_" + std::to_string(lr);
        run_experiment(lr, 32, SMALL_RANDOM, out_dir);
    }

    // 2) Эксперименты с batch_size (lr=0.1, init=SMALL_RANDOM, lambda_reg = 0.0)
    for (int bs : batch_list) {
        std::string out_dir = "experiments_results/bs_" + std::to_string(bs);
        run_experiment(0.1, bs, SMALL_RANDOM, out_dir);
    }

    // 3) Эксперименты с инициализацией (lr=0.1, batch=32, lambda_reg = 0.0)
    std::vector<std::pair<InitType, std::string>> init_names = {
        {SMALL_RANDOM, "small_random"},
        {ZERO, "zero"},
        {LARGE_RANDOM, "large_random"}
    };

    for (auto [type, name] : init_names) {
        std::string out_dir = "experiments_results/init_" + name;
        run_experiment(0.1, 32, type, out_dir);
    }

    // 4) Эксперименты с batch_size (lr=0.1, batch = 32, init=SMALL_RANDOM)
    for (double lambda_reg: lambda_list) {
        std::string out_dir = "experiments_results/lambda_" + std::to_string(lambda_reg);
        run_experiment(0.1, 32, SMALL_RANDOM, out_dir, lambda_reg);
    }

    // 5) Эксперименты с momentum (lr=0.1, batch = 32, init=SMALL_RANDOM, lambda_reg = 0.0)
    for (double beta: beta_list) {
        std::string out_dir = "experiments_results/beta_" + std::to_string(beta);
        run_experiment(0.1, 32, SMALL_RANDOM, out_dir, 0.0, beta);
    }

    return 0;
}
