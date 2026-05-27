#include "experiments.hpp"
#include "data_generator.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>


void run_experiment(double lr, int batch_size, InitType init_type,
                    const std::string& output_dir,
                    double lambda_reg, double beta,
                    int n_samples, int n_features, int epochs) {
    // Создаём папку
    std::filesystem::create_directories(output_dir);

    // 1. Генерация данных
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    generate_linear_data(X, y, n_samples, n_features);

    // 2. Стратифицированное разделение
    std::vector<std::vector<double>> X_train, X_val;
    std::vector<double> y_train, y_val;
    stratified_split(X, y, X_train, y_train, X_val, y_val);

    // 3. Стандартизация
    std::vector<double> mean, stddev;
    standardize(X_train, mean, stddev, true);
    standardize(X_val, mean, stddev, false);

    // 4. Создание перцептрона с нужной инициализацией
    Perceptron perc(n_features);

    perc.reinitialize(init_type);

    // 5. Обучение
    perc.fit(X_train, y_train, X_val, y_val, epochs, lr, batch_size, lambda_reg, beta);

    // 6. Точность
    auto pred_train = perc.predict(X_train);
    auto pred_val = perc.predict(X_val);
    double acc_train = accuracy(pred_train, y_train);
    double acc_val = accuracy(pred_val, y_val);

    // 7. Сохранение логов
    std::ofstream loss_file(output_dir + "/loss.csv");
    loss_file << "epoch,train_loss,val_loss\n";
    auto& train_loss = perc.get_train_loss_log();
    auto& val_loss = perc.get_val_loss_log();
    for (size_t i = 0; i < train_loss.size(); ++i) {
        loss_file << i << "," << train_loss[i] << "," << val_loss[i] << "\n";
    }

    std::ofstream params_file(output_dir + "/params.txt");
    params_file << "lr=" << lr << "\n";
    params_file << "batch_size=" << batch_size << "\n";
    params_file << "init_type=" << init_type << "\n";
    params_file << "train_accuracy=" << acc_train << "\n";
    params_file << "val_accuracy=" << acc_val << "\n";

    std::cout << "Experiment finished: " << output_dir << std::endl;
}


