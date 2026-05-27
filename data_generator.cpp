#include "data_generator.hpp"


void generate_linear_data(std::vector<std::vector<double>>& X, std::vector<double>& y, int n_samples, int n_features) {
    int sample_index = 0, feature_index = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist0(2.0, 1.0);
    std::normal_distribution<> dist1(4.0, 1.0);

    X.resize(n_samples, std::vector<double>(n_features));
    y.resize(n_samples);

    for (sample_index = 0; sample_index < n_samples / 2; sample_index++) {
        for (feature_index = 0; feature_index < n_features; feature_index++) {
            X[sample_index][feature_index] = dist0(gen);
        }
        y[sample_index] = 0.0;
    }

    for (sample_index = n_samples / 2; sample_index < n_samples; sample_index++) {
        for (feature_index = 0; feature_index < n_features; feature_index++) {
            X[sample_index][feature_index] = dist1(gen);
        }
        y[sample_index] = 1.0;
    }
}


void generate_xor_data(std::vector<std::vector<double>>& X, std::vector<double>& y,
                       int n_samples, double sigma) {
    int sample_index = 0, class_item_index = 0, class_index = 0;

    X.resize(n_samples, std::vector<double>(2));
    y.resize(n_samples);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, sigma);

    std::vector<std::pair<double, double>> centers = {{0.0,0.0}, {1.0,1.0}, {0.0,1.0}, {1.0,0.0}};
    std::vector<int> labels = {0, 0, 1, 1};
    int per_class = n_samples / 4;

    sample_index = 0;

    for (class_index = 0; class_index < 4; class_index++) {
        for (class_item_index = 0; class_item_index < per_class; ++class_item_index) {
            X[sample_index][0] = centers[class_index].first + noise(gen);
            X[sample_index][1] = centers[class_index].second + noise(gen);
            y[sample_index] = labels[class_index];
            ++sample_index;
        }
    }

    while (sample_index < n_samples) {
        class_index = std::uniform_int_distribution<>(0,3)(gen);
        X[sample_index][0] = centers[class_index].first + noise(gen);
        X[sample_index][1] = centers[class_index].second + noise(gen);
        y[sample_index] = labels[class_index];
        ++sample_index;
    }
}

void generate_circle_data(std::vector<std::vector<double>>& X, std::vector<double>& y,
                          int n_samples, double radius) {
    int sample_index = 0;

    X.resize(n_samples, std::vector<double>(2));
    y.resize(n_samples);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-3, 3);

    for (sample_index = 0; sample_index < n_samples; ++sample_index) {
        double x = dis(gen);
        double x2 = dis(gen);
        X[sample_index][0] = x;
        X[sample_index][1] = x2;
        y[sample_index] = (x*x + x2*x2 <= radius*radius) ? 1.0 : 0.0;
    }
}


void stratified_split(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                      std::vector<std::vector<double>>& X_train, std::vector<double>& y_train,
                      std::vector<std::vector<double>>& X_val, std::vector<double>& y_val,
                      double train_ratio) {
    int data_size = X.size();
    int train_data_size0 = 0, train_data_size1 = 0;
    int index = 0;

    std::vector<int> index0, index1;

    for (index = 0; index < data_size; index++) {
        if (y[index] < 0.0001)   index0.push_back(index);
        else   index1.push_back(index);
    }

    int index0_size = index0.size(), index1_size = index1.size();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(index0.begin(), index0.end(), gen);
    std::shuffle(index1.begin(), index1.end(), gen);

    train_data_size0 = static_cast<int>(index0_size * train_ratio);
    train_data_size1 = static_cast<int>(index1_size * train_ratio);

    X_train.clear(); y_train.clear();
    X_val.clear(); y_val.clear();

    for (index = 0; index < train_data_size0; index++) {
        X_train.push_back(X[index0[index]]);
        y_train.push_back(y[index0[index]]);
    }

    for (index = 0; index < train_data_size1; index++) {
        X_train.push_back(X[index1[index]]);
        y_train.push_back(y[index1[index]]);
    }

    for (index = train_data_size0; index < index0_size; index++) {
        X_val.push_back(X[index0[index]]);
        y_val.push_back(y[index0[index]]);
    }

    for (index = train_data_size1; index < index1_size; index++) {
        X_val.push_back(X[index1[index]]);
        y_val.push_back(y[index1[index]]);
    }
}


void standardize(std::vector<std::vector<double>>& data,
                 std::vector<double>& mean,
                 std::vector<double>& stddev,
                 bool compute) {
    int sample_index = 0, feature_index = 0;

    int n_samples = data.size();
    if (n_samples == 0)
        return;
    int n_features = data[0].size();

    double diff = 0.0;

    if (compute) {
        mean.assign(n_features, 0.0);
        stddev.assign(n_features, 0.0);

        for (sample_index = 0; sample_index < n_samples; sample_index++) {
            for (feature_index = 0; feature_index < n_features; feature_index++) {
                mean[feature_index] += data[sample_index][feature_index];
            }
        }

        for (feature_index = 0; feature_index < n_features; feature_index++) {
            mean[feature_index] /= n_samples;
        }

        for (sample_index = 0; sample_index < n_samples; sample_index++) {
            for (feature_index = 0; feature_index < n_features; feature_index++) {
                diff = data[sample_index][feature_index] - mean[feature_index];

                stddev[feature_index] += diff * diff;
            }
        }

        for (feature_index = 0; feature_index < n_features; feature_index++) {
            stddev[feature_index] = std::sqrt(stddev[feature_index] / n_samples);

            if (stddev[feature_index] == 0.0) stddev[feature_index] = 1e-8;
        }
    }

    for (sample_index = 0; sample_index < n_samples; sample_index++) {
        for (feature_index = 0; feature_index < n_features; feature_index++) {
            data[sample_index][feature_index] = (data[sample_index][feature_index] - mean[feature_index]) / stddev[feature_index];
        }
    }

}

double accuracy(const std::vector<int>& pred, const std::vector<double>& real) {
    int correct = 0;
    int data_size = pred.size(), data_index = 0;

    if (data_size == 0)
        return 0.0;

    for (data_index = 0; data_index < data_size; data_index++) {
        if (pred[data_index] == static_cast<int>(real[data_index])) correct++;
    }

    return static_cast<double>(correct) / data_size;
}

