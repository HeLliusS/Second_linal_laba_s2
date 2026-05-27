perceptron.hpp/cpp – класс перцептрона.

data_generator.hpp/cpp – генерация данных (линейные, XOR, окружность), стратификация, стандартизация, точность.

main.cpp – демонстрация работы на линейно разделимых данных с сохранением графиков.

run_experiments.cpp – автоматический прогон экспериментов (lr, batch size, инициализация, lambda, beta).

nonlinear_demo.cpp – проверка на нелинейных данных (XOR, окружность).

Python-скрипты – визуализация loss и разделяющих границ.

-----------------
Класс Perceptron
-----------------

Perceptron(int weight_count);

Создаёт перцептрон с weight_count признаками. Веса инициализируются малыми случайными значениями из равномерного распределения [-0.01, 0.01], bias = 0. Внутри также создаются векторы для логов и momentum (velocity).


void reinitialize(InitType type);

Переинициализирует веса и bias согласно типу InitType:
SMALL_RANDOM – малые случайные (по умолчанию).
ZERO – все веса и bias = 0.
LARGE_RANDOM – нормальное распределение N(0,10).


double sigmoid(double z) const;

Сигмоидная функция активации: 1/(1+exp(-z)).


std::vector<double> forward(const std::vector<std::vector<double>>& X) const;

Прямой проход. Для каждого образца вычисляет sigmoid(w·x + b). Возвращает вектор вероятностей принадлежности к классу 1.


double compute_loss(const std::vector<double>& y_true,
                    const std::vector<double>& y_pred,
                    double lambda_reg = 0.0) const;

Вычисляет бинарную кросс-энтропию с L2-регуляризацией (если lambda_reg > 0). Добавляет член 0.5 * lambda_reg * sum(w^2).


void fit(const std::vector<std::vector<double>>& X_train,
         const std::vector<double>& y_train,
         const std::vector<std::vector<double>>& X_val,
         const std::vector<double>& y_val,
         int epochs, double lr, int batch_size,
         double lambda_reg = 0.0, double beta = 0.0);

Обучает модель методом мини-батчного градиентного спуска.

epochs – количество эпох.

lr – скорость обучения.

batch_size – размер батча.

lambda_reg – коэффициент L2-регуляризации (по умолчанию 0).

beta – коэффициент Momentum (по умолчанию 0 – обычный SGD). При beta > 0 используется обновление с накоплением скорости.
В процессе обучения сохраняет значение loss на обучающей и валидационной выборках для каждой эпохи (в train_loss_log и val_loss_log).


std::vector<int> predict(const std::vector<std::vector<double>>& X, double threshold = 0.5) const;

Возвращает бинарные предсказания (0 или 1) на основе порога (по умолчанию 0.5).


const std::vector<double>& get_weights() const;
double get_bias() const;
const std::vector<double>& get_train_loss_log() const;
const std::vector<double>& get_val_loss_log() const;

Геттеры

----------------------------------------------
Функции подготовки данных (data_generator.hpp)
----------------------------------------------

void generate_linear_data(std::vector<std::vector<double>>& X, std::vector<double>& y,
                          int n_samples, int n_features);

Генерирует линейно разделимые данные: два гауссовых облака с центрами (2,2) и (4,4), σ=1. Классы 0 и 1.


void generate_xor_data(std::vector<std::vector<double>>& X, std::vector<double>& y,
                       int n_samples, double sigma = 0.1);

Генерирует XOR-подобные данные: четыре облака в углах квадрата [0,1]x[0,1] с метками: (0,0) и (1,1) → 0; (0,1) и (1,0) → 1. Добавляется шум sigma.


void generate_circle_data(std::vector<std::vector<double>>& X, std::vector<double>& y,
                          int n_samples, double radius = 2.4);

Точки равномерно в квадрате [-3,3]x[-3,3]. Метка 1, если x1² + x2² ≤ radius², иначе 0.


void stratified_split(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                      std::vector<std::vector<double>>& X_train, std::vector<double>& y_train,
                      std::vector<std::vector<double>>& X_val, std::vector<double>& y_val,
                      double train_ratio = 0.7);

Стратифицированное разделение на обучающую и валидационную выборки с сохранением пропорции классов.


void standardize(std::vector<std::vector<double>>& data,
                 std::vector<double>& mean,
                 std::vector<double>& stddev,
                 bool compute = true);

Z-нормализация. При compute = true вычисляет среднее и стандартное отклонение по выборке и применяет преобразование. При false использует переданные mean и stddev.


double accuracy(const std::vector<int>& pred, const std::vector<double>& real);

Доля правильных ответов.

---------------------
Эксперименты и выводы
---------------------

Для проведения экспериментов используется run_experiments.cpp, который перебирает значения гиперпараметров и сохраняет результаты в папку experiments_results.

  r	                Сходимость	                      Итоговая точность

0.001	   очень медленная, loss снижается плохо	         низкая
0.01	   медленная, требует больше эпох	                 средняя
0.1	     хорошая, стабильная	                           высокая (>0.95)
0.5	     быстрая, но возможны колебания	                 высокая
1.0	     сильные колебания, может расходиться	      низкая или нестабильная

Вывод:   оптимальная скорость – 0.1.



Влияние размера батча (batch size)

batch size	             Характер сходимости	           Точность

1 	                   очень шумная, loss скачет	       высокая
16	                      умеренно шумная	               высокая
32	                          гладкая	                   высокая
64	                гладкая, но медленнее на эпоху	     высокая
256	           очень гладкая, но медленнее обновление	   высокая

Вывод: батч 32 даёт хороший баланс между скоростью и стабильностью.


Влияние инициализации весов

Тип инициализации	                                                Результат
ZERO (все нули)	                        не нарушает симметрию, градиенты одинаковы – обучение не идёт, точность ~0.5
SMALL_RANDOM (±0.01)	                               хорошая сходимость, высокая точность
LARGE_RANDOM (N(0,10))	                     градиенты взрываются, loss растёт, точность низкая

Вывод: малые случайные веса – оптимальный выбор.


L2-регуляризация (lambda_reg)

λ	                                                  Эффект
0.0	                             базовый вариант, возможно небольшое переобучение
0.001	                     небольшое улучшение обобщения (валидационная точность чуть выше)
0.01	                        веса уменьшаются, точность может незначительно снизиться
0.1	                             слишком сильный штраф, точность падает заметно

Вывод: умеренная регуляризация (λ=0.001) полезна, сильная – вредит.


Momentum (beta)
β	                         Сходимость
0.0               базовая, может колебаться
0.5	                  небольшие улучшения
0.9	        значительное ускорение, сглаживание колебаний
0.99	       может вызывать overshoot, замедление в конце

Вывод: β=0.9 – хороший выбор для ускорения сходимости.


Нелинейные данные (XOR, окружность)
На линейно разделимых данных перцептрон достигает точности >95%.

На XOR точность ~0.5 (случайное угадывание) – линейная граница не может разделить классы.

На окружности (класс внутри круга) точность также ~0.5 – прямая линия не способна описать окружность.

Вывод: однослойный перцептрон применим только для линейно разделимых задач.


-----------------------
Требования к окружению
-----------------------

Компилятор C++17 (поддержка std::filesystem, std::shuffle, std::iota).

Python 3 с библиотеками pandas, matplotlib, numpy (для визуализации).





