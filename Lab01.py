import numpy as np


def generate_data(input_num, output_num, num) -> (np.ndarray, np.ndarray):
    return np.random.rand(num, input_num), np.random.randint(0, 2, (num, output_num))


def forward_perceptron(x: np.ndarray, w: np.ndarray, threshold:int) -> np.ndarray:
    return np.where(np.dot(x, w) > threshold, 1, 0)


def forward_delta_rule(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.dot(x, w)


def perception_learning(x: np.ndarray, y: np.ndarray, w: np.ndarray, study_rate: float, t: int) -> np.ndarray:
    y_pred = forward_perceptron(x, w, t)
    for i, y_unit in enumerate(y_pred):
        if y_unit != y[i]:
            if y[i] == 1:
                w[i] += study_rate * x
            elif y[i] == 0:
                w[i] -= study_rate * x
            else:
                raise ValueError("The value of y should be 0 or 1.")
    return w


def delta_rule_learning(x: np.ndarray, y: np.ndarray, w: np.ndarray, study_rate: float) -> np.ndarray:
    y_pred = forward_delta_rule(x, w)
    for i, y_unit in enumerate(y_pred):
        e = y[i] - y_unit
        w[i] += study_rate * e * x
    return w


if __name__ == '__main__':
    n = 100
    test_rate = 0.2
    threshold = 0
    s_r = 0.1
    input_layer_len = 10
    output_layer_len = 2
    data = generate_data(10, 2, n)
    x, y = data
    w_perceptron_learning = np.random.rand(input_layer_len, output_layer_len)
    w_delta_rule = np.random.rand(input_layer_len, output_layer_len)
    for i in range(round(n * test_rate)):
        w_perceptron_learning = perception_learning(x[i], y[i], w_perceptron_learning, s_r, threshold)
        w_delta_rule = delta_rule_learning(x[i], y[i], w_delta_rule, s_r)
    test_result = []
    for i in range(round(n * test_rate), n):
        y_perceptron = forward_perceptron(x[i], w_perceptron_learning, threshold)
        y_delta_rule = forward_delta_rule(x[i], w_delta_rule)
        test_result.append((y_perceptron, y_delta_rule))
    # calculate the accuracy
    accuracy_perceptron = sum([1 for y_pred, y in test_result if y_pred == y]) / len(test_result)
    print(f"Accuracy of perceptron learning: {accuracy_perceptron}")



