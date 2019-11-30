import argparse

class Perceptron:
    def __init__(self):
        self.data = []

    def set_data(self, data):
        self.data = data

    def _predict(self, inputs, weights):
        y = 0
        for input, weight in zip(inputs, weights):
            y += input * weight
        return 1 if y > 0 else 0

    def train(self, weights, learning_type='constant', learning_rate=1.0, max_loops=100):
        print(f"Start training data with learning type {learning_type}, max_loops=f{max_loops}")
        error_counters = []
        for loop in range(max_loops):
            delta_weights = [0.0, 0.0, 0.0]
            error_counter = 0
            if learning_type == 'annealing':
                learning_rate = 1 / (loop + 1)
                print(f"New learning rate at loop {loop}: {learning_rate}")
            for i, d in enumerate(self.data):
                predicted_y = self._predict(d[1:], weights)
                actual_y = d[0]
                error_counter += int(predicted_y != actual_y)

                for j in range(len(weights)):
                    delta_weights[j] += (actual_y - predicted_y) *learning_rate * d[j+1]
            # batch update weights based on errors from last iteration
            for j in range(len(weights)):
                weights[j] += delta_weights[j]
            error_counters.append(error_counter)
            print(f"Updated weights: {weights}\n")
        print(f"Num of errors: {error_counters}")
        return error_counters

def write_output(f, data):
    for d in data:
        f.write(b"%d\t" % d)
    f.write(b"\n")

if __name__ == '__main__':
    # parsing required command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--output')
    args = parser.parse_args()

    # read data from input file
    data = []
    with open(args.data, 'rb') as f:
        for line in f.readlines():
            data_point = []
            values = line.decode("utf-8").split('\t')
            data_point.append(1.0 if values[0] == 'A' else 0.0)
            for v in values[1:]:
                try:
                    data_point.append(float(v))
                except:
                    continue
            # add bias
            data_point.append(1.0)
            data.append(data_point)

    # feed data and train model
    model = Perceptron()
    model.set_data(data)
    weights = [0.0, 0.0, 0.0]
    constant_lr_result = model.train(weights, learning_type='constant')

    weights = [0.0, 0.0, 0.0]
    annealing_lr_result = model.train(weights, learning_type='annealing')

    # write the results to output
    with open(args.output, 'wb+') as f:
        write_output(f, constant_lr_result)
        write_output(f, annealing_lr_result)
