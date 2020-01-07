import argparse
import math


class NaiveBayes:
    def __init__(self):
        self.data = []
        self.summaries = {}
        self.probs_class = {}

    def set_data(self, data):
        self.data = data
        return self

    def train(self):
        """
        Summaries dataset seperated by classes and calculate P(c):
            - standard deviation
            - mean
        """
        if not self.data:
            raise ValueError("Dataset to train not found")
        separated = self._separate_dataset_by_class()
        for c, subset in separated.items():
            self.summaries[c] = [(mean(col), variance(col), len(col)) for col in zip(*subset)]

        total_rows = float(len(data))
        for c, summary in self.summaries.items():
            self.probs_class[c] = summary[0][2]/total_rows
        return self

    def predict(self, row):
        """
        Predict class that input row belongs to
        """
        probabilities = self.probs_class.copy()
        for c, summary in self.summaries.items():
            for i in range(len(summary)):
                mean, var, _ = summary[i]
                probabilities[c] *= self._prob(row[i], mean, var)
        best_label, best_prob = None, -1.0
        for c, prob in probabilities.items():
            print(f"label: {c} - prob: {prob}")
            if best_label is None or prob > best_prob:
                best_label = c
                best_prob = prob
        print(f"Predicted label: {best_label}")
        print("=========================")
        return best_label

    def _separate_dataset_by_class(self):
        d = {}
        for row in self.data:
            if row[0] not in d:
                d[row[0]] = [row[1:]]
            else:
                d[row[0]].append(row[1:])
        return d

    def _prob(self, x, mean, variance):
        exponent = math.exp(-(x-mean)**2/(2*variance))
        return exponent/math.sqrt(2*math.pi*variance)


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def variance(numbers):
    avg = mean(numbers)
    return sum([(num - avg)**2 for num in numbers])/(len(numbers) - 1)


def read_data_from_file(input_file):
    data = []
    with open(input_file, 'rb') as f:
        for line in f.readlines():
            data_point = []
            values = line.decode("utf-8").split('\t')
            data_point.append(values[0])
            for v in values[1:]:
                try:
                    data_point.append(float(v))
                except:
                    continue
            data.append(data_point)
    return data


if __name__ == '__main__':
    # parsing required command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--output')
    args = parser.parse_args()

    # read data from input file
    data = read_data_from_file(args.data)

    # feed data and train model
    model = NaiveBayes()
    model.set_data(data)

    model.train()
    error_count = 0
    for d in data:
        predicted_class = model.predict(d[1:])
        if predicted_class != d[0]:
            error_count += 1

    print("====================================")
    print(f"Errors: {error_count}")

    # write the results to output
    with open(args.output, 'w+') as f:
        mean_A1, var_A1, _ = model.summaries['A'][0]
        mean_A2, var_A2, _ = model.summaries['A'][1]
        prob_A = model.probs_class['A']
        line1 = f"{mean_A1}\t{var_A1}\t{mean_A2}\t{var_A2}\t{prob_A}"
        print(line1, file=f)

        mean_B1, var_B1, _ = model.summaries['B'][0]
        mean_B2, var_B2, _ = model.summaries['B'][1]
        prob_B = model.probs_class['B']
        line2 = f"{mean_B1}\t{var_B1}\t{mean_B2}\t{var_B2}\t{prob_B}"
        print(line2, file=f)

        print(error_count, file=f)
