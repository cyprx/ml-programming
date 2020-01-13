import argparse
import math


class KNNModel:
    def __init__(self):
        self.k = None
        self.data = []
        self.case_base = []

    def set_data(self, data):
        self.data = data
        return self

    def set_k(self, k):
        self.k = k
        return self

    def train(self):

        # CB = {C1}
        self.case_base.append(self.data[0])

        for i in range(1, len(self.data)):
            neighbors = self._get_neighbors_from_cb(self.data[i], self.k)
            predicted_class = self._most_vote(neighbors)
            print(f"Predicted: {predicted_class}, actual: {self.data[i][0]}")
            if predicted_class != self.data[i][0]:
                self.case_base.append(self.data[i])
                print(f"=========> misclassified")
            print(f"Case base size: {len(self.case_base)}")

        print(f"Final case base: {self.case_base}")

    def predict(self, x):
        neighbors = self._get_neighbors_from_cb(x, self.k)
        predicted_class = self._most_vote(neighbors)
        print(f"============> {predicted_class}")
        return predicted_class

    def _euclidean_distance(self, x1, x2):
        d = 0
        for i in range(len(x1)):
            d += float(x1[i] - x2[i])**2
        return math.sqrt(d)

    def _get_neighbors_from_cb(self, x, k):
        distances = []
        for i in range(len(self.case_base)):
            distances.append((self.case_base[i], self._euclidean_distance(x[1:], self.case_base[i][1:])))
        return sorted(distances, key=lambda x: x[1])[:min(k, len(distances))]

    def _most_vote(self, neighbors):
        d = {}

        for neighbor in neighbors:
            label = neighbor[0][0]
            if label not in d:
                d[label] = 0
            d[label] += self._weight(neighbor, neighbors)
        max = None
        max_k = None
        for k, v in d.items():
            if max is None:
                max = v
                max_k = k
                continue
            if v > max:
                max = v
                max_k = k
        print(f"max votes: {d}")
        return max_k

    def _weight(self, x, neighbors):
        if neighbors[0][1] == neighbors[len(neighbors)-1][1]:
            return 1
        return (neighbors[len(neighbors)-1][1] - x[1])/float(-neighbors[0][1] + neighbors[len(neighbors)-1][1])



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
    errors = []
    for k in (2, 4, 6, 8, 10):
        model = KNNModel()
        model.set_data(data)
        model.set_k(k)
        model.train()
        error_count = 0
        for d in data:
            predicted_class = model.predict(d)
            if predicted_class != d[0]:
                error_count += 1
        errors.append(error_count)
        if k == 4:
            stored_case_base = model.case_base

    # write the results to output
    with open(args.output, 'w+') as f:
        line1 = f"{errors[0]}\t{errors[1]}\t{errors[2]}\t{errors[3]}\t{errors[4]}"
        print(line1, file=f)

        for cb in stored_case_base:
            line2 = f"{cb[0]}\t{cb[1]}\t{cb[2]}"
            print(line2, file=f)




