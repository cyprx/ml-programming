import argparse
import math


class KMeans:
    def __init__(self):
        self.k = None
        self.data = []
        self.cluster_centers = []
        self.log_cluster_centers = []
        self.opz_criterias = []

    def set_data(self, data):
        self.data = data
        return self

    def set_prototype(self, cluster_centers=[]):
        self.cluster_centers = cluster_centers
        return self

    def train(self):
        n_loops = 0
        while True: 
            if len(self.log_cluster_centers) > 0:
                if self.log_cluster_centers[len(self.log_cluster_centers)-1] == self.cluster_centers:
                    print(f"Cluster centers unchanged, TERMINATE")
                    break
            n_loops += 1
            print(f"n_loops: {n_loops}")
            clusters = {}
            for i in range(len(self.data)):
                assigned_cluster = self._assign_cluster(self.data[i])
                if assigned_cluster not in clusters:
                    clusters[assigned_cluster] = [self.data[i]]
                else:
                    clusters[assigned_cluster].append(self.data[i])
            self._calc_optimization_criteria(clusters)
            self._recompute_cluster_centers(clusters)


    def _assign_cluster(self, x):
        min_d = None
        assigned_cluster = None
        for c in self.cluster_centers:
            d = self._euclidean_distance(x, c)
            if min_d is None:
                min_d = d
                assigned_cluster = c
            elif min_d > d:
                min_d = d
                assigned_cluster = c
        return assigned_cluster

    def _recompute_cluster_centers(self, clusters: dict):
        new_cluster_centers = []
        for c in self.cluster_centers:
            mean_x, mean_y = [mean(col) for col in zip(*clusters[c])]
            new_cluster_centers.append((mean_x, mean_y))
        self.log_cluster_centers.append(self.cluster_centers)
        self.cluster_centers = new_cluster_centers
        print(f"New cluster centers: {new_cluster_centers}")

    def _euclidean_distance(self, x1, x2):
        d = 0
        for i in range(len(x1)):
            d += float(x1[i] - x2[i])**2
        return math.sqrt(d)

    def _calc_optimization_criteria(self, clusters):
        _sum = float(0)
        for center, cluster in clusters.items():
            for x in cluster:
                _sum += sum([(x[i] - center[i])**2 for i in range(len(x))])
        print(f"Optimization criteria: {_sum}")
        self.opz_criterias.append(_sum)
        return _sum


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def read_data_from_file(input_file):
    data = []
    with open(input_file, 'rb') as f:
        for line in f.readlines():
            data_point = []
            values = line.decode("utf-8").split('\t')
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
    model = KMeans()
    model.set_data(data)
    model.set_prototype([(0, 5), (0, 4) , (0, 3)])

    model.train()

    import os
    input_filename = args.data.split(".")[0]
    progr_file = os.path.join(args.output, input_filename + '-Progr.tsv')
    proto_file = os.path.join(args.output, input_filename + '-Proto.tsv')

    # write the results to output
    with open(progr_file, 'w+') as f:
        lcs = model.log_cluster_centers
        for lc in lcs:
            line = f"{lc[0][0]},{lc[0][1]}\t{lc[1][0]},{lc[1][1]}\t{lc[2][0]},{lc[2][1]}"
            print(line, file=f)

    with open(proto_file, 'w+') as f:
        ocs = model.opz_criterias
        for oc in ocs:
            line = f"{oc}"
            print(line, file=f)
