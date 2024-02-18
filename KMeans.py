from cluster import cluster
import random

class KMeans(cluster):

    def __init__(self, k=5, max_iteration=100):
        super().__init__(k, max_iteration)

    def fit(self, x):
        super().fit(x)
        instance_num = len(x)

        # Normalize the data
        self.normalize(x)

        # Pick random k centroids
        random.seed()
        centroids = random.choices(x, k=self.k)

        # Cluster
        cluster_hypotheses_prev = [0 for i in range(instance_num)]
        cluster_hypotheses_curr = [0 for i in range(instance_num)]
        i = 0
        while i < self.max_iteration:
            cluster_hypotheses_curr = self.cluster(x, centroids)
            if self.converge(cluster_hypotheses_prev, cluster_hypotheses_curr):
                break
            cluster_hypotheses_prev = cluster_hypotheses_curr
            i += 1
        return cluster_hypotheses_curr, centroids

    def normalize(self, x):
        feature_num = len(x[0])
        instance_num = len(x)

        for i in range(feature_num):
            feature_max = 0
            feature_min = float('infinity')
            for j in range(instance_num):
                feature_max = max(feature_max, x[j][i])
                feature_min = min(feature_min, x[j][i])
            diff = float(feature_max - feature_min)
            for j in range(instance_num):
                x[j][i] = (x[j][i] - feature_min) / diff
        return

    def cluster(self, x, centroids):
        instance_num = len(x)
        feature_num = len(x[0])
        centroids_num = len(centroids)
        cluster_hypotheses = [0 for i in range(instance_num)]

        # Cluster the instances
        for i in range(instance_num):
            closest_distance = float('infinity')
            for j in range(centroids_num):
                distance = float(0)
                for k in range(feature_num):
                    distance += (x[i][k] - centroids[j][k]) ** 2
                if distance < closest_distance:
                    cluster_hypotheses[i] = j
                    closest_distance = distance

        # Recalculate the centroids
        cluster_size = [1 for i in range(centroids_num)]
        feature_sum = [[0 for i in range(feature_num)] for i in range(centroids_num)]
        for i in range(instance_num):
            cluster_num = cluster_hypotheses[i]
            for j in range(feature_num):
                feature_sum[cluster_num][j] += x[i][j]
            cluster_size[cluster_num] += 1
        for i in range(centroids_num):
            centroids[i] = [feature_sum[i][j] / cluster_size[i] for j in range(feature_num)]

        return cluster_hypotheses

    def converge(self, prev=[], curr=[]):
        if len(prev) == 0 or len(curr) == 0:
            print("Converge error: Empty lists")
            return False
        if len(prev) != len(curr):
            print("Converge error: Inconsistent lists")
            return False

        for i in range(len(prev)):
            if prev[i] != curr[i]:
                return False
        return True

