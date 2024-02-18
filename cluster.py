import sys

class cluster:

    def __init__(self, k=5, max_iteration=100):
        self.k = k
        self.max_iteration = max_iteration

    def fit(self, x):
        instance_num = len(x)

        # Check input
        if x is None or instance_num == 0:
            print("Empty data")
            sys.exit(1)
        if self.k > instance_num:
            print("Too few instances for ", self.k, " clusters")
            sys.exit(1)
