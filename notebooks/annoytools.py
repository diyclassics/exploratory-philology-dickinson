import annoy


class AnnoyIndex:
    def __init__(self, vectors, labels):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype("float32")
        self.labels = labels
        self.search_in_x_trees = 8

    def build(self, number_of_trees=5):
        self.index = annoy.AnnoyIndex(self.dimension, metric="angular")
        for i, vec in enumerate(self.vectors):
            self.index.add_item(i, vec.tolist())
        self.index.build(number_of_trees)

    def load(self, filename):
        self.index = annoy.AnnoyIndex(self.dimension, metric="angular")
        self.index.load(filename)

    def save(self, filename):
        self.index.save(filename)

    def query(self, vector, k=10):
        indices = self.index.get_nns_by_vector(
            vector.tolist(), k, search_k=self.search_in_x_trees
        )
        return [self.labels[i] for i in indices]
