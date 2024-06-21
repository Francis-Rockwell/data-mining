class Dataset:
    def __init__(self, train_data_public, train_data_internet, test_data_public):
        self.train_data_public = train_data_public
        self.train_data_internet = train_data_internet
        self.test_data_public = test_data_public

    def preprocess(self):
        print(self.train_data_public)
        print(self.train_data_internet)
        print(self.test_data_public)
        pass

    def train_feature(self):
        return self.train_feature

    def train_label(self):
        return self.train_label

    def test_feature(self):
        return self.test_feature

    def test_label(self):
        return self.test_label
