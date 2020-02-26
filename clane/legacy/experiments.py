from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


class NodeClassification:
    def __init__(self, X, Y, test_size):
        # prepare train and test dataset
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(X.numpy(), Y.numpy(), test_size=test_size)
        # prepare initialize the classifier
        self.clf = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=1e+6, n_jobs=4)

    def train(self):
        self.clf.fit(self.train_X, self.train_Y)
    
    def test(self):
        result = dict()
        pred = self.clf.predict(self.test_X)
        result["micro"] = f1_score(pred, self.test_Y, average='micro')
        result["macro"] = f1_score(pred, self.test_Y, average='macro')
        return result
