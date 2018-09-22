import numpy as np

class LogisticRegression(object):
    def __int__(self, alpha=0.1, lamda=0.1, batch_size = 1):
        self.alpha = alpha
        self.lamda = lamda
        self.batch_size = batch_size
        self.m = None
        self.n = None
        self.epsilon = 0.001
        self.max_iter = 10000

    def fit(self,X,y):
        X, y = np.array(X), np.array(y)
        X = np.hstack(np.ones(shape=(X.shape[0],1)), X)
        self.m,self.n = X.shape
        self.weight = np.array([np.random.normal(0,1) for i in range(self.n)]).reshape(1,-1)
        i, j = 0,0
        while j < self.max_iter:
            start = self.batch_size * i
            if start + self.batch_size < self.n:
                end = start + self.batch_size
                i += 1
            else:
                end = self.n
                i = 0
            gradient = self.cals(X[start:end, :], y[start:end, :])
            if np.linalg.norm(gradient, 2) < self.epsilon:
                return
            self.weight -= self.alpha * gradient
            j += 1

    def predict(self, x):
        return self.sigmoid(np.dot(x, self.weight.T))

    def grad_cals(self, train_x, train_y):
        y_pred = self.sigmoid(np.dot(train_x, self.weight.T))
        gradient = np.dot((y_pred - train_y).T, train_x) + 2*self.lamda*self.weight
        return gradient

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))