from sklearn.svm import SVC

class SVMModel:
    def __init__(self, C=1.0, kernel='linear', gamma='scale', probability=True):
        self.model = SVC(C=C, kernel=kernel, gamma=gamma, probability=probability)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def probability(self, X_test):
        return self.model.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)
    
    def save_model(self, filepath):
        import joblib
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        import joblib
        self.model = joblib.load(filepath)