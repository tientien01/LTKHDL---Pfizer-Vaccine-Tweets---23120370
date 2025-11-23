import numpy as np


def accuracy_score(y_true, y_pred):
    """Calculate accuracy score"""
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix for binary classification"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

# ==============================================================================
# 1. LOGISTIC REGRESSION
# ==============================================================================

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, lambda_param=0.1):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.lambda_param = lambda_param  # L2 regularization
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, x):
        """Sigmoid function with numerical stability"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def _compute_loss(self, y, y_pred):
        """Compute binary cross-entropy loss with L2 regularization"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        log_loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        l2_penalty = (self.lambda_param / 2) * np.sum(self.weights ** 2)
        return log_loss + l2_penalty

    def fit(self, X, y, verbose=False):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []

        for iteration in range(self.n_iters):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # Compute loss
            loss = self._compute_loss(y, y_predicted)
            self.loss_history.append(loss)
            
            # Backward pass with L2 regularization
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + (self.lambda_param / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        """Predict probabilities"""
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

# ==============================================================================
# 2. CORRECTED NAIVE BAYES (Multinomial)
# ==============================================================================

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # 1. Tính toán số lượng (Counts)
        self._feature_counts = np.zeros((n_classes, n_features))
        self._class_counts = np.zeros(n_classes)
        self._priors = np.zeros(n_classes)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._feature_counts[idx, :] = np.sum(X_c, axis=0)
            self._class_counts[idx] = np.sum(self._feature_counts[idx, :])
            self._priors[idx] = X_c.shape[0] / n_samples

        # 2. TÍNH TOÁN TRƯỚC (Pre-compute) Log Probabilities
        # Thay vì tính lúc predict, ta tính luôn lúc fit để dùng lại
        # Công thức: log(count + alpha) - log(total_count + alpha * n_features)
        
        numerator = self._feature_counts + self.alpha
        # Reshape _class_counts để broadcasting (n_classes, 1)
        denominator = self._class_counts.reshape(-1, 1) + (self.alpha * n_features)
        
        # Ma trận (n_classes, n_features) chứa xác suất log của từng từ trong từng class
        self._feature_log_prob = np.log(numerator) - np.log(denominator)
        
        # Log Prior
        self._class_log_prior = np.log(self._priors)

    def predict(self, X):
        # 3. DỰ ĐOÁN BẰNG NHÂN MA TRẬN (Vectorization)
        # Công thức: Posterior = Log(Prior) + X . Log(Feature_Prob)^T
        # X: (n_samples, n_features)
        # _feature_log_prob.T: (n_features, n_classes)
        # Kết quả jll (Joint Log Likelihood): (n_samples, n_classes)
        
        jll = np.dot(X, self._feature_log_prob.T) + self._class_log_prior
        
        # Lấy chỉ số của class có xác suất lớn nhất
        return self._classes[np.argmax(jll, axis=1)]


