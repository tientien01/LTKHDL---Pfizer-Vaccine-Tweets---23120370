import numpy as np



# ==============================================================================
# 1. LOGISTIC REGRESSION
# ==============================================================================
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, lambda_param=0.1):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.lambda_param = lambda_param  # L2 regularization
        self.models = {}  # lưu weights và bias cho từng nhãn
        self.loss_history = {}  # lưu loss từng nhãn

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def _compute_loss(self, y, y_pred, weights):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        log_loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        l2_penalty = (self.lambda_param / 2) * np.sum(weights ** 2)
        return log_loss + l2_penalty

    def _fit_binary(self, X, y_binary, verbose=False):
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0
        loss_hist = []

        for iteration in range(self.n_iters):
            linear_model = np.dot(X, weights) + bias
            y_predicted = self._sigmoid(linear_model)
            loss = self._compute_loss(y_binary, y_predicted, weights)
            loss_hist.append(loss)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y_binary)) + (self.lambda_param / n_samples) * weights
            db = (1 / n_samples) * np.sum(y_predicted - y_binary)

            weights -= self.lr * dw
            bias -= self.lr * db

            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.4f}")

        return weights, bias, loss_hist

    def fit(self, X, y, verbose=False):
        """X: features, y: labels (0,1,2,...K-1)"""
        self.classes_ = np.unique(y)
        self.models = {}
        self.loss_history = {}

        for cls in self.classes_:
            if verbose:
                print(f"Training OvR model for class {cls}...")
            # Nhãn nhị phân: cls vs rest
            y_binary = (y == cls).astype(int)
            weights, bias, loss_hist = self._fit_binary(X, y_binary, verbose)
            self.models[cls] = {'weights': weights, 'bias': bias}
            self.loss_history[cls] = loss_hist

    def predict_proba(self, X):
        """Trả về xác suất cho từng lớp"""
        proba = np.zeros((X.shape[0], len(self.classes_)))
        for idx, cls in enumerate(self.classes_):
            weights = self.models[cls]['weights']
            bias = self.models[cls]['bias']
            proba[:, idx] = self._sigmoid(np.dot(X, weights) + bias)
        return proba

    def predict(self, X):
        """Dự đoán nhãn: chọn lớp có xác suất cao nhất"""
        proba = self.predict_proba(X)
        class_idx = np.argmax(proba, axis=1)
        return self.classes_[class_idx]

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


