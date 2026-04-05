import numpy as np
import bisect
from itertools import combinations

class Full_SVM:

    def __init__(self, num_iterations=5, threshold=None):
        #Args:
            #num_iterations (int): Number of iterations for calculating the weight vector and bias.
            #threshold (float, optional): Threshold for filtering projected data points for bias calculation.
            
        
        if not isinstance(num_iterations, int) or num_iterations <= 0:
            raise ValueError("num_iterations must be a positive integer.")
        
        if threshold is not None and (not isinstance(threshold, (int, float)) or threshold < 0):
            raise ValueError("threshold must be a non-negative number.")

        self.num_iterations = num_iterations
        self.threshold = threshold
        self.classifiers = []  # Stores (class_i, class_j, w, b) for each pair
        self.classes = None
        self.n_features = None

    #Function to calculate weight vector (w) and bias (b) for a binary sub-problem
    def get_coordinates(self, X, y_binary, X_neg, X_pos):
        
        # Calculate w: pointing from negative class center to positive class center
        w = np.mean(X_pos, axis=0) - np.mean(X_neg, axis=0)
        w = w.reshape(-1, 1)

        #Avoiding dividing if the norm is 0
        norm = np.linalg.norm(w)
        if norm > 1e-12:
            w = w / norm
        else:
            w = np.zeros_like(w)
        
        # Project full training set onto w
        XP = X @ w
        ind = np.argsort(XP.flatten())
        XP_sorted = XP[ind].flatten()
        YP_sorted = y_binary[ind].flatten()

        # Apply thresholding
        if self.threshold is not None:
            mask = (XP_sorted >= -self.threshold) & (XP_sorted <= self.threshold)
            XP_threshold = XP_sorted[mask]
        else:
            XP_threshold = XP_sorted

        # Error handling for empty threshold array
        if XP_threshold.size < 2:
            raise ValueError(f"Threshold {self.threshold} is too restrictive, resulting in {XP_threshold.size} points. Cannot calculate optimal bias 'b'.")

        #Initialization
        optimal_entropy = float('inf')
        b = 0.0
        num_threshold_points = len(XP_threshold)

        
        for i in range(num_threshold_points - 1):
            T = (XP_threshold[i] + XP_threshold[i+1]) / 2   #midpoint
            j = bisect.bisect_left(XP_sorted, T)

            #Splitting 
            left_YP = YP_sorted[:j]
            right_YP = YP_sorted[j:]
            
            if left_YP.size == 0 or right_YP.size == 0:
                continue

            #Calculate entropy 
            def calculate_entropy(labels):
                n = labels.size
                p_pos = np.sum(labels == 1) / n
                p_neg = 1 - p_pos
                if p_pos <= 0 or p_pos >= 1:
                    return 0
                return -p_pos * np.log2(p_pos) - p_neg * np.log2(p_neg)

            H_left = calculate_entropy(left_YP)
            H_right = calculate_entropy(right_YP)
            
            total_samples = left_YP.size + right_YP.size
            weighted_entropy = (left_YP.size / total_samples) * H_left + (right_YP.size / total_samples) * H_right

            #Choose the best b according to the minimum entropy
            if weighted_entropy < optimal_entropy:
                optimal_entropy = weighted_entropy
                b = -T

        return w, b, optimal_entropy

    #Fits a single binary SVM model for a pair of classes.
    def fit_binary(self, X, y_binary):

        #Splitting data into binary classes
        X_neg_initial = X[y_binary.flatten() == -1]
        X_pos_initial = X[y_binary.flatten() == 1]

        if X_neg_initial.size == 0 or X_pos_initial.size == 0:
            raise ValueError("Both classes must be present to fit the model.")

        X_neg_current = X_neg_initial
        X_pos_current = X_pos_initial

        best_w, best_b, optimal_entropy = None, None, float('inf')

        for i in range(self.num_iterations):
            try:
                w, b, entropy = self.get_coordinates(X, y_binary, X_neg_current, X_pos_current)
            except ValueError as e:
                if i == 0: raise e
                break

            if entropy < optimal_entropy:
                optimal_entropy = entropy
                best_w, best_b = w, b

            # Refine dataset for the next iteration by selecting points closest to the current boundary
            d_pos = (X_pos_current @ w + b).flatten()
            d_neg = (X_neg_current @ w + b).flatten()
            
            # Keep at least 2 points per class to allow further iterations
            new_pos_count = max(2, len(X_pos_current) // 2)
            new_neg_count = max(2, len(X_neg_current) // 2)
            
            pos_indices = np.argsort(np.abs(d_pos))[:new_pos_count]
            neg_indices = np.argsort(np.abs(d_neg))[:new_neg_count]
            
            X_pos_current = X_pos_current[pos_indices]
            X_neg_current = X_neg_current[neg_indices]

            if X_pos_current.size < 2 or X_neg_current.size < 2:
                break

        if best_w is None:
            raise ValueError("Failed to find a suitable hyperplane during training.")

        return best_w, best_b

    #Fit multi-class SVM using One-vs-One 
    def fit(self, X, y):
        #Args:
            #X (np.ndarray): Training features of shape (n_samples, n_features).
            #y (np.ndarray): Training labels of shape (n_samples,).
        
        if not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0:
            raise ValueError("X must be a non-empty 2D numpy array.")
        if not isinstance(y, np.ndarray) or y.ndim != 1 or y.size == 0:
            raise ValueError("y must be a non-empty 1D numpy array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of samples.")

        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        self.classifiers = []

        if len(self.classes) < 2:
            raise ValueError("At least 2 unique classes are required for SVM fitting.")

        # One-vs-One: Train one classifier per pair of classes
        #create pairs
        for class_i, class_j in combinations(self.classes, 2):
            # Create binary labels for this pair: class_i = +1, class_j = -1
            mask = (y == class_i) | (y == class_j)
            X_pair = X[mask]
            y_pair = y[mask]
            
            y_binary = np.where(y_pair == class_i, 1, -1).reshape(-1, 1)
            
            try:
                w, b = self.fit_binary(X_pair, y_binary)
                self.classifiers.append((class_i, class_j, w, b))
            except ValueError as e:
                print(f"Warning: Could not train classifier for pair ({class_i}, {class_j}): {e}")

        if not self.classifiers:
            raise RuntimeError("No classifiers were successfully trained.")

    #Computes the decision scores for each class using voting mechanism.
    def decision_function(self, X):
        #Args:
            #X (np.ndarray): Input features of shape (n_samples, n_features).
        
        #Returns:
            #np.ndarray: Decision scores of shape (n_samples, n_classes).
                        #Higher score indicates stronger confidence for that class.
        
        if not self.classifiers:
            raise RuntimeError("The model has not been fitted yet. Call .fit() first.")
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("Input X must be a 2D numpy array.")
        if X.shape[1] != self.n_features:
            raise ValueError(f"Input X has {X.shape[1]} features, but model was trained with {self.n_features}.")

        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        # Initialize vote counts and confidence scores
        votes = np.zeros((n_samples, n_classes))
        confidence = np.zeros((n_samples, n_classes))
        
        # Each classifier votes for one of its two classes
        for class_i, class_j, w, b in self.classifiers:
            # Calculate decision value (signed distance from hyperplane)
            decision_values = (X @ w + b).flatten()
            
            # Positive decision value -> class_i, negative -> class_j
            votes_for_i = (decision_values > 0).astype(int)
            votes_for_j = (decision_values <= 0).astype(int)
            
            # Map class indices
            idx_i = np.where(self.classes == class_i)[0][0]
            idx_j = np.where(self.classes == class_j)[0][0]
            
            votes[:, idx_i] += votes_for_i
            votes[:, idx_j] += votes_for_j
            
            # Accumulate confidence (absolute decision value as confidence measure)
            confidence[:, idx_i] += votes_for_i * np.abs(decision_values)
            confidence[:, idx_j] += votes_for_j * np.abs(decision_values)
        
        # Return confidence scores (can be used for more nuanced predictions)
        # For pure voting, you could just return votes, but confidence provides more information
        return confidence

    def predict(self, X):
        
        scores = self.decision_function(X)
        
        # In OvO, the class with the highest confidence score wins
        class_indices = np.argmax(scores, axis=1)
        return self.classes[class_indices]

    #Predicts class labels using pure majority voting (without confidence weighting).
    def predict_with_voting(self, X):
        
        
        if not self.classifiers:
            raise RuntimeError("The model has not been fitted yet. Call .fit() first.")
        
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        votes = np.zeros((n_samples, n_classes))
        
        for class_i, class_j, w, b in self.classifiers:
            decision_values = (X @ w + b).flatten()
            
            votes_for_i = (decision_values > 0).astype(int)
            votes_for_j = (decision_values <= 0).astype(int)
            
            idx_i = np.where(self.classes == class_i)[0][0]
            idx_j = np.where(self.classes == class_j)[0][0]
            
            votes[:, idx_i] += votes_for_i
            votes[:, idx_j] += votes_for_j
        
        class_indices = np.argmax(votes, axis=1)
        return self.classes[class_indices]
