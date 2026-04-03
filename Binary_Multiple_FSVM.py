class binary_Multiple_SVM:
    def __init__(self, num_iterations=5, threshold = None):
        #Args:
            #num_iterations (int): Number of iterations for hyperparameter calculation
            #threshold (float, optional): Threshold for filtering projected data points during 'b' calculation. Defaults to None
        
        self.num_iterations = num_iterations
        self.threshold = threshold 
        self.w = None
        self.b = None
        self.classes = None
        

    # Calculates the weight vector (w) and bias (b) for the SVM
    def get_coordinates(self, X, y_, X_neg, X_pos, Y_neg, Y_pos):
        #Args:
            #X (np.ndarray): The input feature matrix.
            #y_ (np.ndarray): The binary labels (-1 or 1) corresponding to X
            #X_neg (np.ndarray): Feature vectors for negative class samples
            #X_pos (np.ndarray): Feature vectors for positive class samples
            #Y_neg (np.ndarray): Labels for negative class samples (all -1)
            #Y_pos (np.ndarray): Labels for positive class samples (all 1)
        
        #Calculating W and normalizing it
        w = (X_pos.T @ Y_pos) + (X_neg.T @ Y_neg)
        
        # Handle cases where w might be a zero vector to prevent division by zero
        norm = np.linalg.norm(w)
        if norm > 0:
            w = w / norm
        
        
        # Calculating b (bias term)
        # Step 1: Project data onto the weight vector 'w'
        XP = X @ w 
        
        # Step 2: Sort the projected data and their corresponding labels
        ind = np.argsort(XP.flatten())  # Get sorted indices
        XP_sorted = XP[ind]  # Sort XP
        YP_sorted = y_[ind]  # Sort Y according to XP

        #Apply thresholding if specified to focus on points near the decision boundary.
        if self.threshold !=  None:
            # Keep only the data points in the interval [-threshold, +threshold]
            XP_threshold = XP_sorted[(XP_sorted >= -self.threshold) & (XP_sorted <= self.threshold)]
        else:
            # If no threshold is set, consider all data points.
            XP_threshold = XP_sorted

        
        # Guard condition: If XP_threshold is empty after filtering, we cannot determine 'b'
        if XP_threshold.size == 0:
            #Return a default 0 for b and a high entropy to indicate a poor separation.
            return w, 0.0, float('inf')
            
        optimal_entropy = float('inf') # Initialize with positive infinity for minimization
        b = 0.0 # Default bias term

        # Iterate through potential split points (midpoints between sorted projected data points)
        #to find the 'b' that minimizes the weighted entropy
        num_threshold_points = len(XP_threshold)
        
        for i in range(num_threshold_points - 1):
            # Get the i-th midpoint as a splitting threshold 'T'
            T = (XP_threshold[i] + XP_threshold[i+1]) / 2

            # Find the index 'j' in the *original sorted projected data* (XP_sorted)
            # where 'T' would be inserted, effectively splitting the data
            j = bisect.bisect_left(XP_sorted, T)
            
            # Split data into left and right based on the threshold 'T'
            # 'left_YP' contains labels for samples <= T, 'right_YP' for samples > T
            left_YP = YP_sorted[:j]
            right_YP = YP_sorted[j:]
            
            # Compute entropy for the LEFT split
            n_left = len(left_YP) + 1e-11 #epsilon to denominators to prevent division by zero
            p_left_neg = np.sum(left_YP == -1) / n_left
            p_left_pos = 1 - p_left_neg
            H_left = 0
            if p_left_neg > 0 and p_left_pos > 0:
                H_left = -p_left_neg * np.log2(p_left_neg) - p_left_pos * np.log2(p_left_pos)

            # Compute entropy for the RIGHT split
            n_right = len(right_YP) + 1e-11
            p_right_neg = np.sum(right_YP == -1) / n_right
            p_right_pos = 1 - p_right_neg
            H_right = 0
            if p_right_neg > 0 and p_right_pos > 0:
                H_right = -p_right_neg * np.log2(p_right_neg) - p_right_pos * np.log2(p_right_pos)

            
            # Calculate the weighted entropy for the current split.
            # This represents the impurity of the split, aiming to minimize it.
            total_samples = len(XP) + 1e-11
            weighted_entropy = (n_left / total_samples) * H_left + \
                               (n_right / total_samples) * H_right

            # Keep track of the split that yields the smallest entropy.
            if weighted_entropy < optimal_entropy:
                optimal_entropy = weighted_entropy
                b = -T # The bias 'b' is the negative of the optimal threshold 'T'
        
        return w, b, optimal_entropy

    def fit(self, X, y):
        #Args:
            #X (np.ndarray): The training feature matrix.
            #y (np.ndarray): The training labels.

        #Raises:
            #ValueError: If the input data `X` or labels `y` are not valid (e.g., empty, wrong shape).
        
        if not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0:
            raise ValueError("Input X must be a non-empty 2D numpy array.")
        if not isinstance(y, np.ndarray) or y.ndim != 1 or y.size == 0:
            raise ValueError("Input y must be a non-empty 1D numpy array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")

        #Making sure there are 2 classes
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError(f"Expected exactly 2 classes, but found {len(self.classes)}: {self.classes}")

        # Convert labels to binary (-1 or 1) for internal calculation.
        # The first unique class found is mapped to 1, the second to -1.
        y_binary = np.where(y == self.classes[0], 1, -1).reshape(-1, 1)
                                                                 
        '''# Check if the array has more than two dimensions
        n_samples, n_features = X.shape
        #self.w = np.zeros((n_classes, n_features, 1))
        #self.b = np.zeros(n_classes)
        
        X = np.array(X)
        y = np.array(y)

        #for j, cls in enumerate(self.classes):
        # Create binary labels for current class (1 for cls, -1 otherwise)
        y_ = np.where(y == self.classes[0], 1, -1)'''
        
        # Initialize lists to store coordinates (w, b, entropy) from each iteration.
        #all_coordinates = []

        # Initial split of data into positive and negative samples.
        X_neg_current = X[y_binary.flatten() == -1]
        X_pos_current = X[y_binary.flatten() == 1]

        Y_neg_current = -np.ones((len(X_neg_current), 1))
        Y_pos_current = np.ones((len(X_pos_current), 1))

        # Guard condition: Ensure both positive and negative classes are present.
        if X_neg_current.size == 0 or X_pos_current.size == 0:
            raise ValueError("Both positive and negative classes must be present in the training data.")

        
        optimal_entropy = float('inf') # Initialize with positive infinity for minimization
        # Iteratively refine w and b
        for i in range(self.num_iterations):
            w, b, entropy = self.get_coordinates(X, y_binary, X_neg_current, X_pos_current, Y_neg_current, Y_pos_current)
            #all_coordinates.append([w, b, entropy])
        

            #Calculate distances to the current hyperplane for both positive and negative samples.
            # These distances help identify the 'closest' points for the next iteration.
            d_pos = X_pos_current @ w + b
            d_neg = X_neg_current @ w + b
            
            # Sort points by their distance to the hyperplane.
            # We keep half of the data that is closest to the margin (smallest absolute distance).
            pos_indices_sorted = np.argsort(np.abs(d_pos).flatten())
            neg_indices_sorted = np.argsort(np.abs(d_neg).flatten())
            
            # Select the half of the data points closest to the decision boundary.
            X_pos_current = X_pos_current[pos_indices_sorted[:len(X_pos_current) // 2]]
            X_neg_current = X_neg_current[neg_indices_sorted[:len(X_neg_current) // 2]]
            
            # Re-create labels for the selected subset of data.
            Y_neg_current = -np.ones((len(X_neg_current), 1))
            Y_pos_current = np.ones((len(X_pos_current), 1))


            if optimal_entropy > entropy:
                optimal_entropy = entropy
                self.w = w
                self.b = b

            # Guard condition: If either class becomes empty after shrinking, break early.
            if X_neg_current.size == 0 or X_pos_current.size == 0:
                print(f"Warning: One or both classes became empty after shrinking in iteration {i+1}. Stopped early.")
                break
            
        '''#at the end we select the coordenates with best entropy
        entropy_column = [sublist[2] for sublist in coordenates]
        #print(entropy_column)
        optimal_indx = np.argmin(entropy_column)
        #print(optimal_indx)
        self.w = coordenates[optimal_indx][0]
        self.b = coordenates[optimal_indx][1]'''
    
    def predict(self, X, return_approx=False):
        #Args:
            #X (np.ndarray): The input feature matrix for prediction.
            #return_approx (bool): If True, returns the raw decision function values (approximate values).
                                  #If False, returns the predicted class labels.

        if self.w is None or self.b is None or self.classes is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("Input X for prediction must be a 2D numpy array.")
        if X.shape[1] != self.w.shape[0]:
            raise ValueError(f"Input X has {X.shape[1]} features, but model was trained with {self.w.shape[0]} features.")

        
        # Calculate the decision function value (distance from hyperplane)
        approx = X @ self.w + self.b
        
        if return_approx:
            return approx

        # Map the decision function values to the original class labels
        return np.where(approx >= 0, self.classes[0], self.classes[1])

        
    def decision_score(self, X):
        #Calculates the decision score (distance from the hyperplane) for new data points.

        if self.w is None or self.b is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("Input X for decision_score must be a 2D numpy array.")
        if X.shape[1] != self.w.shape[0]:
            raise ValueError(f"Input X has {X.shape[1]} features, but model was trained with {self.w.shape[0]} features.")

        return X @ self.w + self.b
