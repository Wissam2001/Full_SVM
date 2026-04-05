# Full_SVM 🛠️
This class implements a Support Vector Machine (SVM) from scratch, focusing on finding optimal hyperplanes (weights `w` and bias `b`) using a closed-form solution with entropy minimization. The implementation supports **One-vs-One (OvO)** multi-class strategies, making it flexible for various classification tasks.

# Features ✨
- **Closed-form solution** for weight vector calculation
- **Entropy-based bias selection** for optimal decision boundaries
- **Multi-class support** with One-vs-One strategies
- **Robust error handling** and input validation
- **Threshold filtering** for bias calculation on projected data
- **Support Two voting** For voting with distance into consederation (decision value), or voting only

# Research Objectives 🎯
1. **Develop a computationally efficient SVM**
2. **Explore entropy minimization** as an alternative to margin maximization for bias selection
3. **Create an educational implementation** that demonstrates core SVM concepts

# Mathematical Approach 📐
**Weight Vector Calculation** 
The weight vector `w` is computed as the normalized difference between class means:
$\mathbf{w} = \frac{\boldsymbol{\mu}_+ - \boldsymbol{\mu}_-}{\|\boldsymbol{\mu}_+ - \boldsymbol{\mu}_-\|} \tag{1}$
**Bias Selection**
The optimal bias `b` is found by projecting all data points onto `w` and selecting the threshold `T` that minimizes the number of projected data points for finding the optimal 'b':
\[
H(p) = -p_{\text{pos}} \log_2(p_{\text{pos}}) - p_{\text{neg}} \log_2(p_{\text{neg}})
\]

\[
T = \arg\min_{t} \left( \frac{n_{\text{left}}}{n_{\text{total}}} H_{\text{left}} + \frac{n_{\text{right}}}{n_{\text{total}}} H_{\text{right}} \right)
\]

\[
b = -T
\]
**Iterative Refinement**
Each iteration keeps only the points closest to the current decision boundary, focusing the classifier on potential support vectors.

# Libraries 📚
- **NumPy:** Used for array manipulations, mean calculations, projections, and sorting operations.
- **bisect:**  Used to efficiently find insertion points for threshold values when calculating the optimal bias, ensuring O(log n) complexity for sorted array operations.
- **itertools:** Used for generating combinations of class pairs in the One-vs-One strategy.

# Installation 💻

**Clone the repository:**
```bash
!git clone https://github.com/Wissam2001/Full_SVM.git
cd Full_SVM
from Full_SVM import Full_SVM
```

**Run the code:**
exp
```python
f_svm = Full_SVM.Full_SVM(num_iterations=5, threshold=0.5)
```



**Important⚠️**
Make sure to install the used libraries for this project

# Contributing 🤝
- Add cross-validation support
- Optimize entropy calculation for large datasets

# Contact ✉️
• **Email:** wissambadia4@gmail.com

• **LinkedIn:** [Badia Ouissam Lakas](linkedin.com/in/badia-ouissam-lakas-a66a28214)   




