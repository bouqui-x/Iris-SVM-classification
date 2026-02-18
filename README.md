# Support Vector Machine (SVM) - Iris Classification

A machine learning project implementing Support Vector Machine (SVM) for classifying iris flowers into three species based on their physical characteristics.

## Project Overview

This project demonstrates the use of **Support Vector Machine (SVM)** for multi-class classification. SVM is a powerful supervised learning algorithm used for classification, regression, and outlier detection tasks. It works by finding the optimal hyperplane that separates data points of different classes in high-dimensional space.

## Objective

Build an SVM classifier to predict the species of iris flowers based on four physical features:
- Sepal length
- Sepal width
- Petal length
- Petal width

## Dataset

**Dataset**: Iris Flower Dataset (Built-in scikit-learn)

**Classes**: 3 species of Iris flowers
1. Iris Setosa (0)
2. Iris Versicolor (1)
3. Iris Virginica (2)

**Features**:
- `sepal length (cm)`: Length of the sepal
- `sepal width (cm)`: Width of the sepal
- `petal length (cm)`: Length of the petal
- `petal width (cm)`: Width of the petal

**Dataset Size**: 150 samples (50 samples per class)

## What is Support Vector Machine (SVM)?

**Support Vector Machine (SVM)** is a supervised machine learning algorithm that:
- Finds the optimal hyperplane that maximizes the margin between different classes
- Works exceptionally well in high-dimensional spaces
- Is effective when the number of dimensions exceeds the number of samples
- Uses support vectors (data points closest to the hyperplane) to define the decision boundary

### Key Concepts:
1. **Hyperplane**: A decision boundary that separates different classes
2. **Support Vectors**: Data points closest to the hyperplane
3. **Margin**: Distance between the hyperplane and the nearest data points
4. **Kernel**: Function to transform data into higher dimensions (for non-linear separation)

##  Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `matplotlib` - Data visualization
  - `scikit-learn` - Machine learning (SVM model, dataset)
  - `numpy` - Numerical computations (implicit)

## 📈 Project Workflow

### 1. Data Loading
```python
from sklearn.datasets import load_iris
iris = load_iris()
```
- Loaded the famous Iris dataset from scikit-learn

### 2. Data Exploration
```python
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df["flower_name"] = df.target.apply(lambda x: iris.target_names[x])
```
- Created a pandas DataFrame for easier analysis
- Added target labels and flower names

### 3. Data Visualization
```python
plt.scatter(df0["sepal length (cm)"], df0["sepal width (cm)"], 
            color="green", marker="+")
plt.scatter(df1["sepal length (cm)"], df1["sepal width (cm)"], 
            color="blue", marker=".")
```
- Visualized Setosa vs Versicolor using sepal measurements
- Showed clear separation between the two species

### 4. Data Preparation
```python
X = df.drop(["target", "flower_name"], axis="columns")
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
- Split features and target variable
- Created train-test split (80-20)
- Training set: 120 samples
- Testing set: 30 samples

### 5. Model Training
```python
model = SVC()
model.fit(X_train, y_train)
```
- Initialized SVM classifier with default parameters
- Trained on the training data

### 6. Model Evaluation
```python
model.score(X_test, y_test)
```
- Evaluated model accuracy on test data

## 📊 Results

The SVM classifier achieved excellent performance on the Iris dataset:

- **Model**: Support Vector Classifier (SVC)
- **Training Samples**: 120
- **Testing Samples**: 30
- **Expected Accuracy**: ~95-100% (Iris is a well-separated dataset)

### Model Performance
- Successfully classified three iris species
- High accuracy due to well-separated classes
- Effective feature separation using hyperplanes

## 📈 Visualizations

The project includes scatter plots showing:
- **Sepal Length vs Sepal Width** for Setosa and Versicolor
- Clear visual separation between species
- Color-coded markers for different species

## How to Run

### Prerequisites

```bash
pip install pandas matplotlib scikit-learn
```

### Running the Code

1. **Option 1: Jupyter Notebook**
   ```bash
   jupyter notebook
   # Open the .ipynb file and run all cells
   ```

2. **Option 2: Python Script**
   ```bash
   python iris_svm_classification.py
   ```

3. **Option 3: Google Colab**
   - Upload the notebook to Google Colab
   - Run all cells (no installation needed)

## Project Structure

```
iris-svm-classification/
│
├── iris_svm_classification.ipynb    # Main notebook
├── iris_svm_classification.py       # Python script version
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

## Key Insights

### Why SVM Works Well for Iris Dataset:

1. **Linearly Separable Classes**
   - Iris Setosa is completely separable from the other two species
   - Versicolor and Virginica have some overlap but are mostly separable

2. **Clear Feature Boundaries**
   - Distinct differences in petal and sepal measurements
   - Four features provide good discrimination

3. **Small Dataset**
   - 150 samples is sufficient for SVM
   - Well-balanced classes (50 samples each)

4. **High-Dimensional Space**
   - SVM excels in 4D feature space
   - Finds optimal hyperplane efficiently

## Learning Outcomes

This project demonstrates:
- Loading and exploring built-in scikit-learn datasets
- Data preprocessing and feature-target separation
- Visualization of multi-class data
- Training/testing split methodology
- SVM classifier implementation
- Model evaluation and accuracy measurement

## Future Improvements

- [ ] **Hyperparameter Tuning**
  - Experiment with different kernels (linear, RBF, polynomial)
  - Tune C parameter (regularization)
  - Optimize gamma parameter

- [ ] **Cross-Validation**
  - Implement k-fold cross-validation
  - Get more robust accuracy estimates

- [ ] **Feature Importance**
  - Analyze which features contribute most
  - Try feature selection techniques

- [ ] **Advanced Visualizations**
  - 3D scatter plots
  - Decision boundary visualization
  - Confusion matrix heatmap

- [ ] **Compare Models**
  - Test against other classifiers (KNN, Decision Trees, Random Forest)
  - Performance comparison

- [ ] **Model Deployment**
  - Create a simple web interface
  - Build a prediction API



## Additional Resources

- [Scikit-learn SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)
- [Iris Dataset Information](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
- [Understanding SVM](https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/)

## Contributing

Contributions are welcome! Feel free to:
- Add more visualization techniques
- Implement hyperparameter tuning
- Compare with other classification algorithms

## Author

[Ojumu Oluwabukola]
- GitHub: [@bouqui-x](https://github.com/bouqui-x)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Iris dataset: Fisher's classic dataset (1936)
- Scikit-learn library for machine learning tools
- Tech4Dev WomenTechster data science and engineering community
