# Comprehensive Data Analysis and Machine Learning Project

## Table of Contents
- [Comprehensive Data Analysis and Machine Learning Project](#comprehensive-data-analysis-and-machine-learning-project)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Data Visualization](#data-visualization)
    - [Distribution and Outlier Visualization](#distribution-and-outlier-visualization)
  - [Features](#features)
  - [Dependencies](#dependencies)
  - [Machine Learning Models](#machine-learning-models)
    - [Classifiers Implemented](#classifiers-implemented)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Model Evaluation Process](#model-evaluation-process)
    - [Results Storage](#results-storage)
  - [Dependencies](#dependencies-1)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Results](#results)
  - [Code Structure](#code-structure)
  - [Customization](#customization)
  - [Output](#output)
  - [Detailed Component Explanation](#detailed-component-explanation)
    - [Violin Plot](#violin-plot)
    - [Box Plot](#box-plot)
    - [Jittered Points](#jittered-points)
  - [Data Preprocessing](#data-preprocessing)
  - [Performance Considerations](#performance-considerations)
  - [Troubleshooting](#troubleshooting)
  - [FAQs](#faqs)
  - [Version History](#version-history)
  - [Future Improvements](#future-improvements)

## Overview
This project provides a comprehensive suite for data analysis, visualization, and machine learning modeling. It includes tools for visualizing data distributions and outliers, as well as the application and evaluation of multiple machine learning models on the dataset.



## Data Visualization
### Distribution and Outlier Visualization
- Combination of violin plots, box plots, and jittered points
- Interactive Plotly-based display
- High-resolution image export capability



## Features
- **Multi-variable Visualization**: Automatically creates subplots for all columns in the input DataFrame.
- **Combination Plot**: Each variable is represented by:
  - A violin plot showing the probability density of the data
  - A box plot indicating quartiles, median, and potential outliers
  - Jittered points representing individual data points for granular view
- **Interactive Display**: Utilizes Plotly's interactive features for dynamic data exploration.
- **Customizable Layout**: Flexible options for adjusting plot size, colors, and styling.
- **Export Functionality**: Capability to save the plot as a high-resolution PNG image.
- **Scalability**: Designed to handle datasets with varying numbers of variables.
- **Informative Labeling**: Clear subplot titles and axis labels for easy interpretation.

## Dependencies
- Python 3.x
- Plotly (for interactive plotting)
- NumPy (for numerical operations)
- Pandas (for data manipulation, implied in the code)
- Kaleido (for high-quality image export)

## Machine Learning Models

### Classifiers Implemented
A total of 25 different classifiers from various categories were implemented:

1. **Linear Models**
   - Logistic Regression
   - Stochastic Gradient Descent (SGD)
   - Ridge Classifier
   - Passive Aggressive Classifier

2. **Tree-based Models**
   - Decision Tree Classifier
   - Random Forest Classifier
   - Extra Trees Classifier

3. **Ensemble Methods**
   - AdaBoost Classifier
   - Gradient Boosting Classifier
   - Histogram-based Gradient Boosting Classifier
   - Bagging Classifier
   - LightGBM Classifier
   - XGBoost Classifier
   - CatBoost Classifier

4. **Nearest Neighbors**
   - K-Nearest Neighbor Classifier
   - Radius Neighbor Classifier
   - Nearest Centroid Classifier

5. **Naive Bayes Models**
   - Gaussian Naive Bayes
   - Bernoulli Naive Bayes
   - Categorical Naive Bayes
   - Complement Naive Bayes
   - Multinomial Naive Bayes

6. **Support Vector Machines**
   - Support Vector Classifier (SVC)
   - Linear Support Vector Classifier

7. **Neural Networks**
   - Multi-layer Perceptron Classifier

8. **Discriminant Analysis**
   - Linear Discriminant Analysis
   - Quadratic Discriminant Analysis

9. **Other**
   - Gaussian Process Classifier
   - Kernel Ridge Classifier
   - Dummy Classifier (baseline)

### Evaluation Metrics
The models are evaluated using a comprehensive set of metrics:

1. **Accuracy**: Overall correctness of the model
2. **Precision**: Ratio of true positives to total predicted positives
3. **Recall**: Ratio of true positives to total actual positives
4. **F1 Score**: Harmonic mean of precision and recall
5. **AUC-ROC**: Area Under the Receiver Operating Characteristic curve
6. **Cohen's Kappa**: Agreement between predicted and actual classes, accounting for chance
7. **Log Loss**: Logarithmic loss between predicted probabilities and actual class
8. **Average Precision**: Area under the precision-recall curve
9. **Jaccard Score**: Intersection over Union of the predicted and actual positive labels
10. **Balanced Accuracy**: Average of recall obtained on each class
11. **Specificity**: True negative rate
12. **Geometric Mean**: Geometric mean of sensitivity and specificity
13. **Index of Balanced Accuracy (IBA)**: Weighted version of geometric mean
14. **Confusion Matrix Components**: True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
15. **Cross-Validation Scores**: Mean and standard deviation of 10-fold stratified cross-validation

### Model Evaluation Process
1. Each model is trained on the training dataset.
2. Predictions are made on the test dataset.
3. Comprehensive metrics are calculated for each model.
4. 10-fold stratified cross-validation is performed to assess model stability.
5. Results are compiled into a DataFrame for easy comparison and analysis.

### Results Storage
- The evaluation results for all models are stored in a structured DataFrame.
- This allows for easy comparison, visualization, and further analysis of model performances.


## Dependencies
- Python 3.x
- Plotly (for interactive plotting)
- NumPy (for numerical operations)
- Pandas (for data manipulation, implied in the code)
- Kaleido (for high-quality image export)

## Installation
Ensure you have the required packages installed:

```bash
pip install -r requirements.txt
```

For virtual environment users:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install plotly numpy pandas kaleido
```

## Usage
1. Prepare your dataset as a pandas DataFrame named `df`.
2. Import the necessary libraries and paste the provided code into your Python script or Jupyter notebook.
3. Run the script to generate the visualization.
4. The plot will be displayed interactively in your default browser or notebook.
5. A PNG file of the plot will be saved in your working directory.

Example:
```python
import pandas as pd
# Your existing code here
# ...
fig.show()
save_figure(fig)
```

## Results
- **Location**: Results are saved in the `results` directory
``

## Code Structure
1. **Import Statements**: Required libraries are imported.
2. **Save Function Definition**: `save_figure` function is defined for exporting the plot.
3. **Data Preparation**: The input DataFrame is processed to determine subplot layout.
4. **Subplot Creation**: A Plotly subplot figure is initialized.
5. **Plot Generation Loop**: Iterates through DataFrame columns, creating violin and box plots for each.
6. **Layout Customization**: Adjusts the overall appearance of the plot.
7. **Display and Save**: Shows the interactive plot and saves it as an image.

## Customization
- **Color Scheme**: Modify the `colors` variable to change the plot color palette.
- **Plot Dimensions**: Adjust height and width in the `update_layout` function.
- **Subplot Titles**: Customize subplot titles in the `make_subplots` function call.
- **Marker Properties**: Alter size, opacity, and other properties of plot elements.
- **Export Settings**: Modify the `save_figure` function parameters for custom image export.

## Output
- An interactive HTML plot displayed in the default web browser or Jupyter notebook.
- A high-resolution PNG file (default name: "Outlier_Graph.png") saved in the working directory.

## Detailed Component Explanation
### Violin Plot
- Represents the probability density of the data at different values.
- Wider sections indicate a higher probability of data points in that range.

### Box Plot
- Shows the quartiles of the dataset.
- The box represents the interquartile range (IQR).
- The line in the box is the median.
- Whiskers extend to show the rest of the distribution.
- Points beyond the whiskers are potential outliers.

### Jittered Points
- Individual data points are plotted with a slight random offset.
- Provides a view of the raw data distribution, especially useful for smaller datasets.

## Data Preprocessing
- Ensure your DataFrame (`df`) is clean and properly formatted.
- Handle missing values appropriately before visualization.
- Consider normalizing or scaling data if variables are on vastly different scales.

## Performance Considerations
- Large datasets may require significant processing time and memory.
- For very large datasets, consider sampling or using a more performant plotting library.

## Troubleshooting
- **ImportError**: Ensure all required libraries are installed.
- **MemoryError**: For large datasets, try reducing the number of plotted points or use data sampling.
- **Kaleido Issues**: Make sure Kaleido is properly installed for image export functionality.

## FAQs
1. **Q: Can I use this with a CSV file?**
   A: Yes, load your CSV into a pandas DataFrame first: `df = pd.read_csv('your_file.csv')`

2. **Q: How can I change the output image format?**
   A: Modify the `fig.write_image()` call in the `save_figure` function, changing the file extension.


## Version History
- v1.1.0: Added 25 machine learning models and additional visualizations
- v1.0.0: Initial release with basic visualization functionality

## Future Improvements
- Implement automated hyperparameter tuning for ML models
- Add more advanced visualization techniques (e.g., t-SNE, UMAP)
- Develop a web interface for easy model selection and result visualization
