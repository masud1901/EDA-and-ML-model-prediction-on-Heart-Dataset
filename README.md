# All-in-one-classification-model
I have been learning about data science and machine learning for quite a while now, and I'm excited to share my progress. So far, I've focused on mastering supervised learning, as it's a great starting point in this field, and I have ambitious plans for the future.

In this Jupyter notebook, I've created an all-in-one binary classification template, that includes a total of seven types of classification algorithms along with their comparative analysis.

For starters, I used a clean dataset, as wanted to understand the process and not get into the whole preprocessing stuff. The dataset contains columns like user ID, gender, age, estimated salary, and whether the person made a purchase or not. The task was to predict, based on age and estimated salary, whether a person would purchase our required product or not. Thus, it is a binary classification business problem.

For implementing the classification algorithms and preprocessing, I relied on the popular sklearn library, along with numpy, pandas,seaborn, and matplotlib.

To streamline the process, I created a function that takes X_train, y_train, and X_test as parameters, returning two dictionaries: y_pred and models. This function performs the training and generates the y_pred dictionary through an iterative process.

After that, I constructed a function that takes y_true and y_pred as input and outputs a dictionary representing the confusion matrix. I then used seaborn heatmap to visualize and present the results in a single figure.

Then I created a larger figure using matplotlib to showcase all seven training sets along with their corresponding decision boundaries.

Based on the figures and analysis, I would suggest that the Naive Bayes and Support Vector classifier models are the best fits, and either of them would be ideal for further analysis.
