
**Deep Learning Challenge: : Alphabet Soup Nonprofit Foundation
**

This deep learning challenge focused on examining a dataset from the nonprofit organization Alphabet Soup to develop a binary classifier that forecasts which funding applicants are most likely to succeed in their projects. The report emphasizes the key variables and data details that were essential throughout the analysis.




Key Steps to Completing the Challenge:

Step 1: Preprocess the Data

Loaded the charity_data.csv into a Pandas DataFrame and identified:

1. The target variable for the model.
The feature variable(s) for the model.


2. Dropped the EIN and NAME columns.


3.Determined the count of unique values for each column.



4.For columns containing more than 10 unique values, identified the number of data points corresponding to each unique value.



5. Based on the number of data points, combined "rare" categorical variables into a new category, "Other", and verified the replacement was successful.


6. Encoded categorical variables using pd.get_dummies().



7.Split the preprocessed data into a feature array, X, and a target array, y. Then, utilized the train_test_split function to separate the data into training and testing sets.



8.Scaled the training and testing feature datasets by creating an instance of StandardScaler, fitting it to the training data, and transforming the data.





Step 2: Compile, Train, and Evaluate the Model

1. Continued using Google Colab to carry out the preprocessing steps from Step 1. Created a neural network model using TensorFlow and Keras, specifying the number of input features and nodes for each layer.



2. Designed two hidden layers and chose suitable activation functions (Model 1).



3. Added an output layer with the correct activation function.

4. Verified the structure of the model.


5. Compiled and trained the model.


6. Created a callback to save the model's weights every five epochs.


7. Evaluated the model using the test data to assess loss and accuracy.


8. Saved and exported the results to an HDF5 file, named AlphabetSoupCharity.h5.



Step 3: Optimize the Model


Objective: Optimize the model to achieve a target predictive accuracy of over 75%.

In this step, the following actions were performed:


Included the variable NAME, except in the Keras Tuner optimized model.

Reduced the number of epochs in one model.

Adjusted the number of neurons in the hidden layers, either by adding or reducing them.

Increased the number of hidden layers.

Tested various activation functions for the hidden layers (Tanh, ReLU, Sigmoid).

Note: Over 12 models were evaluated, both with and without the NAME variable. The final set of four models, including the initial unoptimized one, demonstrated the highest accuracy.



Alternative Machine Learning Models

While the challenge focused on a deep learning approach, I also investigated alternative machine learning models to compare the results. Specifically, I utilized the Random Forest Classifier and XGBoost Classifier, which offered valuable insights regarding accuracy and model performance.


1. Random Forest Classifier:

This ensemble learning method combines multiple decision trees to enhance predictive accuracy. It was especially effective in addressing the imbalanced dataset and contributed to reducing overfitting by averaging the predictions from various trees.



The model exhibited strong performance, making it a suitable alternative for binary classification tasks.




2. XGBoost Classifier:

XGBoost is a robust implementation of gradient boosting that constructs models sequentially by optimizing the residuals from prior models. It demonstrated strong performance on the tabular data utilized in this challenge.


The model effectively managed the complexity of the dataset and achieved competitive results when compared to the neural network models.


Both of these models offered interpretable and quick-to-train alternatives to deep learning. They were valuable for validating results and enhancing overall model performance. By comparing the outcomes of these models with the deep learning approach, I was able to evaluate the strengths of both techniques in terms of accuracy, training time, and computational efficiency.



Resources and Tools:


Pandas

scikit-learn


Google Colab

NumPy

TensorFlow

Keras



