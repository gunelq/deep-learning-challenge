# Alphabet Soup Charity Report

## Overview of the Analysis
The aim of this analysis is to create machine learning models and neural networks that can forecast the likelihood of applicants succeeding when funded by Alphabet Soup. Essentially, the challenge is to develop a tool that assists Alphabet Soup in identifying the applicants most likely to thrive in their initiatives.




### Data Overview
The dataset contains more than 34,000 organizations that have received funding from Alphabet Soup over the years. The metadata for each organization has been captured with the following columns:

**EIN and NAME:** Identification columns

**APPLICATION_TYPE:** Alphabet Soup application type

**AFFILIATION:** Affiliated sector of industry

**CLASSIFICATION:** Government organization classification

**USE_CASE:** Use case for funding

**ORGANIZATION:** Organization type

**STATUS:** Active status

**INCOME_AMT:** Income classification

**SPECIAL_CONSIDERATIONS:** Special considerations for application

**ASK_AMT:** Funding amount requested

**IS_SUCCESSFUL:** Whether the money was used effectively


 Machine Learning Process
The process for building and evaluating binary classifier models included the following stages:

1. Data Preprocessing:
1. Target Variable:
The target variable for this model is IS_SUCCESSFUL.

   *- 0:* Applicants' funding not successful (16,038 instances)
   
   *- 1:* Applicants' funding successful (18,261 instances)
   
2. Feature Variables:


   **- Step 2: Compile, Train, and Evaluate the Model:**  The features are `APPLICATION_TYPE`, `AFFILIATION`,`CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, and `ASK_AMT`.
   
   **- Step 3: Optimize the Model:** The features are the same as Step 2, with the addition of `NAME`. In the Keras Turner Optimization Model, the `NAME` variable was excluded.

 3. Dropped Columns:

The columns **`EIN`** and **`NAME`** were dropped to preprocess the data


2. Compiling, Training, and Evaluating the Model:

In this phase, neural network models were created using early stopping to prevent overfitting.

 Model 1:

   **- 1 Hidden Layer:** 100 neurons with ReLU activation
   
   **- 2 Hidden Layer:** 50 neurons with ReLU activation
   
   **- Output Layer:** 1 neuron with Sigmoid activation

   **- Epochs:** 200
   
   **- Accuracy:** 0.7262
   
   **- Loss:** 0.5550
   
   
- Justification: The ReLU activation function utilized in the hidden layers enables the model to capture intricate relationships, whereas the Sigmoid function in the output layer is appropriate for tasks involving binary classification.





 Model 2 (Keras Turner):

*Hyperparameters:*
- First Units: 9
  
- Number of Layers: 3
- Units: 9, 3, 3
- Activation: Sigmoid
- Epochs: 20
  
**- Accuracy:** 0.7289

**- Loss:** 0.5634

- Justification: This model employed Keras Tuner to optimize hyperparameters. Adjustments were made to the layers and activation functions to enhance the model's performance and architecture.





Model 3:

   **- First Hidden Layer:** 100 neurons with ReLU activation
   
   **- Second Hidden Layer:** 100 neurons with ReLU activation
   
   **- Third Hidden Layer:** 50 neurons with ReLU activation
   
   **- Fourth Hidden Layer:** 20 neurons with ReLU activation
   
   **- Output Layer:** 1 neuron with Sigmoid activation
   
   **- Epochs:** 100
   
   **- Accuracy:** 0.7562
   
   **- Loss:** 0.4906

 
Justification: This model concentrated on optimizing the quantity of hidden layers and neurons, along with incorporating the NAME column. This configuration contributed to an increase in accuracy.





Model 4:
  **- First Hidden Layer:** 20 neurons with ReLU activation
  
  **- Second Hidden Layer:** 10 neurons with Tanh activation
  
  **- Third Hidden Layer:** 5 neurons with Tanh activation
  
  **- Output Layer:** 1 neuron with Sigmoid activation
  
  **- Epochs:** 50
  
  **- Accuracy:** 0.7541
  
  **- Loss:** 0.4908
  


- Justification: This model decreased the number of hidden layers and neurons. The selection of activation functions (ReLU and Tanh) was informed by the results from Keras Tuner. Including the NAME column played a role in attaining improved accuracy.





Summary:

This analysis sought to forecast the success of applicants receiving funding from Alphabet Soup through the use of neural network models. The dataset included more than 34,000 organizations, featuring a range of metadata attributes. The target variable was binary, indicating whether the funding was successful or not. Multiple machine learning models were trained to develop a classifier that could effectively predict success.




Model Results:
Model 1 acted as the baseline, attaining an accuracy of 72.62%. Although it demonstrated moderate performance, the excessive number of neurons in the hidden layers raised concerns regarding overfitting, as it did not result in substantial improvements in accuracy.


Model 2 (Keras Tuner) provided a slight improvement by optimizing hyperparameters, reaching an accuracy of 72.89%. This model utilized fewer neurons but did not yield significant enhancements in performance, suggesting that more advanced tuning may be required to further boost accuracy.



Model 3 implemented a deeper architecture with four hidden layers and attained an accuracy of 75.62%. The model's intricate structure enabled it to capture more nuanced patterns in the data, making it the top-performing model to date.



Model 4 decreased the number of layers and neurons while employing both ReLU and Tanh activation functions. It achieved a similar accuracy of 75.41%, illustrating that a simpler model can perform nearly as well as the more complex Model 3, thus offering greater computational efficiency.


   
Both Model 3 and Model 4 achieved the highest results, with accuracies nearing 76%. Model 3’s deeper architecture was able to capture more complexity in the data, whereas Model 4’s simplicity and reduced number of layers provided an efficient alternative without compromising performance. Despite these improvements, the accuracy reached a plateau around 76%, suggesting a limit to the model's performance with the existing dataset and architecture.

To surpass the 75% accuracy threshold, further enhancements could be investigated through alternative optimization techniques, additional feature engineering, or the exploration of different machine learning models.


Given the moderate performance of neural networks, alternative models such as the Random Forest Classifier and XGBoost Classifier may yield better results, especially with large datasets like the one utilized here:

The Random Forest Classifier achieved an accuracy of 74%, accompanied by strong recall and precision values, positioning it as a robust option for predicting success.


    - **Accuracy:** 74%
    - **Precision (Class 1):** 0.73
    - **Recall (Class 1):** 0.81
    - **AUC:** 0.7993
    - **f1-score (Class 1):** 0.77
      
The XGBoost Classifier, recognized for its efficiency with large datasets, attained an accuracy of 75%, comparable to the neural networks but demonstrating a higher recall.


    - **Accuracy:** 75%
    - **Precision (Class 1):** 0.73
    - **Recall (Class 1):** 0.85
    - **AUC:** 0.8178
    - **f1-score (Class 1):** 0.79
 


Both neural networks and alternative machine learning models such as Random Forest and XGBoost exhibited comparable performance, with XGBoost showing slightly superior recall and AUC values. It is recommended to consider Models 3 and 4 for predicting applicant success; however, integrating Random Forest or XGBoost could enhance model performance by more effectively managing complex features.



