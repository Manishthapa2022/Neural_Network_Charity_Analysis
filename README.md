# Neural_Network_Charity_Analysis

## Overview of the Analysis

As part of this project, I along with Beks have been been tasked to create a binary classifier from the features in the dataset that is capable of predicting whether applicants will be successful if funded by Alphabet Soup. Alphabet Soup's business team has provided me a CSV file named **charity_data.csv** containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. 
The foloowing are the technical deliverables as part of the project:
* Preprocessing Data for a Neural Network model
* Compile, train, and evaluate the Model
* Optimize the Model
* A written report on Neural Network Model
For this project, I will be carrying out my analysis on Google Colabs. 

## Results

### Data Preprocessing
As part of the data preprocessing, the following was done:
* IS_SUCCESSFUL was made the Target.
* For deliverable 1 and 2, EIN and NAME columns were dropped from the dataset as they were considered neither Target nor Feature. For Deliverable 3,additional columns with feature importance less than 1% were dropped. 
* For Deliverable 1 and 2, nine columns that is APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS and ASK_AMT were considered as features, whereas in case of Deliverable 3, 17 columns were selected after running the feature importance algorithm.  

```python
# choose the columns with a relevance below 1% 
importance_df=importance_df[importance_df["Relevance"]<0.01]
least_relevant=importance_df["Features"].tolist()
least_relevant
# dropping all the columns with less than 1 % relevance
X=X.drop(X[least_relevant], axis=1)
print(X.shape)
X.head()
```
![]()

### Compiling, Training and Evaluating the Model

**Accuracy of model after Deliverable 2**

![]()

For this project, I created three models and took the following steps:
* For the first model nn, the number of layers were increased from 2 to 3. For Deliverable 1 and 2 two hidden layers were used that was increased to 3 for Deliverable 3. Three layers were used to improve the capacity/computation efficiency of the model. 
**Accuracy of model after changing the layers**

![]()

* For the second model nn1, the number of neurons were increased to three times the input parameters to increase the weights and ensure good training. 
**Accuracy of the model after increasing neurons**

![]()

* For the third model nn2, the activation function for hidden layer 2 and 3 was changed from relu to sigmoid to boost the accuracy and also significantly reduce both the training and evaluation times. 
**Accuracy of the model after changing activation function**

![]()

* Inn all three models (nn, nn1 and nn2), there was a gradual reduction in the number of the neurons starting from the first to last hidden layer to improve the dynamics of the model. 
* Unfortunaetely, we were unable to achieve the Target model performance of 75%. 
* To improve the overall performance of the model, additional columns with feature importance less than 1% were dropped. After that, three models nn, nn1 and nn2 were created where the number of neurons, no of hidden layers and the activation function were changed. No of epochs were not increased as there was negligible improvement in the accuracy scores at around epoch 30 to 40. 

## Summary

![Overall Results](https://github.com/Manishthapa2022/Neural_Network_Charity_Analysis/blob/main/Analysis/Overall_results.png)

Overall, in spite of the changes made in the models on this dataset, there is negligible improvement in the overall accuracy score and our best model performance was 72.5% and we missed the overall target of 75% by 2.5%. Further analysis of the dataset is required to understand 

