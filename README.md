# House-Price-Prediction-Using-Machine-Learning

## Experimental Results and Comparison

### CLASSIFICATION
Overall Classification Performance Comparison
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|---|---|---|---|---|
| RandomForest | 0.9896 | 0.9860 | 0.9876 | 0.9868 |
| XGBoost | 0.9889 | 0.9859 | 0.9863 | 0.9861 |
| LightGBM | 0.9883 | 0.9842 | 0.9846 | 0.9844 |
| CatBoost | 0.9861 | 0.9836 | 0.9817 | 0.9827 |

![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/d1b1ba6224c6477b182fc6c7cb5a4f9397e1abf7/House%20Price%20Prediction/Main/Classification_Output/Screenshot%202026-01-15%20132619.png)

Per-Class Performance for RandomForest (Best Model)

Class | Precision | Recall | F1-Score | Support  
------- | -------- | -------- | -------- | --------  
High | 0.9916 | 0.9916 | 0.9916 | 4184  
Low | 0.9747 | 0.9811 | 0.9779 | 1218  
Medium | 0.9915 | 0.9900 | 0.9908 | 5210  


![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/3ded46bab81d1cae76fc3478e46509f6ae4ff109/House%20Price%20Prediction/Main/Classification_Output/image.png)

### REGRESSION
Table 4.20: Analysis Low Price Range

| Model        | MAE   | RMSE   | R2_train | R2_test |
|--------------|-------|--------|----------|---------|
| RandomForest | 3,559 | 12,001 | 0.9929   | 0.9520  |
| XGBoost      | 4,065 | 12,517 | 0.9952   | 0.9478  |
| LightGBM     | 4,589 | 12,598 | 0.9893   | 0.9471  |
| CatBoost     | 4,196 | 11,577 | 0.9921   | 0.9553  |
![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/63d0dd808fa1fbaa8bd164650a87a612a1956b1b/House%20Price%20Prediction/Main/Regression_Output/Regression_Low_Price_Range.png)

![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/1523f3a62a432037c428aa713780cf527934ebf6/House%20Price%20Prediction/Main/Regression_Output/Regression_Medium_Price_Range.png)
![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/7f8b8b1bb7d183e182bc61d64375271ea6100029/House%20Price%20Prediction/Main/Regression_Output/Regression_High_Price_Range.png)

## High-Level Architecture Diagram
![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/81a48ea4ce2aba6bb330c889bcf3712383d7d101/House%20Price%20Prediction/Main/High-Level%20Architecture%20Diagram/image.png)

Performance of the Regression Models by Price Range
![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/74de96dabf1c5992bb3a2b007495d16e679a0cbd/House%20Price%20Prediction/Main/Regression_Output/image.png)
