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


Per-Class Performance for RandomForest (Best Model)

Class | Precision | Recall | F1-Score | Support  
------- | -------- | -------- | -------- | --------  
High | 0.9916 | 0.9916 | 0.9916 | 4184  
Low | 0.9747 | 0.9811 | 0.9779 | 1218  
Medium | 0.9915 | 0.9900 | 0.9908 | 5210  


### REGRESSION
Analysis Low Price Range

| Model        | MAE   | RMSE   | R2_train | R2_test |
|--------------|-------|--------|----------|---------|
| RandomForest | 3,559 | 12,001 | 0.9929   | 0.9520  |
| XGBoost      | 4,065 | 12,517 | 0.9952   | 0.9478  |
| LightGBM     | 4,589 | 12,598 | 0.9893   | 0.9471  |
| CatBoost     | 4,196 | 11,577 | 0.9921   | 0.9553  |


Medium Price Range

| Model        | MAE   | RMSE   | R2_train | R2_test |
|--------------|-------|--------|----------|---------|
| RandomForest | 3,655 | 16,932 | 0.9962   | 0.9834  |
| CatBoost     | 5,640 | 18,436 | 0.9929   | 0.9803  |
| LightGBM     | 5,906 | 19,781 | 0.9922   | 0.9773  |
| XGBoost      | 9,266 | 20,009 | 0.9869   | 0.9768  |


High Price Range

| Model        | MAE     | RMSE    | R2_train | R2_test |
|--------------|---------|---------|----------|---------|
| RandomForest | 53,675  | 266,989 | 0.9808   | 0.9528  |
| LightGBM     | 86,868  | 306,793 | 0.9659   | 0.9377  |
| CatBoost     | 86,001  | 310,606 | 0.9670   | 0.9361  |
| XGBoost      | 100,395 | 318,490 | 0.9616   | 0.9329  |


## High-Level Architecture Diagram
![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/81a48ea4ce2aba6bb330c889bcf3712383d7d101/House%20Price%20Prediction/Main/High-Level%20Architecture%20Diagram/image.png)

The two-stage machine learning model for house price prediction utilizes a modular, hierarchical architecture that enhances flexibility and scalability. The first stage involves a Random Forest Classifier to categorize properties into predefined price ranges based on key attributes. The second stage uses separate Random Forest Regression models for each price range, improving prediction accuracy through market segmentation. The system integrates these stages into a coherent workflow, ensuring that new property instances are classified and routed to the appropriate regression model for price estimation. The architecture supports reproducibility and transparent experimentation, offering an efficient and scalable solution for house price prediction.


Random Forest Price Range Confusion Matrix
|       | High | Low | Medium |
| ----------- | ----------- | ----------- | ----------- |
| **Actual High** | 4157      | 1      | 26       |
| **Actual Low**   | 9        | 1193        | 16        |
| **Actual Medium**   | 29        | 25        | 5156        |


Scatter Plot of Actual vs  Predicted Prices for All Ranges
![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/cff7474d1ab89cdbec73b8ad052ea80ed0e6f6bd/House%20Price%20Prediction/Main/Scatter%20Plot%20of%20Actual%20vs%20%20Predicted%20Prices%20for%20All%20Ranges/image.png)

The scatter plot of actual house prices versus predicted prices demonstrates strong alignment with the ideal prediction line (y = x), indicating that the machine learning model generally performs well in estimating house prices. Most properties, particularly those in the low and medium price ranges, show minimal deviation from the ideal line, suggesting high prediction accuracy. However, high-priced properties exhibit larger prediction errors, reflecting the model's difficulty in accurately predicting more expensive houses due to greater variability and complex influencing factors. The increasing forecast error with higher prices indicates heteroscedasticity, where prediction uncertainty grows for higher-priced homes. In conclusion, while the model excels in predicting lower and mid-range house prices, further improvement is needed for high-priced properties, potentially through additional features, more high-end samples, and advanced learning techniques.


Distribution of Residuals per Price Range
![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/5886e26b23eafd4cba17701511e859300a895a26/House%20Price%20Prediction/Main/Result/Distribution%20of%20Residuals%20per%20Price%20Range/image.png)

The residual analysis, which shows the difference between predicted and actual house prices, reveals that the model performs best for low and medium-priced properties. For low-priced houses, residuals are concentrated around zero, indicating accurate predictions with minimal errors. Medium-priced properties show a wider spread of residuals, suggesting increased prediction variance but no strong systematic bias. High-priced properties exhibit the largest residual distribution with long tails, indicating larger prediction errors and higher uncertainty due to factors like premium locations and luxury features not fully captured by the model. In conclusion, the model performs well for lower and mid-range properties, but improvements are needed for high-priced properties by incorporating richer features and more advanced techniques.

Boxplot of Residuals per Price Range
![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/c05ec13b15d204d054685613f661a531d28031e5/House%20Price%20Prediction/Main/Result/Boxplot%20of%20Residuals%20per%20Price%20Range/image.png)

The boxplot of residuals across three house price categories reveals that the model performs well for low-priced houses, with residuals tightly clustered around zero and minimal outliers, indicating accurate and consistent predictions. For medium-priced houses, the residuals show moderate variability, with occasional overestimations and underestimations, but no significant bias. However, for high-priced properties, the residuals exhibit a much wider spread with extreme outliers, indicating significant prediction errors. This reflects the model's difficulty in accurately predicting high-value properties due to factors like luxury features and market dynamics that are not fully captured. Overall, the analysis highlights the model’s strong performance for low and medium-priced homes but suggests that improvements are needed for high-priced properties, including richer feature sets and more advanced techniques.
