# House-Price-Prediction-Using-Machine-Learning

## ABSTRACT
This project presents a two-stage hybrid approach for accurate house price prediction by combining classification and regression techniques. The dataset for this study was gathered manually by scraping property listings from the iProperty website, which includes data from various states in Malaysia. Each listing was processed to extract important details such as built-up area, property type, number of bedrooms and bathrooms, furnishing status, number of parking spaces, and price per square foot. The collected data was then merged into a comprehensive dataset and underwent cleaning to remove missing or inconsistent entries. In the first stage, houses are classified into three price categories which is Low, Medium, and High by using machine learning, with Random Forest delivering the best performance. Several machine learning models, including Random Forest, XGBoost, LightGBM, and CatBoost, were used to classify house prices into quartiles, and their performance was evaluated using classification metrics such as accuracy, precision, recall, and F1-score. In the second stage, three separate regression models are built, one for each price category. Each model is trained to learn the unique patterns and behaviours specific to its respective price range. The final system integrates the classification and regression stages, where the predicted price category determines the corresponding regressor to generate an accurate house price prediction. This approach leverages key house attributes to enhance prediction accuracy and align more closely with real property market trends. The use of this combined multi-regression strategy improves the overall predictive performance, reduces errors, and better reflects the actual property market behaviour.


## Dataset Description

The dataset for this research was collected from the iProperty Malaysia website, covering 13 states and federal territories: Johor, Kedah, Kelantan, Melaka, Penang, Perak, Putrajaya, Sabah, Sarawak, Selangor, Terengganu, Labuan, and Pahang. The data was gathered using Selenium WebDriver in Python, which automates browser actions to mimic human browsing behavior, inspect web pages, and extract property details.

The dataset for this research was collected from the iProperty Malaysia website using Selenium WebDriver in Python, which automates browser interactions for web scraping. The script handled dynamic scrolling and pagination to capture listings spread across multiple pages. User-agent spoofing was implemented to avoid being blocked by the website, mimicking real browser requests.

XPath was used to extract key property details, such as price, location, and number of bedrooms. The script also handled lazy loading and infinite scrolling to ensure all dynamically loaded data was captured by continuously comparing page height. Pagination was managed by clicking the "Next" button to access additional pages.

After scraping, the data was cleaned and preprocessed by removing duplicates, handling missing values, and standardizing formats. A new price range variable was created by categorizing prices into low, medium, and high ranges using quartiles. The cleaned data was then saved in CSV format, ready for use in training machine learning models for both classification (predicting price range) and regression (estimating exact prices).

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

SHAP Beeswarm Plot (Global Feature Importance)
![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/6eb5638ff6318952301d4fd908f72039c9c9debd/House%20Price%20Prediction/Main/Result/SHAP%20Beeswarm%20Plot%20(Global%20Feature%20Importance)/image.png)

SHAP (SHapley Additive exPlanations) analysis was employed to understand the model's decision-making process. The SHAP summary plot reveals that features such as 'Price per SqFt' and 'Built-up' area have the greatest impact on the model's predictions. High values (red) push  redictions upward while low values (blue) pull them downward for these top features reflecting a strong positive relationship with price. In contrast, features like 'Furnishing', 'Carpark', and 'Bedroom' show smaller spreads around zero indicating a weaker overall influence. This demonstrates that the model’s predictions are primarily driven by size-related and location-related attributes which aligns with real-world property valuation principles.

iProperty Website Architecture with House Attributes and Pricing Flow for Selangor
![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/a88129835c9aa7db1d97eb71c36fdaf8c085764b/House%20Price%20Prediction/Main/Streamlit/iProperty%20Website%20Architecture%20with%20House%20Attributes%20and%20Pricing%20Flow%20for%20Selangor.png)

User Input Flow in Streamlit for House Attribute Based Price Prediction in Selangor
![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/baae2de9f38166974f12cf5c809938786e1d4f0b/House%20Price%20Prediction/Main/Streamlit/image.png)

