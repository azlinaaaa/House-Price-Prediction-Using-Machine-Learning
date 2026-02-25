# 🏠 House Price Prediction Using Machine Learning (Malaysia Market)

## 📌 Project Overview

This project develops a **two-stage hybrid machine learning system** to predict residential house prices in Malaysia using real-world property listings scraped from the iProperty website.

Instead of relying on a single regression model, the system applies a **market segmentation strategy**:

1. **Stage 1 – Price Range Classification (Low / Medium / High)**
2. **Stage 2 – Separate Regression Models for Each Segment**

This approach improves prediction accuracy and better reflects real-world property valuation strategies.

---

## 🎯 Business Objective

Accurate property pricing is critical for:

- Real estate valuation
- Investment decision-making
- Market analysis
- Property technology (PropTech) solutions

This project demonstrates how **data-driven segmentation and modular ML architecture** can improve reliability, scalability, and business alignment in real estate analytics.

---

## 📊 Dataset Description

![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/936d2de0194e5b04fc8649fab47df037539244db/House%20Price%20Prediction/Main/Web%20Scraping/Extracted%20Attributes/image.png)
The dataset was collected from the **iProperty Malaysia website** using Selenium WebDriver in Python.

### Coverage
Properties from 13 Malaysian states and federal territories:

Johor, Kedah, Kelantan, Melaka, Penang, Perak, Putrajaya, Sabah, Sarawak, Selangor, Terengganu, Labuan, and Pahang.

### Data Collection Process
- Automated browser interaction using Selenium
- Managed dynamic scrolling and lazy loading
- Handled pagination across listing pages
- Implemented user-agent spoofing
- Extracted attributes using XPath

### Extracted Features
- Built-up area  
- Property type  
- Bedrooms  
- Bathrooms  
- Furnishing status  
- Carpark  
- Price per SqFt  
- Location  
- Total price  

### Data Preprocessing
- Removed duplicates
- Handled missing values
- Standardized formats
- Created price segmentation using quartiles (Low / Medium / High)
- Exported cleaned dataset to CSV

---

## 🧠 Model Architecture

### 🔹 Stage 1: Price Classification

Goal: Predict property price range (Low / Medium / High)

Models evaluated:
- Random Forest
- XGBoost
- LightGBM
- CatBoost

**Best Model: Random Forest**

| Metric | Score |
|--------|--------|
| Accuracy | 0.9896 |
| Macro F1-Score | 0.9868 |

This stage identifies the correct market segment before price estimation.

---

### 🔹 Stage 2: Segment-Based Regression

Instead of one global model, three separate regression models were trained — one for each price range.

#### Low Price Range

| Model | R² Test |
|-------|---------|
| CatBoost | 0.9553 |

#### Medium Price Range

| Model | R² Test |
|-------|---------|
| RandomForest | 0.9834 |

#### High Price Range

| Model | R² Test |
|-------|---------|
| RandomForest | 0.9528 |

This segmentation significantly reduces prediction error by allowing each model to learn range-specific patterns.

---

## 📈 Performance Insights

### Strengths
- Very high classification accuracy (~99%)
- Strong regression performance (R² > 0.95 across ranges)
- Excellent alignment between actual vs predicted prices
- Minimal systematic bias in low and medium segments

### Identified Limitation
- Higher prediction variance for luxury properties
- Suggests opportunity for richer features (amenities, premium location indicators)

---


## 🏗 System Workflow

```

User Input
↓
Feature Processing
↓
Random Forest Classifier
↓
Route to Price-Range-Specific Regressor
↓
Final Price Prediction

```

The architecture is modular, scalable, and production-ready.

---

## 💼 Business Impact

This project demonstrates:

- End-to-end machine learning pipeline development
- Real-world data scraping and preprocessing
- Market segmentation strategy implementation
- Multi-model experimentation and evaluation
- Model interpretability and transparency
- Deployment-ready design using Streamlit

iProperty Website Architecture with House Attributes and Pricing Flow for Selangor ![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/a88129835c9aa7db1d97eb71c36fdaf8c085764b/House%20Price%20Prediction/Main/Streamlit/iProperty%20Website%20Architecture%20with%20House%20Attributes%20and%20Pricing%20Flow%20for%20Selangor.png) User Input Flow in Streamlit for House Attribute Based Price Prediction in Selangor ![image alt](https://github.com/azlinaaaa/House-Price-Prediction-Using-Machine-Learning/blob/baae2de9f38166974f12cf5c809938786e1d4f0b/House%20Price%20Prediction/Main/Streamlit/image.png)
---

## 🛠 Tech Stack

- Python  
- Selenium  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- LightGBM  
- CatBoost  
- SHAP  
- Streamlit  

---

## 🚀 Key Takeaways

- Designed a two-stage hybrid ML architecture
- Achieved strong predictive performance across price segments
- Applied business-oriented segmentation strategy
- Built a scalable and interpretable solution
- Demonstrated full ML lifecycle: Data → Modeling → Evaluation → Deployment

---

## 📎 Future Improvements

- Add geospatial encoding (latitude/longitude)
- Include neighborhood-level economic indicators
- Apply advanced ensemble stacking
- Expand to rental price prediction
- Integrate automated model monitoring

---

## 👩‍💻 Author

**Norazlina Mohd Shariff**  
Final-Year Data Science Student  

Passionate about building data-driven systems that solve real-world business problems.
