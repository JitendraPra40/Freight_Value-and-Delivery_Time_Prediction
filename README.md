# **Freight Value Prediction using Machine Learning**

## **Project Overview**
This project aims to predict the shipping cost (**freight value**) of eCommerce products based on product dimensions, weight, price, and geographic location. We explore multiple machine learning models and evaluate their performance using relevant metrics.

---

## **Dataset**
The dataset used contains:
- Product details: weight, dimensions (length, width, height), and price.
- Geographic information: customer and seller states.
- Target variable: **freight_value** (shipping cost).

### **Features Selected**
| **Feature**                  | **Reason for Selection**                         |
|------------------------------|-------------------------------------------------|
| `product_weight_g`           | Weight directly affects shipping costs.         |
| `product_length_cm`          | Dimensions influence shipping costs.            |
| `product_height_cm`          | Larger items cost more to ship.                 |
| `product_width_cm`           | Width adds to size-based cost.                  |
| `price`                      | High-priced products may require special care.  |
| `review_score`               | Customer reviews may reflect delivery quality.  |
| `customer_state`             | Impacts shipping distance.                      |
| `seller_state`               | Determines the origin of shipping.              |
| `seller_zip_code_prefix`     | Additional geographic detail.                   |
| `customer_zip_code_prefix`   | Additional geographic detail.                   |
| **Target**: `freight_value`  | Shipping cost to be predicted.                  |

---

## **Tools and Libraries**
The following tools and libraries are used in this project:

- **Data Analysis & Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**:
  - RandomForestRegressor
  - Linear Regression
  - Polynomial Regression
- **Evaluation Metrics**: R² Score, Mean Absolute Error (MAE)

---

## **Workflow**

### **1. Data Cleaning**
- **Duplicates**: Identified and removed duplicate rows.
- **Missing Values**: Removed rows with null values.
- **Outlier Removal**: Handled outliers in `freight_value` using the **IQR method**.

### **2. Feature Engineering**
- One-Hot Encoding: Converted `customer_state` and `seller_state` into numerical features using **one-hot encoding**.
- Removed irrelevant or duplicate features.

### **3. Model Development**

#### **a. Random Forest Regressor**
- **Model**: A RandomForestRegressor with 100 estimators.
- **Results**:
  - **R² Score**: 0.73
  - **Mean Absolute Error (MAE)**: 1.76

#### **b. Linear Regression**
- **Model**: A basic Linear Regression.
- **Results**:
  - **R² Score**: 0.53
  - **Mean Absolute Error (MAE)**: 2.7

#### **c. Polynomial Regression**
- **Model**: Transformed features using PolynomialFeatures (degree=2).
- **Results**:
  - **R² Score**: ~0.62
  - **Mean Squared Error**: ~12.47

---

## **Evaluation Metrics**
We evaluate model performance using:
1. **R² Score**: Proportion of variance explained by the model.
2. **Mean Absolute Error (MAE)**: Measures average absolute prediction error.

### **Model Comparison**
| **Model**                  | **R² Score** | **MAE** |
|----------------------------|--------------|---------|
| **Random Forest**          | 0.73         | 1.76    |
| **Linear Regression**      | 0.53         | 2.7     |
| **Polynomial Regression**  | ~0.62        | ~12.47  |

---

## **Key Insights**
1. **Product Weight and Dimensions**: Key drivers of freight value.
2. **Geographic Features**: Both customer and seller locations significantly influence costs.
3. **Model Performance**: RandomForestRegressor is the best-performing model.

---

## **Future Improvements**
1. **Feature Engineering**:
   - Calculate distance between customer and seller locations.
   - Add interaction features (e.g., product volume).
2. **Hyperparameter Tuning**:
   - Optimize RandomForest parameters using **GridSearchCV**.
3. **Advanced Models**:
   - Test boosting algorithms such as **XGBoost** and **LightGBM**.
4. **Outlier Handling**:
   - Use log transformations instead of dropping outliers.

---

## **Running the Project**
1. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Place `eCommerce_dataset.csv` in the project directory.
3. Run the Python script to execute the workflow.

---

## **Author**
**Jitendra Prajapati**  
For queries, contact me at: [jitendra.pra40@gmail.com](mailto:jitendra.pra40@gmail.com)

-------------------------------------------------------------------------------------------------


# Delivery Time Prediction Using Machine Learning

## Project Overview
This project focuses on predicting delivery time for eCommerce orders using machine learning techniques. A cleaned and feature-engineered dataset is utilized to train and evaluate models, including Random Forest Regressor and Linear Regression. Outliers were removed, derived features were created, and one-hot encoding was applied to categorical variables for effective model training.

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Feature Engineering](#feature-engineering)
3. [Data Cleaning](#data-cleaning)
4. [Modeling Techniques](#modeling-techniques)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Results](#results)
7. [How to Run the Project](#how-to-run-the-project)
8. [Technologies Used](#technologies-used)

---

## Dataset Overview
The dataset contains eCommerce order information such as shipping dates, delivery dates, customer and seller states, product dimensions, and freight values. The key target variable is **delivery time**, calculated as the difference between the order purchase date and the delivery date.

### Features Selected
| Feature                        | Description                                      |
|--------------------------------|--------------------------------------------------|
| `freight_value`                | Cost of freight/shipping                         |
| `product_weight_g`             | Product weight in grams                          |
| `product_length_cm`            | Product length in centimeters                    |
| `product_height_cm`            | Product height in centimeters                    |
| `product_width_cm`             | Product width in centimeters                     |
| `customer_state`               | State where the customer resides                 |
| `seller_state`                 | State where the seller resides                   |
| `order_purchase_timestamp`     | Timestamp when the order was placed              |
| `order_delivered_customer_date`| Date when the product was delivered              |
| `shipping_limit_date`          | Seller's shipping deadline                       |
| `order_estimated_delivery_date`| Estimated delivery date                          |
| **Derived Features**           | Calculated columns for improved predictions      |

---

## Feature Engineering
Derived features were created to enhance model performance:
1. **`delivery_time`**: (Target) Number of days between `order_purchase_timestamp` and `order_delivered_customer_date`.
2. **`Shipping Delay`**: Difference between `shipping_limit_date` and `order_purchase_timestamp`.
3. **`Estimated - Actual Delivery`**: Difference between `order_estimated_delivery_date` and `order_delivered_customer_date`.

Features such as dates were converted to **datetime format**.

---

## Data Cleaning
1. **Handling Null Values**:
   - All rows with null values were dropped.
2. **Outlier Removal**:
   - Outliers were identified and removed using the **IQR (Interquartile Range)** method on the `delivery_time` column.
3. **Duplicate Removal**:
   - Duplicate records were dropped.

---

## Modeling Techniques
The following machine learning models were implemented:

1. **Random Forest Regressor**
   - Ensemble model trained with 100 estimators.
2. **Linear Regression with Polynomial Features**
   - Polynomial degree = 2.
3. **Standard Linear Regression**

### Train-Test Split
- 80% training and 20% testing.

---

## Evaluation Metrics
Models were evaluated using the following metrics:
- **R² Score**: Measures the proportion of variance explained by the model.
- **Mean Absolute Error (MAE)**: Average of absolute differences between predictions and actual values.
- **Mean Squared Error (MSE)** (for polynomial regression).

---

## Results
| Model                              | R² Score | MAE     |
|------------------------------------|------------|---------|
| **Random Forest Regressor**        | 0.76       | 3.51    |
| **Linear Regression (Polynomial)** | 0.76       | 3.73    |
| **Linear Regression**              | 0.68       | 3.96    |

- Random Forest performed the best with an **R² score of 0.76**.

---

## How to Run the Project
### Prerequisites
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`

### Steps
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <project-folder>
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset file `eCommerce_dataset.csv` in the working directory.
4. Run the script:
   ```bash
   python delivery_time_prediction.py
   ```
5. Outputs such as evaluation metrics and visualizations will be displayed.

---

## Technologies Used
- **Python**: Core programming language
- **pandas**: Data cleaning and manipulation
- **numpy**: Numerical computations
- **matplotlib & seaborn**: Data visualization
- **scikit-learn**: Machine learning modeling

---

## Visualizations
1. **Boxplots**: To identify and remove outliers.
2. **Histograms**: To visualize the distribution of delivery time.

---

## Future Improvements
- Add more location-based features (e.g., distance between customer and seller).
- Optimize hyperparameters for the Random Forest model.
- Experiment with advanced models such as XGBoost or Gradient Boosting Regressors.

---

## Acknowledgements
- Dataset provided for eCommerce delivery time analysis.
- Open-source libraries and tools for data science.

---

## Author
**Jitendra Prajapati**  
Contact: [jitendra.pra40@gmail.com](mailto:jitendra.pra40@gmail.com)


---------------------------------------------------------

# E-Commerce Orders Analysis

## Project Overview
This project analyzes an e-commerce dataset to derive insights into product performance, customer behavior, payment preferences, and review patterns. The analysis involves data cleaning, handling outliers, and visualizing trends using Python libraries such as `Pandas`, `NumPy`, `Matplotlib`, and `Seaborn`.

---

## Dataset Description
The analysis is based on multiple datasets from an e-commerce platform:

1. **olist_order_items_dataset.csv** - Order details and product IDs.
2. **olist_products_dataset.csv** - Product category details.
3. **product_category_name_translation.csv** - Translations of product categories to English.
4. **olist_orders_dataset.csv** - Order statuses and timestamps.
5. **olist_order_reviews_dataset.csv** - Customer review scores.
6. **olist_order_payments_dataset.csv** - Payment details, including methods and installments.

---

## Goals of the Project
1. Identify the top product categories by:
   - Order counts
   - Revenue
   - Cancellations
   - Positive reviews
2. Analyze customer review score distribution.
3. Understand payment method preferences and installment usage for credit card payments.
4. Visualize the results to derive actionable insights.

---

## Steps Performed

### 1. Data Loading and Merging
- **Merged datasets** using LEFT JOINS on relevant keys like `order_id` and `product_id`.
- Retained essential columns such as `order_id`, `product_id`, `product_name`, `price`, `payment_type`, `review_score`, and `order_status`.

### 2. Data Cleaning
- **Duplicate rows** were removed.
- **Missing values** were identified and handled.
- Data types were converted for consistency (e.g., review scores and payment installments to integers).

### 3. Outlier Detection and Handling
- Outliers in the `price` column were visualized using:
   - **Boxplots**
   - **Histograms**
   - **Violin plots**
- Applied **capping** using the 3rd and 97th percentiles to handle extreme outliers.

### 4. Analysis and Visualization

#### Case 1: Product Categories by Order Count and Revenue
- Identified and visualized **top 10 product categories** by:
   - Order frequency (bar chart)
   - Revenue (gradient-colored bar chart)

#### Case 2: Product Categories with Most Cancellations
- Analyzed `order_status` for "canceled" orders to determine frequently abandoned categories.

#### Case 3: Review Analysis
- Identified products with the most **5-star reviews**.
- Created a **pie chart** for review score proportions.

#### Case 4: Payment Method and Installment Preferences
- Visualized the distribution of **payment methods** using a pie chart.
- For credit card payments, identified the most preferred **installment plans**.

---

## Key Findings
1. **Top Products**: Certain categories dominate in terms of orders and revenues.
2. **Cancellations**: Some product categories experience frequent order cancellations.
3. **Reviews**: A majority of customers provide **5-star reviews**, but other scores reveal room for improvement.
4. **Payments**:
   - **Credit card** is the most popular payment method.
   - Installments are preferred, with fewer installments being more common.

---

## Technologies Used
- **Python Libraries**:
   - `pandas` - Data cleaning and transformation
   - `numpy` - Numerical computations
   - `matplotlib` & `seaborn` - Data visualization
- **Tools**: Jupyter Notebook

---

## Visualizations
Key insights are presented using:
- **Bar Charts**: Product order counts and revenue analysis.
- **Boxplots & Histograms**: Outlier detection and price distributions.
- **Pie Charts**: Review score proportions and payment method distributions.
- **Violin Plots**: Price distribution across product categories.

---

## How to Run the Code
1. Ensure all datasets are located in the `dataset/` folder.
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```
3. Run the notebook using Jupyter Notebook or any Python IDE that supports `.ipynb` files.

---

## Author
**Jitendra Prajapati**

---

## Conclusion
This analysis provides valuable insights into customer behavior, product performance, and payment trends. These findings can be used by e-commerce businesses to improve product offerings, reduce cancellations, and enhance customer satisfaction.



# E-Commerce Data Analysis Project

## Table of Contents
- [Project Overview](#project-overview)  
- [Dataset Description](#dataset-description)  
- [Objective](#objective)  
- [Technologies Used](#technologies-used)  
- [Steps in Analysis](#steps-in-analysis)  
- [Key Visualizations](#key-visualizations)  
- [Results and Insights](#results-and-insights)  
- [How to Run the Project](#how-to-run-the-project)  
- [Folder Structure](#folder-structure)  
- [Future Scope](#future-scope)  
- [Author](#author)  
- [Acknowledgements](#acknowledgements)

---

## Project Overview
This project analyzes a real-world **E-Commerce Dataset** from the Brazilian e-commerce platform **Olist**. The goal is to extract actionable insights, visualize trends, and identify key business metrics such as sales performance, customer behavior, product analysis, and review impact.

---

## Dataset Description
The analysis uses the following datasets:
- **Orders**: Details on order status, timestamps, and customer IDs.
- **Products**: Product names, categories, and prices.
- **Order Payments**: Payment types and transaction values.
- **Order Reviews**: Customer reviews and ratings.
- **Customers**: Unique customer IDs and geolocation.
- **Sellers**: Seller information.
- **Geolocation**: Customer and seller locations.

You can download the datasets [here](https://www.kaggle.com/olistbr/brazilian-ecommerce).

---

## Objective
The main goals of the project include:
1. Understanding sales trends and customer purchasing behavior.  
2. Analyzing product categories to identify top-performing products.  
3. Investigating customer satisfaction through reviews and ratings.  
4. Visualizing seller and customer geolocation data.  

---

## Technologies Used
- **Python**: Data manipulation and analysis  
- **Pandas**: Data preprocessing  
- **NumPy**: Numerical analysis  
- **Matplotlib/Seaborn**: Data visualization  
- **Power BI**: Interactive dashboards  
- **Jupyter Notebook**: Development environment  

---

## Steps in Analysis
1. Data Cleaning and Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Key Metrics Calculation  
4. Visualization of Insights  
5. Building an Interactive Dashboard in Power BI  

---

## Key Visualizations
- **Sales Trends**: Monthly and yearly order distribution.  
- **Top Products**: Best-selling products by category.  
- **Customer Behavior**: Purchase frequency and review distribution.  
- **Geographical Analysis**: Customer and seller locations on maps.  

---

## Results and Insights
- Identified top-performing product categories.  
- Sales peak during certain months of the year.  
- Customer reviews correlate with product sales performance.  
- Most customers are concentrated in specific geographical regions.  

---

## How to Run the Project
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/ecommerce-analysis.git
