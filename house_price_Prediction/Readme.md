# Housing Price Prediction using Linear Regression

A machine learning project that predicts house prices based on various features using Linear Regression with Scikit-learn.


## 🔍 Overview

**Problem Type**: Supervised Learning - Regression
- **Input**: House features (area, bedrooms, location, amenities)
- **Output**: Continuous value (house price)
- **Algorithm**: Linear Regression
- **Goal**: Predict house prices for new properties

**What the code does:**
1. Loads housing data from CSV file
2. Preprocesses categorical variables using one-hot encoding
3. Splits data into training and testing sets
4. Trains a Linear Regression model
5. Evaluates model performance with multiple metrics
6. Visualizes results with plots
7. Makes predictions for new houses

## 📊 Dataset

**Expected CSV Structure (`housing.csv`):**

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `area` | Numeric | House area in sq ft | 7420, 8960, 9960 |
| `bedrooms` | Numeric | Number of bedrooms | 2, 3, 4 |
| `bathrooms` | Numeric | Number of bathrooms | 1, 2, 3 |
| `stories` | Numeric | Number of floors | 1, 2, 3 |
| `mainroad` | Categorical | Access to main road | "yes", "no" |
| `guestroom` | Categorical | Has guest room | "yes", "no" |
| `basement` | Categorical | Has basement | "yes", "no" |
| `hotwaterheating` | Categorical | Hot water heating | "yes", "no" |
| `airconditioning` | Categorical | Has AC | "yes", "no" |
| `parking` | Numeric | Parking spaces | 0, 1, 2, 3 |
| `prefarea` | Categorical | In preferred area | "yes", "no" |
| `furnishingstatus` | Categorical | Furnishing status | "furnished", "semi-furnished", "unfurnished" |
| `price` | Numeric | **Target variable** - House price | 13300000, 12250000 |

## 💻 Code Explanation

### 1. Library Imports
```python
import pandas as pd       # Data manipulation and analysis
import numpy as np        # Numerical computations
import matplotlib.pyplot as plt  # Basic plotting
import seaborn as sns     # Statistical visualizations
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.linear_model import LinearRegression      # ML algorithm
from sklearn.metrics import mean_squared_error, r2_score  # Evaluation metrics
```

### 2. Data Loading and Preprocessing
```python
data = pd.read_csv("housing.csv")
data = pd.get_dummies(data, drop_first=False)
```

**One-Hot Encoding Explained:**
- Converts categorical variables into numeric format
- Each category becomes a separate binary column (0 or 1)

**Example transformation:**
```
Original:
mainroad = "yes" → mainroad_yes = 1, mainroad_no = 0
mainroad = "no"  → mainroad_yes = 0, mainroad_no = 1

furnishingstatus = "furnished" → 
  furnishingstatus_furnished = 1
  furnishingstatus_semi-furnished = 0  
  furnishingstatus_unfurnished = 0
```

**Why `drop_first=False`?**
- Keeps all dummy columns
- Alternative: `drop_first=True` drops one column to prevent multicollinearity

### 3. Data Exploration
```python
print(data.head())           # First 5 rows
print("Dataset shape:", data.shape)  # (rows, columns)
print(data.info())           # Data types and missing values
```

**What to look for:**
- Number of samples and features
- Data types (numeric vs object)
- Missing values (Non-Null Count)
- Memory usage

### 4. Feature-Target Separation
```python
X = data.drop("price", axis=1)  # Features (independent variables)
y = data["price"]               # Target (dependent variable)
```

**Machine Learning Terminology:**
- **X (Features)**: Input variables used to make predictions
- **y (Target)**: Output variable we want to predict
- **axis=1**: Drop column (axis=0 would drop rows)

### 5. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Parameters Explained:**
- **test_size=0.2**: 20% for testing, 80% for training
- **random_state=42**: Ensures reproducible results (same split every time)

**Why split data?**
- Train on one portion, test on unseen data
- Prevents overfitting
- Gives realistic performance estimate

### 6. Model Training
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

**Linear Regression Formula:**
```
Price = β₀ + β₁×area + β₂×bedrooms + β₃×bathrooms + ... + βₙ×feature_n
```

**What `.fit()` does:**
- Finds optimal coefficients (β values) for each feature
- Minimizes the difference between predicted and actual prices
- Uses Ordinary Least Squares method

### 7. Making Predictions
```python
y_pred = model.predict(X_test)
```
- Uses trained model to predict prices for test data
- Returns array of predicted values

## 📈 Model Evaluation

### 1. Mean Squared Error (MSE)
```python
mse = mean_squared_error(y_test, y_pred)
```
**What it measures:**
- Average of squared differences between actual and predicted values
- **Lower is better** (0 = perfect predictions)
- Units: Price² (e.g., if price in lakhs, MSE in lakhs²)

**Formula:** `MSE = (1/n) × Σ(actual - predicted)²`

### 2. R² Score (Coefficient of Determination)
```python
r2 = r2_score(y_test, y_pred)
```
**What it measures:**
- Proportion of variance in target explained by features
- **Range**: 0 to 1 (higher is better)
- **1.0** = Perfect fit, **0.0** = No better than predicting mean

**Interpretation:**
- 0.8 = Model explains 80% of price variation
- 0.5 = Model explains 50% of price variation

### 3. Custom Accuracy (Tolerance-based)
```python
tolerance = 0.10  # 10%
accurate_preds = np.abs((y_pred - y_test) / y_test) <= tolerance
custom_accuracy = np.mean(accurate_preds)
```
**What it measures:**
- Percentage of predictions within 10% of actual price
- More interpretable than MSE for business use
- Example: 0.75 = 75% of predictions within 10% tolerance

## 📊 Visualizations

### 1. Actual vs Predicted Scatter Plot
```python
plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         color="red", linewidth=2)
```
**Purpose:**
- Shows how close predictions are to actual values
- **Red diagonal line** = perfect predictions (y=x)
- Points closer to line = better predictions

**What to look for:**
- Points scattered around diagonal = good model
- Points far from diagonal = poor predictions
- Patterns in scatter = model bias

### 2. Correlation Heatmap
```python
data_encoded = pd.get_dummies(data, drop_first=True)
sns.heatmap(data_encoded.corr(), annot=False, cmap="coolwarm")
```
**Purpose:**
- Shows relationships between all features
- Identifies which features strongly affect price
- Detects multicollinearity (highly correlated features)

**Color interpretation:**
- **Red/Warm colors**: Positive correlation
- **Blue/Cool colors**: Negative correlation
- **White/Neutral**: No correlation

## 🏠 Making New Predictions

### Step 1: Create New House Data
```python
new_house_raw = pd.DataFrame({
    "area": [8500],
    "bedrooms": [4],
    "bathrooms": [3],
    "stories": [2],
    "mainroad": ["yes"],
    # ... other features
})
```

### Step 2: Apply Same Preprocessing
```python
new_house_encoded = pd.get_dummies(new_house_raw)
new_house_encoded = new_house_encoded.reindex(columns=X.columns, fill_value=0)
```

**Critical steps:**
1. **One-hot encode** new data same way as training data
2. **Align columns** with training features
3. **Fill missing columns** with 0 (features not present in new data)

### Step 3: Predict
```python
predicted_price = model.predict(new_house_encoded)
```

---

**Note**: This implementation provides a solid foundation for housing price prediction. For production use, consider data validation, cross-validation, and more sophisticated feature engineering.