# Haven — House Price Prediction App

An end-to-end Machine Learning project that estimates house prices using property characteristics. Built with Python, Pandas, Scikit-learn, and Streamlit.

## Overview

Haven is an interactive house price prediction application that uses property features to generate estimated prices. It includes data exploration, preprocessing, model comparison, and a web-based prediction interface.

## Features

- House price estimation using property details
- Exploratory Data Analysis (EDA) with visualizations
- Categorical feature encoding and numerical preprocessing
- Linear Regression and Random Forest model comparison
- Five-fold cross-validation for model evaluation
- Interactive Streamlit web interface
- Downloadable property prediction report in CSV format

## Dataset

The project uses `data/Housing.csv`, containing 545 records and 13 columns.

**Target variable:** `price`

**Input features:**
- Area
- Bedrooms
- Bathrooms
- Stories
- Main road access
- Guestroom availability
- Basement availability
- Hot water heating
- Air conditioning
- Parking spaces
- Preferred area
- Furnishing status

The dataset has no missing values or duplicate rows according to the initial inspection. Please verify the dataset's license and attribution requirements before redistribution.

## Machine Learning

Two regression models are evaluated:

1. Linear Regression
2. Random Forest Regressor

### Five-fold cross-validation results

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| Mean MAE | ₹807,180 | ₹799,887 |
| Mean R² | 0.6324 | 0.6303 |

These are cross-validation results on this dataset, not a guarantee of future prediction accuracy. The differences between the models are small.

## Project Structure

```text
house-price-prediction/
├── data/
│   └── Housing.csv
├── models/
│   └── house_price_model.pkl
├── app.py
├── train.py
├── eda.py
├── inspect_data.py
├── compare_models.py
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/bhavyabapanapalli/house-price-prediction.git
cd house-price-prediction
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

### 4. Run exploratory data analysis

```bash
python inspect_data.py
python eda.py
```

### 5. Train and evaluate the models

```bash
python train.py
python compare_models.py
```

The training script saves the selected model to `models/house_price_model.pkl`.

### 6. Launch the web application

```bash
streamlit run app.py
```

Streamlit will display a local URL where you can use the application.

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Git and GitHub

## Limitations

- Model performance depends on the dataset and its representativeness.
- The dataset contains only 545 records.
- The app provides an estimate, not a formal property valuation.
- Actual property prices may differ from predictions.

## Future Improvements

- Compare additional regression models and tune hyperparameters
- Improve validation and error analysis
- Add more representative location-specific data
- Deploy the application online
- Add prediction history and richer analytics

## Author

**Bhavya Bapanapalli**

GitHub: [bhavyabapanapalli](https://github.com/bhavyabapanapalli)

---

*This project was developed for learning and demonstrating an end-to-end machine learning workflow.*
