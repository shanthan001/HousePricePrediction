# House Price Prediction

## Project Overview
This project aims to predict house prices using a dataset from the House Prices Prediction Kaggle competition. The project involves data preprocessing, exploratory data analysis, feature engineering, and building predictive models.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Setup Instructions](#setup-instructions)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Predictive Modeling](#predictive-modeling)
- [Model Evaluation](#model-evaluation)
- [Contributors](#contributors)

## Data Description
The dataset used in this project is sourced from the House Prices Prediction Kaggle competition. It includes various features related to house properties such as location, size, condition, and amenities.

**Dataset Link:** [House Prices Prediction Kaggle competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

## Exploratory Data Analysis
In this step, we explore the dataset to understand the data distribution, identify missing values, and uncover relationships between features.

1. **Automated Data Profiling:**
    We use the `ydata-profiling` tool to generate an initial data profiling report.
    ```python
    import pandas as pd
    from ydata_profiling import ProfileReport

    df = pd.read_csv('train.csv')
    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    profile.to_file("your_report.html")
    ```

2. **Business Understanding:**
    - Formulate a business problem related to house price prediction.
    - Propose data science solutions and select the most feasible one based on the dataset.

## Feature Engineering
We derive new features from the existing ones to improve the performance of our models.

1. **Descriptive Features:**
    We selected features based on property characteristics, condition, design, and amenities.

    | Feature Name     | Domain Concept     | Feature Description                          | Feature Type | Data Type |
    |------------------|--------------------|----------------------------------------------|--------------|-----------|
    | LotArea          | House Size         | Size of the lot in square feet               | Continuous   | Float     |
    | TotalBsmtSF      | House Size         | Total square feet of basement area           | Continuous   | Float     |
    | 1stFlrSF         | House Size         | First Floor square feet                      | Continuous   | Float     |
    | 2ndFlrSF         | House Size         | Second Floor square feet                     | Continuous   | Float     |
    | GrLivArea        | House Size         | Above grade (ground) living area in square feet | Continuous | Float     |
    | OverallQual      | House Condition    | Rates the overall material and finish of the house | Continuous | Integer   |
    | OverallCond      | House Condition    | Rates the overall condition of the house     | Continuous   | Integer   |
    | YearBuilt        | House Condition    | Original construction date                   | Continuous   | Integer   |
    | YearRemodAdd     | House Condition    | Remodel date                                 | Continuous   | Integer   |
    | HouseStyle       | House Design       | Style of dwelling                            | Categorical  | String    |
    | RoofStyle        | House Design       | Type of roof                                 | Categorical  | String    |
    | Exterior1st      | House Design       | Exterior covering on house                   | Categorical  | String    |
    | Exterior2nd      | House Design       | Exterior covering on house (if more than one material) | Categorical | String  |
    | MasVnrType       | House Design       | Masonry veneer type                          | Categorical  | String    |
    | PoolArea         | House Amenities    | Pool area in square feet                     | Continuous   | Float     |
    | GarageArea       | House Amenities    | Size of garage in square feet                | Continuous   | Float     |
    | OpenPorchSF      | House Amenities    | Open porch area in square feet               | Continuous   | Float     |
    | EnclosedPorch    | House Amenities    | Enclosed porch area in square feet           | Continuous   | Float     |
    | WoodDeckSF       | House Amenities    | Wood deck area in square feet                | Continuous   | Float     |

## Predictive Modeling
We build various models to predict house prices based on the engineered features.

1. **Task Formulation:**
    - Identify the type of problem (classification, regression, etc.).
    - For this project, the task is a multi-class classification problem.

2. **Feature Correlation:**
    - Split the dataset into training and test sets.
    - Identify and select the top 10 features most correlated with the target variable.

3. **Model Training and Hyperparameter Tuning:**
    - Train a Decision Tree Classifier with default hyperparameters.
    - Use RandomizedSearchCV for hyperparameter tuning.

    ```python
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    # Split the dataset
    X = df.drop('SalePriceCategory', axis=1)
    y = df['SalePriceCategory']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    # Train a Decision Tree Classifier
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Hyperparameter tuning
    param_dist = {
        'criterion': ['gini', 'entropy'],
        'max_depth': randint(1, 30),
        'min_samples_split': randint(2, 30),
        'min_samples_leaf': randint(1, 30),
    }
    random_search = RandomizedSearchCV(dtc, param_distributions=param_dist, n_iter=5, cv=10, scoring='accuracy')
    random_search.fit(X_train, y_train)
    print("Best Parameters:", random_search.best_params_)
    ```

## Model Evaluation
Evaluate the performance of the model using various metrics and visualizations.

1. **Confusion Matrix:**
    ```python
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_pred_best = random_search.best_estimator_.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Reds')
    plt.show()
    ```

2. **Classification Report:**
    ```python
    from sklearn.metrics import classification_report

    print(classification_report(y_test, y_pred_best))
    ```

3. **Learning Curve:**
    ```python
    from sklearn.model_selection import learning_curve, LearningCurveDisplay

    model = DecisionTreeClassifier(**random_search.best_params_)
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring='accuracy')
    display = LearningCurveDisplay(train_sizes=train_sizes, train_scores=train_scores, test_scores=test_scores)
    display.plot()
    plt.show()
    ```

## Contributors
- **Syanthan Vullingala** - Master of Digital Innovation - Data Science at Dalhousie University

For any queries, please contact:
- Syanthan Vullingala: vullingalasyanthan17223@gmail.com

---

