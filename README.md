# Titanic Survivor Prediction: A Feature Engineering Deep Dive

This project is an analysis of the Titanic dataset from the Kaggle competition. The goal is to predict whether a passenger survived the Titanic based on things such as age, gender, class, and fare price

This project moves beyond a basic model to showcase an iterative and in-depth feature engineering process, demonstrating how to systematically create and test new features to improve model performance.

## Project Workflow

This project walks through the entire data science pipeline:
1.  **Data Cleaning:** Handling missing values for `Age`, `Fare`, and `Embarked`.
2.  **Iterative Feature Engineering:** Systematically creating new features by splitting existing columns (`Name`, `Cabin`, `Ticket`) and combining multiple data points into composite scores such as SurvivalScore
3.  **Model Training & Tuning:** Using an XGBoost classifier and optimizing its performance with `GridSearchCV` to find the best hyperparameters.
4.  **Prediction:** Generating the final submission file based on the best-performing model.

## Feature Engineering Highlights

The key to this project's success was extensive feature engineering. The final model uses a rich set of features created from the raw data:

* **`Title`**: Extracted from passenger names (e.g., 'Mr', 'Mrs', 'Master') to capture social status and family roles.
* **`SurvivalScore`**: A custom-built composite score that weights `Sex`, `Pclass`, `IsChild`, and relative fare price to create a single, powerful predictive signal.
* **`Deck`**: The deck letter extracted from the `Cabin` number, representing the passenger's physical location on the ship.
* **Ticket Prefixes**: Common prefixes from the `Ticket` number were one-hot encoded to capture information about booking location or ticket type.
* **`FamilySize` & `GroupSize`**: Calculated from family and ticket information to understand the size of a passenger's traveling party.
* **`IsMother`**: A binary flag to identify adult women traveling with children.

## Results

The final model, a tuned XGBoost classifier, achieved a cross-validated accuracy of **85.19%**. This result demonstrates the significant performance increase gained through a methodical feature engineering and hyperparameter tuning process.

## How to Run

This repository contains two main files:

1.  **`titanic_data_analysis.ipynb`**: A Jupyter/Colab notebook showing the full exploratory process, including visualizations and tests of different features.
2.  **`train_predict.py`**: A Final python script that runs from top to bottom

To run the final script:
1.  Clone this repository.
2.  Make sure you have the required libraries: `pandas`, `numpy`, `xgboost`, `scikit-learn`.
3.  Place the `train.csv` and `test.csv` files from Kaggle in the main directory.
4.  Run the script from your terminal:
    ```bash
    python train_predict.py
    ```
5.  This will generate a `RULREAL.csv` file ready for submission to Kaggle.


