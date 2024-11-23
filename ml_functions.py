import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Z-Score Normalization."""
    return (df - df.mean()) / df.std()

def split_features_and_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into features and target."""
    y = df[target_col]
    X = df.drop(columns=target_col)
    return X, y

def pre_pro_kidney(dataset: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess the Chronic Kidney Disease dataset."""
    ckd_path = dataset
    nodataval = "?"
    # Read
    data, meta = arff.loadarff(ckd_path)
    ckd = pd.DataFrame(data)

    # Decode strings
    is_str_cols = ckd.dtypes == object
    str_columns = ckd.columns[is_str_cols]
    ckd[str_columns] = ckd[str_columns].apply(lambda s: s.str.decode("utf-8"))

    # Handle nodata values
    ckd = ckd.replace(nodataval, np.nan)

    # Convert remaining false string columns
    other_numeric_columns = ["sg", "al", "su"]
    ckd[other_numeric_columns] = ckd[other_numeric_columns].apply(pd.to_numeric)

    # Use categorical data type
    categoric_columns = pd.Index(set(str_columns) - set(other_numeric_columns))
    ckd[categoric_columns] = ckd[categoric_columns].astype("category")

    #* Remove the "ground-truth" column and store its values aside

    ckd, y = split_features_and_target(ckd, "class")

    # Check the number of missing values
    ckd.isna().sum(axis=0)

    fillna_mean_cols = pd.Index(
    set(ckd.columns[ckd.dtypes == "float64"]) - set(other_numeric_columns)
    )
    fillna_most_cols = pd.Index(
    set(ckd.columns[ckd.dtypes == "category"]) | set(other_numeric_columns)
    )
    assert set(fillna_mean_cols.union(fillna_most_cols)) == set(ckd.columns)
    
    ckd[fillna_mean_cols] = ckd[fillna_mean_cols].fillna(ckd[fillna_mean_cols].mean())
    ckd[fillna_most_cols] = ckd[fillna_most_cols].fillna(
    ckd[fillna_most_cols].mode().iloc[0])

    ckd = pd.get_dummies(ckd, drop_first=True)
    
    # Data Normalization
    ckd = (ckd - ckd.mean()) / (ckd.std())

    #Convert the class value y to 1 for chronic kidney disease and 0 for no chronic kidney disease 
    trans_table = {"notckd": 0, "ckd": 1}  # Map 'notckd' to 0, 'ckd' to 1
    y = y.map(lambda x: trans_table[x])

    return ckd,y
def pre_pro_banknote(dataset: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess the Banknote Authentication dataset."""
    data=[]
    with open('data_banknote_authentication.txt', "r") as doc: 
        for line in doc:
            line=line.replace('\n','')
            data.append(line.split(','))
    names=['variance', 'skewness', 'curtosis', 'entropy', 'class']
    data = pd.DataFrame(data[:], columns=names)
    # Convert data type 
    Numerical_columns=['variance', 'skewness', 'curtosis', 'entropy', 'class']
    
    # Convert each column in Numerical_columns to numeric
    data[Numerical_columns]=data[Numerical_columns].apply(pd.to_numeric)
    
    #* Remove the "ground-truth" column and store its values aside
    data, y = split_features_and_target(data, "class")
    # Normalize 
    data = normalize_data(data)
    
    
    return data, y

def preprocessing(dataset: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Main preprocessing function for both datasets."""
    if "chronic_kidney_disease" in dataset:
        return pre_pro_kidney(dataset)
    elif "data_banknote_authentication" in dataset:
        return pre_pro_banknote(dataset)
    else:
        raise ValueError("Invalid dataset provided!")
def preparation(data, y,nb_splits=5):
    """
    Prepare a dataset for training by splitting it between a training set and a test set, 
    and then splitting the training set for cross-validation.
    
    Parameters:
    - data (pd.DataFrame): A processed DataFrame containing the preprocessed features.
    - y (pd.Series): The target column "class".
    - nb_splits (int) : The number of splits of kfolds
                     
    Returns:
    - X_train (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Test features.
    - y_train (pd.Series): Training target.
    - y_test (pd.Series): Test target.
    - cross_validation_splits (list of tuples): A list of tuples, where each tuple contains 
      (X_train_cv, X_val_cv, y_train_cv, y_val_cv) for cross-validation splits.
    """
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)
    
    # Set up KFold cross-validation
    kf = KFold(nb_splits, shuffle=True, random_state=42)
    
    # Generate cross-validation splits from the training set
    cross_validation_splits = []
    for train_index, val_index in kf.split(X_train):
        X_train_cv = X_train.iloc[train_index]
        X_val_cv = X_train.iloc[val_index]
        y_train_cv = y_train.iloc[train_index]
        y_val_cv = y_train.iloc[val_index]
        cross_validation_splits.append((X_train_cv, X_val_cv, y_train_cv, y_val_cv))
    
    return X_train, X_test, y_train, y_test, cross_validation_splits

def training(X_train, y_train, cross_validation_splits, list_models):
    """
    Trains multiple models using cross-validation and selects the best model for each algorithm.
    
    Returns:
    - best_models: dict, contains the best-performing model for each algorithm.
    """
    models = {
        "Logistic Regression": list_models[0],
        "Support Vector Machine":list_models[1],
        "Decision Tree": list_models[2],
        "Random Forest": list_models[3],
        "K-Nearest Neighbors": list_models[4]
    }
    best_models={}

    # Iterate over models
    for model_name, model in models.items():
        trained_models=[]
        model_accuracy = []

        # Iterate over cross-validation splits
        for X_train_cv, X_val_cv, y_train_cv, y_val_cv in cross_validation_splits:
            # Fit the model
            model_tr=model.fit(X_train_cv, y_train_cv)

            # Predict on the validation set
            y_pred = model_tr.predict(X_val_cv)
            accuracy = accuracy_score(y_val_cv, y_pred)
            model_accuracy.append(accuracy)  # Store predictions
            trained_models.append(model_tr)
        best_model=trained_models[np.argmax(model_accuracy)]
        # Store predictions for the model
        best_models[model_name] = best_model

    return best_models

def results(best_models,X_test,y_test):
    
    print("Models Performance Summary:")
    
    # Initialize a list to collect metric results
    results = []
    
    # Iterate over models and evaluate them
    for model_name, best_model in best_models.items():
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Append results to the list
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    # Convert results to a pandas DataFrame
    df_results = pd.DataFrame(results)
    
    return df_results
# Display the resulats in the format of a dataframe
from sklearn.ensemble import RandomForestClassifier

def select_rf_features_by_threshold(data, y, threshold=0.001, random_state=42):
    """
    Function to train a Random Forest model, extract feature importance, and select features with importance above a specified threshold.

    ### Parameters:
    - data: *DataFrame*  
    The input data containing the features.
    - y: array  
    The target labels.
    - threshold: float  
    The importance threshold for selecting features.
    
    ### Returns:
    - selected_features: *pd.Series*  
    The selected features based on the threshold.
    """
    # Entraîner le modèle Random Forest
    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(data, y)
    
    # Obtenir les importances des caractéristiques
    importances = rf.feature_importances_
    
    # Créer un DataFrame pour afficher l'importance des caractéristiques
    features_importance = pd.DataFrame({'Feature': data.columns, 'Importance': importances})
    
    # Trier les caractéristiques par importance décroissante
    features_importance = features_importance.sort_values(by='Importance', ascending=False)
    
    # Afficher les importances des caractéristiques
    print(features_importance)
    
    # Sélectionner les caractéristiques avec une importance supérieure au seuil
    selected_features = features_importance.loc[features_importance['Importance'] > threshold, 'Feature']
    
    print(f"Features selected with importance above threshold ({threshold}):")
    print(selected_features)
    
    return selected_features



def remove_highly_correlated_features(data, threshold=0.7):
    """
    Function to remove highly correlated features from a DataFrame.
    
    Parameters:
    - data : DataFrame, the input data (features).
    - threshold : float, the correlation threshold to decide which features to drop (default is 0.7).
    
    Returns:
    - df_reduced : DataFrame, the data with highly correlated features removed.
    """
    # Calculate the correlation matrix
    corr_matrix = data.corr().abs()
    
    # Get the upper triangle of the correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify features to drop
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    print("Features to drop:")
    print(to_drop)
    
    # Drop only one feature from each pair
    df_reduced = data.drop(columns=to_drop)
    
    # Show the reduced DataFrame
    print("Reduced DataFrame:")
    print(df_reduced.head())
    
    return df_reduced

models=[LogisticRegression(),svm.SVC(),DecisionTreeClassifier(),RandomForestClassifier(), KNeighborsClassifier()]
