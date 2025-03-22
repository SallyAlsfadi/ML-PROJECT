import pandas as pd
import os
import numpy as np
def load_breast_cancer():
   
    columns = [
        "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
        "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se",
        "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
        "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
        "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
        "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
    
 
    df = pd.read_csv("data/breast_cancer.csv", header=None, names=columns)
    
    
    df.drop(columns=["id"], inplace=True)
    

    diagnosis_col = df.pop("diagnosis")
   
    df["diagnosis"] = diagnosis_col.map({"M": 1, "B": 0})

    return df

def load_mushroom():
   
    columns = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
        "gill-attachment", "gill-spacing", "gill-size", "gill-color",
        "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
        "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
        "ring-number", "ring-type", "spore-print-color", "population", "habitat"
    ]
    
  
    df = pd.read_csv("data/mushroom.csv", header=None, names=columns)
    
    # Convert target variable ('class') to numeric:
    # 'e' (edible) → 0, 'p' (poisonous) → 1
    df["class"] = df["class"].map({"e": 0, "p": 1})
    

    class_col = df.pop("class")
    df["class"] = class_col

    return df




def load_robot_failures():
 
    with open("data/robot_lp1.csv", "r") as file:
        lines = file.readlines()


    clean_lines = [line for line in lines if not line.strip().startswith("normal")]


    with open("data/robot_lp1_clean.csv", "w") as file:
        file.writelines(clean_lines)

    sample_line = clean_lines[0].strip().split()
    num_features = len(sample_line) - 1 

    
    feature_names = [f"feature_{i+1}" for i in range(num_features)]
    column_names = feature_names + ["failure_type"]

   
    df = pd.read_csv("data/robot_lp1_clean.csv", sep=r"\s+", header=None, names=column_names, engine="python")
    df.dropna(subset=["failure_type"], inplace=True)

   
    df["failure_type"] = df["failure_type"].astype(int)
 
    print("Unique failure types:", df["failure_type"].unique())

    return df

    





def min_max_scaling(column):
    """Manually apply Min-Max scaling."""
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val) if max_val != min_val else column

def load_heart_failure_data(filepath):
    """
    Loads and preprocesses the Heart Failure Prediction dataset 
    without using ML libraries.
    
    Args:
    filepath (str): Path to the dataset file.

    Returns:
    pd.DataFrame: Preprocessed dataset ready for analysis.
    """
    # Load dataset
    df = pd.read_csv(filepath)

    # Handling missing values (if any)
    df.dropna(inplace=True)
    
    # Convert categorical columns to numeric
    categorical_columns = ["sex", "smoking", "diabetes", "anaemia", "high_blood_pressure", "DEATH_EVENT"]
    for col in categorical_columns:
        df[col] = df[col].astype(int)

    # Manually normalize numerical columns using Min-Max scaling
    numerical_columns = ["age", "creatinine_phosphokinase", "ejection_fraction", 
                         "platelets", "serum_creatinine", "serum_sodium", "time"]
    
    for col in numerical_columns:
        df[col] = min_max_scaling(df[col])

    print("✅ Heart Failure Prediction dataset loaded and preprocessed successfully!")
    print("🔹 First 5 rows of the dataset:\n", df.head())
    return df

def load_wine_quality(filepath: str) -> pd.DataFrame:
    """
    Loads and preprocesses the Wine Quality dataset.

    Args:
    filepath (str): Path to the wine quality dataset CSV file.

    Returns:
    pd.DataFrame: Preprocessed Wine Quality dataset.
    """
    df = pd.read_csv(filepath, sep=";")  # Wine dataset uses ';' as delimiter

    # Apply Min-Max scaling to all features except the target
    numerical_columns = df.columns[:-1]  # All columns except 'quality'
    for col in numerical_columns:
        df[col] = min_max_scaling(df[col])

    # Move target column to the end (optional, just for consistency)
    quality_col = df.pop("quality")
    df["quality"] = quality_col




    print("✅ Wine Quality dataset loaded and preprocessed successfully!")
    print("🔹 First 5 rows of the dataset:\n", df.head())
    
    return df




def train_test_split_features_target(df, target_column, test_size=0.2, random_seed=None):
    """
    Splits the dataset into training and testing sets, separating features and target.
    
    Parameters:
        df (pd.DataFrame): The full dataset.
        target_column (str): Name of the column to be used as the target.
        test_size (float): Proportion of test data (0.2 = 20%).
        random_seed (int, optional): Seed for reproducibility.
    
    Returns:
        X_train, y_train, X_test, y_test: Split feature and target sets.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

  
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    test_len = int(len(shuffled_df) * test_size)

    test_df = shuffled_df.iloc[:test_len]
    train_df = shuffled_df.iloc[test_len:]

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return X_train, y_train, X_test, y_test







if __name__ == "__main__":
  #  print("Breast Cancer Dataset:")
   # df_bc = load_breast_cancer()
  #  print(df_bc.head(), "\n")

    #print("Mushroom Dataset:")
    #df_mushroom = load_mushroom()
    #print(df_mushroom.head())

     #print("Robot Execution Failures Dataset:")
     #df_robot = load_robot_failures()
     #print(df_robot.head())
     #df_heart = load_heart_failure_data("data/heart_failure_clinical_records_dataset.csv")
     #wine_quality_filepath = "data/winequality-white.csv"  # Adjust filename if needed
     #wine_df = load_wine_quality(wine_quality_filepath)
    # Load the dataset
    df = load_breast_cancer()

# Split data
    X_train, y_train, X_test, y_test = train_test_split_features_target(df, target_column="diagnosis", test_size=0.2)

# Ready to train your own ML algorithms on X_train, y_train
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    print(X_train.head())
