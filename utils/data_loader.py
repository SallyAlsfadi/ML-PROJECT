import pandas as pd

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






if __name__ == "__main__":
  #  print("Breast Cancer Dataset:")
   # df_bc = load_breast_cancer()
  #  print(df_bc.head(), "\n")

    #print("Mushroom Dataset:")
    #df_mushroom = load_mushroom()
    #print(df_mushroom.head())

     print("Robot Execution Failures Dataset:")
     df_robot = load_robot_failures()
     print(df_robot.head())