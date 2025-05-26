import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import time as t
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.metrics as m
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
import warnings as w
w.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv("c:/Users/PC/Downloads/StudentPerformancePrediction-ML-main/bd_students_per_v2.csv")

# Show columns and a sample
print("Columns in bd_students_per_v2.csv:")
print(data.columns.tolist())
print("\nFirst few rows:")
print(data.head())

# --- Visualization Menu ---
target_col = "stu_group"
feature_cols = [c for c in data.columns if c not in ["id", "full_name", target_col]]
while True:
    print("\nChoose a graph to display:")
    for idx, col in enumerate(feature_cols):
        print(f"{idx+1}. {target_col} vs {col} Graph")
    print(f"{len(feature_cols)+1}. No Graph\n")
    ch = int(input("Enter Choice: "))
    if 1 <= ch <= len(feature_cols):
        col = feature_cols[ch-1]
        print(f"Loading Graph for {target_col} vs {col}...\n")
        t.sleep(1)
        plt.figure(figsize=(10, 6))
        if data[col].dtype == object or len(data[col].unique()) < 20:
            sb.countplot(x=col, hue=target_col, data=data)
        else:
            sb.boxplot(x=target_col, y=col, data=data)
        plt.title(f"{target_col} vs {col}")
        plt.show()
    elif ch == len(feature_cols)+1:
        print("Exiting graph menu...\n")
        t.sleep(1)
        break

# --- Preprocessing ---
drop_cols = ["id", "full_name"]
data = data.drop(drop_cols, axis=1)

# Encode categorical columns
# Store label encoders for each categorical column
encoders = {}

for column in data.columns:
    if data[column].dtype == object:
        le = pp.LabelEncoder()
        le.fit(data[column])
        encoders[column] = le
        data[column] = le.transform(data[column])

# Features/labels split
feats = data.drop(target_col, axis=1).values
lbls = data[target_col].values
ind = int(len(data) * 0.70)
feats_Train = feats[0:ind]
feats_Test = feats[(ind+1):len(feats)]
lbls_Train = lbls[0:ind]
lbls_Test = lbls[(ind+1):len(lbls)]

# --- Classification Models ---
models = {
    "Decision Tree": tr.DecisionTreeClassifier(),
    "Random Forest": es.RandomForestClassifier(),
    "Perceptron": lm.Perceptron(),
    "Logistic Regression": lm.LogisticRegression(),
    "MLP Classifier": nn.MLPClassifier(activation="logistic")
}
for name, model in models.items():
    model.fit(feats_Train, lbls_Train)
    lbls_pred = model.predict(feats_Test)
    acc = np.mean(lbls_pred == lbls_Test)
    print(f"\nAccuracy measures using {name}:")
    print(m.classification_report(lbls_Test, lbls_pred))
    print(f"Accuracy using {name}: {round(acc, 3)}\n")
    t.sleep(1)

# --- Test Specific Input ---
choice = input("Do you want to test specific input (y or n): ")
if(choice.lower() == "y"):
    print("Enter values for the following features:")
    input_vals = []
    for col in data.columns:
        if col == target_col:
            continue
        # Show options for categorical columns
        if col in encoders:
            print(f"Options for {col}: {list(encoders[col].classes_)}")
        while True:
            val = input(f"{col}: ").strip().replace('"', '').replace("'", "")
            try:
                val = float(val)
                break
            except:
                if col in encoders:
                    if val not in encoders[col].classes_:
                        print(f"Invalid input '{val}' for {col}. Please enter one of: {list(encoders[col].classes_)}")
                        continue
                    val = encoders[col].transform([val])[0]
                    break
                else:
                    print(f"Unexpected categorical value for {col}.")
                    continue
        input_vals.append(val)
    arr = np.array(input_vals)
    for name, model in models.items():
        pred = model.predict(arr.reshape(1, -1))
        print(f"Prediction using {name}: {pred[0]}")
    print("\nExiting...")
    t.sleep(1)
else:
    print("Exiting..")
    t.sleep(1)
