# Import necessary libraries for data manipulation, visualization, and machine learning
import pandas as pd  # For data handling and manipulation
import seaborn as sb  # For statistical data visualization
import matplotlib.pyplot as plt  # For plotting graphs
import time as t  # For adding delays (sleep)
import sklearn.utils as u  # For shuffling data
import sklearn.preprocessing as pp  # For preprocessing, e.g., label encoding
import sklearn.tree as tr  # For Decision Tree classifier
import sklearn.ensemble as es  # For Random Forest classifier
import sklearn.metrics as m  # For model evaluation metrics
import sklearn.linear_model as lm  # For Perceptron and Logistic Regression
import sklearn.neural_network as nn  # For MLP Classifier (Neural Network)
import numpy as np  # For numerical operations
import warnings as w  # For handling warnings
w.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Function to print the main menu for visualization options
def print_menu():
    # Print a formatted menu for the user to select visualization or analysis options
    print("\n" + "="*50)
    print("Student Performance Prediction - Visualization Menu")
    print("="*50)
    print("1. Marks Class Count Graph")
    print("2. Marks Class Semester-wise Graph")
    print("3. Marks Class Gender-wise Graph")
    print("4. Marks Class Nationality-wise Graph")
    print("5. Marks Class Grade-wise Graph")
    print("6. Marks Class Section-wise Graph")
    print("7. Marks Class Topic-wise Graph")
    print("8. Marks Class Stage-wise Graph")
    print("9. Marks Class Absent Days-wise Graph")
    print("10. Continue to Prediction & Exit Visualization")
    print("11. Show All Graphs")
    print("12. Visualize Accuracy Measures")
    print("="*50)

# Function to get and validate user's menu choice
def get_choice():
    # Continuously prompt the user until a valid choice (1-12) is entered
    while True:
        try:
            ch = int(input("Enter your choice (1-12): "))
            if 1 <= ch <= 12:
                return ch
            else:
                print("Please enter a number between 1 and 12.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Function to pause execution for a short time (for user experience)
def pause(msg="Loading Graph..."):
    # Print a message and wait for 1 second
    print(msg)
    t.sleep(1)

# Function to display a specific graph based on user's choice
def show_graph(ch, data):
    # Depending on the user's choice, display the corresponding graph using seaborn
    if ch == 1:
        pause()
        print("\tMarks Class Count Graph")
        sb.countplot(x='Class', data=data, order=['L', 'M', 'H'])
        plt.show()
    elif ch == 2:
        pause()
        print("\tMarks Class Semester-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='Semester', hue='Class', data=data, hue_order=['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif ch == 3:
        pause()
        print("\tMarks Class Gender-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='gender', hue='Class', data=data, order=['M', 'F'], hue_order=['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif ch == 4:
        pause()
        print("\tMarks Class Nationality-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='NationalITy', hue='Class', data=data, hue_order=['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif ch == 5:
        pause()
        print("\tMarks Class Grade-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='GradeID', hue='Class', data=data, order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12'], hue_order = ['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif ch == 6:
        pause()
        print("\tMarks Class Section-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='SectionID', hue='Class', data=data, hue_order = ['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif ch == 7:
        pause()
        print("\tMarks Class Topic-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='Topic', hue='Class', data=data, hue_order = ['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif ch == 8:
        pause()
        print("\tMarks Class Stage-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='StageID', hue='Class', data=data, hue_order = ['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif ch == 9:
        pause()
        print("\tMarks Class Absent Days-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='StudentAbsenceDays', hue='Class', data=data, hue_order = ['L', 'M', 'H'], axes=axesarr)
        plt.show()

# Function to show all graphs sequentially
def show_all_graphs(data):
    # Loop through all graph options (1-9) and display each graph
    for i in range(1, 10):
        show_graph(i, data)

# Function to visualize the accuracy of all classifiers as a bar chart
def visualize_accuracies(accD, accR, accP, accL, accN):
    # Display a bar chart comparing the accuracy of each classifier
    import matplotlib.pyplot as plt
    classifiers = [
        "Decision Tree",
        "Random Forest",
        "Perceptron",
        "Logistic Regression",
        "MLP Classifier"
    ]
    accuracies = [
        accD,
        accR,
        accP,
        accL,
        accN
    ]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(classifiers, accuracies, color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f'])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Classifier Accuracy Comparison")
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{acc:.3f}", ha='center', va='bottom')
    plt.show()

# Main function that orchestrates the workflow
def main():
    # Load the dataset from CSV file
    data = pd.read_csv("AI-Data.csv")
    accD = accR = accP = accL = accN = None  # Initialize accuracy variables

    # Visualization menu loop
    while True:
        print_menu()  # Show menu
        ch = get_choice()  # Get user choice
        if ch == 10:
            # Exit visualization and proceed to prediction
            print("Exiting Visualization...\n")
            t.sleep(1)
            break
        elif ch == 11:
            # Show all graphs
            show_all_graphs(data)
        elif ch == 12:
            # Prepare data for accuracy visualization
            temp_data = data.copy()
            # Drop categorical and irrelevant columns for modeling
            temp_data = temp_data.drop("gender", axis=1)
            temp_data = temp_data.drop("StageID", axis=1)
            temp_data = temp_data.drop("GradeID", axis=1)
            temp_data = temp_data.drop("NationalITy", axis=1)
            temp_data = temp_data.drop("PlaceofBirth", axis=1)
            temp_data = temp_data.drop("SectionID", axis=1)
            temp_data = temp_data.drop("Topic", axis=1)
            temp_data = temp_data.drop("Semester", axis=1)
            temp_data = temp_data.drop("Relation", axis=1)
            temp_data = temp_data.drop("ParentschoolSatisfaction", axis=1)
            temp_data = temp_data.drop("ParentAnsweringSurvey", axis=1)
            temp_data = temp_data.drop("AnnouncementsView", axis=1)
            u.shuffle(temp_data)  # Shuffle data to avoid bias
            # Map GradeID to integers if present
            gradeID_dict = {"G-01" : 1,
                            "G-02" : 2,
                            "G-03" : 3,
                            "G-04" : 4,
                            "G-05" : 5,
                            "G-06" : 6,
                            "G-07" : 7,
                            "G-08" : 8,
                            "G-09" : 9,
                            "G-10" : 10,
                            "G-11" : 11,
                            "G-12" : 12}
            temp_data = temp_data.replace({"GradeID" : gradeID_dict})
            # Encode categorical columns using LabelEncoder
            for column in temp_data.columns:
                if temp_data[column].dtype == type(object):
                    le = pp.LabelEncoder()
                    temp_data[column] = le.fit_transform(temp_data[column])
            # Split data into train and test sets (70% train, 30% test)
            ind = int(len(temp_data) * 0.70)
            feats = temp_data.values[:, 0:4]  # Features: first 4 columns
            lbls = temp_data.values[:,4]      # Labels: 5th column
            feats_Train = feats[0:ind]
            feats_Test = feats[(ind+1):len(feats)]
            lbls_Train = lbls[0:ind]
            lbls_Test = lbls[(ind+1):len(lbls)]
            # Train and evaluate Decision Tree
            modelD = tr.DecisionTreeClassifier()
            modelD.fit(feats_Train, lbls_Train)
            lbls_predD = modelD.predict(feats_Test)
            accD = sum(a==b for a,b in zip(lbls_Test, lbls_predD)) / len(lbls_Test)
            # Train and evaluate Random Forest
            modelR = es.RandomForestClassifier()
            modelR.fit(feats_Train, lbls_Train)
            lbls_predR = modelR.predict(feats_Test)
            accR = sum(a==b for a,b in zip(lbls_Test, lbls_predR)) / len(lbls_Test)
            # Train and evaluate Perceptron
            modelP = lm.Perceptron()
            modelP.fit(feats_Train, lbls_Train)
            lbls_predP = modelP.predict(feats_Test)
            accP = sum(a==b for a,b in zip(lbls_Test, lbls_predP)) / len(lbls_Test)
            # Train and evaluate Logistic Regression
            modelL = lm.LogisticRegression()
            modelL.fit(feats_Train, lbls_Train)
            lbls_predL = modelL.predict(feats_Test)
            accL = sum(a==b for a,b in zip(lbls_Test, lbls_predL)) / len(lbls_Test)
            # Train and evaluate MLP Classifier
            modelN = nn.MLPClassifier(activation="logistic")
            modelN.fit(feats_Train, lbls_Train)
            lbls_predN = modelN.predict(feats_Test)
            accN = sum(a==b for a,b in zip(lbls_Test, lbls_predN)) / len(lbls_Test)
            # Visualize all classifier accuracies
            visualize_accuracies(accD, accR, accP, accL, accN)
        else:
            # Show the selected graph
            show_graph(ch, data)

    # --- Data preprocessing for model training and prediction ---
    # Drop categorical and irrelevant columns for modeling
    data = data.drop("gender", axis=1)
    data = data.drop("StageID", axis=1)
    data = data.drop("GradeID", axis=1)
    data = data.drop("NationalITy", axis=1)
    data = data.drop("PlaceofBirth", axis=1)
    data = data.drop("SectionID", axis=1)
    data = data.drop("Topic", axis=1)
    data = data.drop("Semester", axis=1)
    data = data.drop("Relation", axis=1)
    data = data.drop("ParentschoolSatisfaction", axis=1)
    data = data.drop("ParentAnsweringSurvey", axis=1)
    #data = data.drop("VisITedResources", axis=1)
    data = data.drop("AnnouncementsView", axis=1)
    u.shuffle(data)  # Shuffle data

    # Initialize counters for accuracy calculation for each classifier
    countD = 0
    countP = 0
    countL = 0
    countR = 0
    countN = 0

    # Map GradeID to integers if present (not needed here, but kept for consistency)
    gradeID_dict = {"G-01" : 1,
                    "G-02" : 2,
                    "G-03" : 3,
                    "G-04" : 4,
                    "G-05" : 5,
                    "G-06" : 6,
                    "G-07" : 7,
                    "G-08" : 8,
                    "G-09" : 9,
                    "G-10" : 10,
                    "G-11" : 11,
                    "G-12" : 12}
    data = data.replace({"GradeID" : gradeID_dict})

    # Encode categorical columns using LabelEncoder
    for column in data.columns:
        if data[column].dtype == type(object):
            le = pp.LabelEncoder()
            data[column] = le.fit_transform(data[column])

    # Split data into train and test sets (70% train, 30% test)
    ind = int(len(data) * 0.70)
    feats = data.values[:, 0:4]  # Features: first 4 columns
    lbls = data.values[:,4]      # Labels: 5th column
    feats_Train = feats[0:ind]
    feats_Test = feats[(ind+1):len(feats)]
    lbls_Train = lbls[0:ind]
    lbls_Test = lbls[(ind+1):len(lbls)]

    # Train and evaluate Decision Tree
    modelD = tr.DecisionTreeClassifier()
    modelD.fit(feats_Train, lbls_Train)
    lbls_predD = modelD.predict(feats_Test)
    for a,b in zip(lbls_Test, lbls_predD):
        if(a==b):
            countD += 1
    accD = (countD/len(lbls_Test))
    print("\nAccuracy measures using Decision Tree:")
    print(m.classification_report(lbls_Test, lbls_predD),"\n")
    print("\nAccuracy using Decision Tree: ", str(round(accD, 3)))
    t.sleep(1)

    # Train and evaluate Random Forest
    modelR = es.RandomForestClassifier()
    modelR.fit(feats_Train, lbls_Train)
    lbls_predR = modelR.predict(feats_Test)
    for a,b in zip(lbls_Test, lbls_predR):
        if(a==b):
            countR += 1
    print("\nAccuracy Measures for Random Forest Classifier: \n")
    #print("\nConfusion Matrix: \n", m.confusion_matrix(lbls_Test, lbls_predR))
    print("\n", m.classification_report(lbls_Test,lbls_predR))
    accR = countR/len(lbls_Test)
    print("\nAccuracy using Random Forest: ", str(round(accR, 3)))
    t.sleep(1)

    # Train and evaluate Perceptron
    modelP = lm.Perceptron()
    modelP.fit(feats_Train, lbls_Train)
    lbls_predP = modelP.predict(feats_Test)
    for a,b in zip(lbls_Test, lbls_predP):
        if a == b:
            countP += 1
    accP = countP/len(lbls_Test)
    print("\nAccuracy measures using Linear Model Perceptron:")
    print(m.classification_report(lbls_Test, lbls_predP),"\n") 
    print("\nAccuracy using Linear Model Perceptron: ", str(round(accP, 3)), "\n")
    t.sleep(1)

    # Train and evaluate Logistic Regression
    modelL = lm.LogisticRegression()
    modelL.fit(feats_Train, lbls_Train)
    lbls_predL = modelL.predict(feats_Test)
    for a,b in zip(lbls_Test, lbls_predL):
        if a == b:
            countL += 1
    accL = countL/len(lbls_Test)
    print("\nAccuracy measures using Linear Model Logistic Regression:")
    print(m.classification_report(lbls_Test, lbls_predL),"\n")
    print("\nAccuracy using Linear Model Logistic Regression: ", str(round(accP, 3)), "\n")
    t.sleep(1)

    # Train and evaluate MLP Classifier (Neural Network)
    modelN = nn.MLPClassifier(activation="logistic")
    modelN.fit(feats_Train, lbls_Train)
    lbls_predN = modelN.predict(feats_Test)
    for a,b in zip(lbls_Test, lbls_predN):
        if a==b:
            countN += 1
    print("\nAccuracy measures using MLP Classifier:")
    print(m.classification_report(lbls_Test, lbls_predN),"\n")
    accN = countN/len(lbls_Test)
    print("\nAccuracy using Neural Network MLP Classifier: ", str(round(accN, 3)), "\n")

    # Prompt user for custom input to predict a new student's performance
    choice = input("Do you want to test specific input (y or n): ")
    if(choice.lower()=="y"):
        # Collect user input for all relevant features
        gen = input("Enter Gender (M or F): ")
        if (gen.upper() == "M"):
           gen = 1
        elif (gen.upper() == "F"):
           gen = 0
        nat = input("Enter Nationality: ")
        pob = input("Place of Birth: ")
        gra = input("Grade ID as (G-<grade>): ")
        if(gra == "G-02"):
            gra = 2
        elif (gra == "G-04"):
            gra = 4
        elif (gra == "G-05"):
            gra = 5
        elif (gra == "G-06"):
            gra = 6
        elif (gra == "G-07"):
            gra = 7
        elif (gra == "G-08"):
            gra = 8
        elif (gra == "G-09"):
            gra = 9
        elif (gra == "G-10"):
            gra = 10
        elif (gra == "G-11"):
            gra = 11
        elif (gra == "G-12"):
            gra = 12
        sec = input("Enter Section: ")
        top = input("Enter Topic: ")
        sem = input("Enter Semester (F or S): ")
        if (sem.upper() == "F"):
           sem = 0
        elif (sem.upper() == "S"):
           sem = 1
        rel = input("Enter Relation (Father or Mum): ")
        if (rel == "Father"):
           rel = 0
        elif (rel == "Mum"):
           rel = 1
        rai = int(input("Enter raised hands: "))
        res = int(input("Enter Visited Resources: "))
        ann = int(input("Enter announcements viewed: "))
        dis = int(input("Enter no. of Discussions: "))
        sur = input("Enter Parent Answered Survey (Y or N): ")
        if (sur.upper() == "Y"):
           sur = 1
        elif (sur.upper() == "N"):
           sur = 0
        sat = input("Enter Parent School Satisfaction (Good or Bad): ")
        if (sat == "Good"):
           sat = 1
        elif (sat == "Bad"):
           sat = 0
        absc = input("Enter No. of Abscenes(Under-7 or Above-7): ")
        if (absc == "Under-7"):
           absc = 1
        elif (absc == "Above-7"):
           absc = 0
        # Prepare input array for prediction (only selected features)
        arr = np.array([rai, res, dis, absc])
        # Predict using all trained models
        predD = modelD.predict(arr.reshape(1, -1))
        predR = modelR.predict(arr.reshape(1, -1))
        predP = modelP.predict(arr.reshape(1, -1))
        predL = modelL.predict(arr.reshape(1, -1))
        predN = modelN.predict(arr.reshape(1, -1))
        # Convert numeric predictions back to class labels
        if (predD == 0):
            predD = "H"
        elif (predD == 1):
            predD = "M"
        elif (predD == 2):
            predD = "L"
        if (predR == 0):
            predR = "H"
        elif (predR == 1):
            predR = "M"
        elif (predR == 2):
            predR = "L"
        if (predP == 0):
            predP = "H"
        elif (predP == 1):
            predP = "M"
        elif (predP == 2):
            predP = "L"
        if (predL == 0):
            predL = "H"
        elif (predL == 1):
            predL = "M"
        elif (predL == 2):
            predL = "L"
        if (predN == 0):
            predN = "H"
        elif (predN == 1):
            predN = "M"
        elif (predN == 2):
            predN = "L"
        t.sleep(1)
        # Display predictions from all models
        print("\nUsing Decision Tree Classifier: ", predD)
        t.sleep(1)
        print("Using Random Forest Classifier: ", predR)
        t.sleep(1)
        print("Using Linear Model Perceptron: ", predP)
        t.sleep(1)
        print("Using Linear Model Logisitic Regression: ", predL)
        t.sleep(1)
        print("Using Neural Network MLP Classifier: ", predN)
        print("\nExiting...")
        t.sleep(1)
    else:
        # If user does not want to test input, exit
        print("Exiting..")
        t.sleep(1)

# Entry point for the script
if __name__ == "__main__":
    # Run the main function if this script is executed directly
    main()