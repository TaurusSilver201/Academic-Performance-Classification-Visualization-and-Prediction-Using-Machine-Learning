This Python script is a comprehensive tool for analyzing student performance and predicting outcomes using multiple machine learning models.

Summary of its workflow:

Data Handling & Visualization:
The script loads student performance data from a CSV file and provides an interactive menu for visualizing different aspects of the dataset, including class distribution, semester-wise performance, gender-wise performance, nationality, grades, sections, topics, stages, and absenteeism. Seaborn and Matplotlib are used for plotting, with optional sequential visualization of all graphs.

Preprocessing:
Categorical features (e.g., gender, nationality, grade, section, topic, parent responses) are encoded numerically using LabelEncoder. Irrelevant or redundant columns are removed, and the dataset is shuffled to avoid bias.

Model Training & Evaluation:
The script splits the dataset into training (70%) and testing (30%) sets and trains five machine learning models:

*Decision Tree Classifier

*Random Forest Classifier

*Perceptron (Linear Model)

*Logistic Regression

*MLP Classifier (Neural Network)

Each model is evaluated on the test set, and detailed classification reports and accuracy metrics are displayed. It also provides a bar chart comparing the accuracies of all classifiers.

Interactive Prediction:
Users can input custom data for a new student (e.g., gender, nationality, grade, section, raised hands, resources visited, absences, parent responses). The script predicts the student’s performance class (Low, Medium, High) using all trained models and displays the results.

User-Friendly Design:
The script features a menu-driven interface, short delays for readability, and step-by-step guidance for both visualization and prediction tasks.

