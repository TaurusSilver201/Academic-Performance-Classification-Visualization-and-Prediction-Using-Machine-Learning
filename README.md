# Student Performance Prediction System

A comprehensive machine learning system for predicting student academic performance using multiple classification algorithms with interactive data visualization capabilities.

## Overview

This system analyzes student data to predict academic performance levels (High, Medium, Low) using various machine learning classifiers. It provides an interactive menu for data visualization and comprehensive model comparison with accuracy metrics.

## Features

- **Multiple ML Algorithms**: Decision Tree, Random Forest, Perceptron, Logistic Regression, and MLP Classifier
- **Interactive Visualizations**: 11 different graph types for data exploration
- **Model Comparison**: Side-by-side accuracy comparison of all classifiers
- **Custom Predictions**: Interactive input system for predicting individual student performance
- **Comprehensive Metrics**: Classification reports with precision, recall, and F1-scores
- **Data Preprocessing**: Automated categorical encoding and feature selection

## Dataset Requirements

The system expects a CSV file named `AI-Data.csv` with the following columns:

### Student Information
- `gender`: Student gender (M/F)
- `NationalITy`: Student nationality
- `PlaceofBirth`: Student's place of birth
- `StageID`: Educational stage
- `GradeID`: Grade level (G-02 to G-12)
- `SectionID`: Class section
- `Topic`: Subject topic
- `Semester`: Semester (F/S)

### Academic Engagement
- `raisedhands`: Number of times student raised hand
- `VisITedResources`: Number of resources visited
- `AnnouncementsView`: Number of announcements viewed
- `Discussion`: Number of discussion participations

### Family Information
- `Relation`: Parent relationship (Father/Mum)
- `ParentAnsweringSurvey`: Parent survey participation (Y/N)
- `ParentschoolSatisfaction`: Parent satisfaction level (Good/Bad)

### Attendance
- `StudentAbsenceDays`: Absence pattern (Under-7/Above-7)

### Target Variable
- `Class`: Performance level (H=High, M=Medium, L=Low)

## Installation

### Prerequisites
- Python 3.7+
- Required libraries (install via pip):

```bash
pip install pandas seaborn matplotlib scikit-learn numpy
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction
```

2. Ensure your dataset `AI-Data.csv` is in the project directory

3. Run the application:
```bash
python student_performance_predictor.py
```

## Usage

### Interactive Menu System

The application provides a comprehensive menu with the following options:

```
1. Marks Class Count Graph
2. Marks Class Semester-wise Graph
3. Marks Class Gender-wise Graph
4. Marks Class Nationality-wise Graph
5. Marks Class Grade-wise Graph
6. Marks Class Section-wise Graph
7. Marks Class Topic-wise Graph
8. Marks Class Stage-wise Graph
9. Marks Class Absent Days-wise Graph
10. Continue to Prediction & Exit Visualization
11. Show All Graphs
12. Visualize Accuracy Measures
```

### Visualization Options

#### Individual Graphs (Options 1-9)
Each option displays specific categorical analysis:
- **Count plots** showing distribution of performance classes
- **Hue-based grouping** for comparative analysis
- **Ordered visualization** for better interpretation

#### All Graphs (Option 11)
Displays all visualization graphs sequentially for comprehensive data exploration.

#### Accuracy Comparison (Option 12)
- Trains all 5 machine learning models
- Displays comparative bar chart of accuracies
- Shows numerical accuracy values for each classifier

### Machine Learning Models

The system implements and compares five different algorithms:

1. **Decision Tree Classifier**
   - Simple, interpretable tree-based model
   - Good for understanding feature importance

2. **Random Forest Classifier**
   - Ensemble method using multiple decision trees
   - Reduces overfitting and improves accuracy

3. **Perceptron**
   - Linear classification algorithm
   - Fast and efficient for linearly separable data

4. **Logistic Regression**
   - Probabilistic linear classifier
   - Provides probability estimates for predictions

5. **MLP Classifier (Neural Network)**
   - Multi-layer perceptron with logistic activation
   - Capable of learning complex non-linear patterns

### Custom Prediction

After model training, you can input custom student data for prediction:

```
Do you want to test specific input (y or n): y
Enter Gender (M or F): F
Enter Nationality: KW
Place of Birth: KuwaIT
Grade ID as (G-<grade>): G-07
Enter Section: A
Enter Topic: Math
Enter Semester (F or S): F
Enter Relation (Father or Mum): Father
Enter raised hands: 85
Enter Visited Resources: 75
Enter announcements viewed: 15
Enter no. of Discussions: 25
Enter Parent Answered Survey (Y or N): Y
Enter Parent School Satisfaction (Good or Bad): Good
Enter No. of Abscenes(Under-7 or Above-7): Under-7
```

The system will then provide predictions from all five models.

## Data Preprocessing Pipeline

### Feature Selection
The system automatically removes categorical features that don't contribute directly to the core prediction:
- Drops demographic and contextual columns
- Retains numerical engagement metrics
- Uses only 4 key features: `raisedhands`, `VisITedResources`, `Discussion`, `StudentAbsenceDays`

### Encoding Process
- **Label Encoding**: Converts categorical variables to numerical format
- **Data Shuffling**: Randomizes data order to prevent bias
- **Train-Test Split**: 70% training, 30% testing

### Grade Mapping
Converts grade IDs to numerical values:
```python
G-01 → 1, G-02 → 2, ..., G-12 → 12
```

## Model Evaluation

### Performance Metrics
For each classifier, the system provides:
- **Accuracy Score**: Overall prediction accuracy
- **Classification Report**: Precision, recall, F1-score for each class
- **Detailed Breakdown**: Per-class performance analysis

### Output Format
```
Accuracy using Decision Tree: 0.XXX
Accuracy using Random Forest: 0.XXX
Accuracy using Linear Model Perceptron: 0.XXX
Accuracy using Linear Model Logistic Regression: 0.XXX
Accuracy using Neural Network MLP Classifier: 0.XXX
```

## File Structure

```
student-performance-prediction/
├── student_performance_predictor.py
├── AI-Data.csv
├── README.md
└── requirements.txt
```

## Sample Dataset Format

```csv
gender,NationalITy,PlaceofBirth,StageID,GradeID,SectionID,Topic,Semester,Relation,raisedhands,VisITedResources,AnnouncementsView,Discussion,ParentAnsweringSurvey,ParentschoolSatisfaction,StudentAbsenceDays,Class
M,KW,KuwaIT,Middle,G-07,A,Math,F,Father,15,16,2,20,Yes,Good,Under-7,M
F,KW,KuwaIT,Middle,G-07,A,Math,F,Father,20,20,3,25,Yes,Good,Under-7,M
```

## Customization Options

### Adding New Features
To include additional features in the prediction model:
1. Modify the feature selection in the data preprocessing section
2. Update the `feats` array to include new column indices
3. Adjust the custom input section accordingly

### Model Parameters
You can tune model parameters by modifying the classifier initialization:
```python
# Example: Tuning Random Forest
modelR = es.RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
```

## Known Limitations

- **Feature Selection**: Currently uses only 4 features for prediction
- **Data Requirements**: Expects specific column names and formats
- **Encoding**: Simple label encoding may not be optimal for all categorical variables
- **Validation**: Uses simple train-test split without cross-validation

## Future Enhancements

- [ ] Cross-validation for more robust evaluation
- [ ] Feature importance analysis and visualization
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Support for additional file formats (Excel, JSON)
- [ ] Web interface for easier interaction
- [ ] Model persistence (save/load trained models)
- [ ] Advanced preprocessing options

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Support

For questions and support:
- Create an issue in the GitHub repository
- Check the console output for error messages
- Ensure your dataset follows the required format

## Acknowledgments

- Built using scikit-learn for machine learning capabilities
- Seaborn and Matplotlib for data visualization
- Pandas for data manipulation and analysis
