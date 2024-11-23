# Machine Learning Development Project

## Project Overview
This project involves applying a Machine Learning workflow to perform binary classification using two datasets:
- **Banknote Authentication Dataset** ([UCI Repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication))
- **Chronic Kidney Disease Dataset** ([Kaggle](https://www.kaggle.com/mansoordaku/ckdisease))

The main objectives include the implementation of good programming practices, collaborative work using Git, and the creation of an end-to-end machine learning pipeline.

## Objectives
- Develop and follow good programming practices.
- Utilize standard development tools and collaborate effectively using Git for version control.
- Implement and validate a comprehensive machine learning workflow.
- Apply the workflow to different datasets and compare model performances.

## Workflow Description
The machine learning workflow involves:
1. **Importing the Dataset**: Load the data from relevant sources.
2. **Data Preprocessing**:
   - Handle missing values (e.g., replace them with averages or medians).
   - Center and normalize the data.
3. **Dataset Splitting**:
   - Create training and test sets.
   - Implement cross-validation by further splitting the training set.
4. **Model Training**:
   - Apply up to five different classification models.
   - Conduct feature selection to optimize performance.
5. **Model Validation**:
   - Compare the results using key metrics (accuracy, precision, recall).

## Project Structure
- **Main Python Script (`ml_functions.py`)**:
  - Contains functions for data preprocessing, dataset preparation, model training, and result display.
- **Jupyter Notebook (`main.ipynb`)**:
  - Demonstrates the application of the functions on both datasets.
  - Includes visual comparisons and results interpretation.

## Models Used
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier

## Tools and Technologies
- **Programming Language**: Python 3.8.10
- **Libraries**:
  - `pandas` for data manipulation
  - `numpy` for numerical computations
  - `scikit-learn` for model building and evaluation
  - `matplotlib` for visualization
  - `scipy` for handling specific data formats
- **Version Control**: Git for collaborative work and versioning

## Results and Discussion
Each model's performance is displayed and analyzed through:
- **Metrics**: Accuracy, precision, and recall scores for comparisons.
- **Visualizations**: Graphs and tables that highlight model effectiveness.
- **Final Analysis**: Commentary on which model performed best for each dataset and a recommendation for use.

## Deliverables
- **Repository Structure**:
  - `functions.py`: Main Python script with core functions.
  - `main.ipynb`: Notebook showcasing the workflow application.
  - Datasets and any other necessary files.
- **Submission**: Ensure the Git repository is shared with the instructor with appropriate access rights before the deadline (November 20th, 11 PM).

## Good Programming Practices
This project adheres to:
- Modular coding with reusable functions.
- Documented code and inline comments.
- Proper version control using branches for collaborative development.

## Getting Started
1. **Clone the Repository** :
   ```bash
   git clone <repository-url>
   cd <repository-directory>

2. **Install Dependecies** :
   ```bash
   pip install -r requirements.txt

3. **Run the Notebook** : Open and run main.ipynb to see the models in action.

##  Contributors
This project was developed as part of the TAF MCE course led by Elsa Dupraz at IMT Atlantique.

## Authors
- Khalil ABDELHEDI, email: khalil.abdelhedi@imt-atlantique.net
- Skander MAHJOUB, email: skander.mahjoub@imt-atlantique.net
- Ibrahim ABID, email: ibrahim.abid@imt-atlantique.net
- Jonathan LEMENTEC, email : jonathan.le-mentec@imt-atlantique.net

## References
[Reproducible Machine Learning Guidelines](https://mikecroucher.github.io/reproducible_ML/)
