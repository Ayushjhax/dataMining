# Predicting Diabetes: README

## Project Overview

This project aims to develop a machine-learning model to predict the likelihood of diabetes in patients based on various medical parameters. The dataset used for this project is the Pima Indians Diabetes Database, which contains several features such as glucose level, blood pressure, insulin level, and BMI.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Results](#results)

## Dataset

The dataset used in this project is taken from Kaggle and was originally contributed by National Institute of Diabetes and Digestive and Kidney Diseases.

Link to The Dataset : [Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)

Feature Set Exploration :

- Pregnancies
- Glucose
- blood pressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (1 for diabetes, 0 for no diabetes)

## Requirements

To run this project, you need the following libraries and packages installed:

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
  
  Alternatively, you can run it on : 
- Google Colab Notebook

You can install the required libraries using the following command:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

## Installation

Clone the repository to your local machine using:

```bash
git clone https://github.com/Ayushjhax/dataMining.git
cd Predicting_Diabetes
```

## Usage

1. Ensure all the required libraries are installed.
2. Open the Colab Notebook `Predicting_Diabetes.ipynb` to see the step-by-step implementation and analysis.
3. Run the notebook cells to preprocess the data, train the model, and evaluate its performance.

## Model Evaluation

The following machine learning models were evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Naive Bayes

The models were evaluated based on the following metrics:

- Accuracy
- Precision
- Recall

## Results

The Random Forest Classifier performed the best with an accuracy of 75% and,  an F1 score of 65% after Model Tuning and adding hyperparameters.

## Demo Video


https://github.com/Ayushjhax/dataMining/assets/116433617/763b1f3f-3981-487e-8aa7-bc7ca4be909c

## Authors
- Ayush Kumar Jha and Himanshu Rawat
- GitHub: @Ayushjhax and @himanshu-rawat77

## Feedback

If you have any questions, feedback, or issues, please don't hesitate to reach out to [Ayush Kumar Jha](https://www.linkedin.com/in/ayushjhax/)  or [Himanshu Rawat](https://www.linkedin.com/in/himanshu-rawat-1011sh/) or Email us at ayushkmrjha@yahoo.com or himanshu.rawat7789@gmail.com.
