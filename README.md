# Campus Placement Predictor

**Campus Placement Predictor** is a machine learning project designed to predict the likelihood of a student's placement based on their academic performance, test scores, and other relevant features. The project uses **Gradient Boosting Regression** to provide accurate predictions, empowering universities and companies to streamline the campus recruitment process.

---

## Features

- Predicts the likelihood of student placement.
- Utilizes **Gradient Boosting Regression** for enhanced performance.
- Supports data preprocessing and feature selection for better accuracy.
- User-friendly interface for predictions.
- Optimized for analyzing key factors affecting placements.

---

## Table of Contents

1. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
2. [Setup](#setup)
3. [Dataset](#dataset)
4. [Model Details](#model-details)
5. [Results](#results)
6. [Technologies Used](#technologies-used)
7. [Contributing](#contributing)


---

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.7 or higher ( creaeted in python 3.11.9 )
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, joblib, json, time

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Mihit10/campus-placement-predictor.git
   ```

2. Navigate to the project directory:

   ```bash
   cd campus-placement-predictor
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---
### setup

- If you clone the entire reopsitory, you can directly run the main.py file on streamlit webApp
- For training/modifying data, you can modify model.py and run it to regenerate .joblib files. Then run main.py normally.
- If graphs are not installed, uncomment the getGraphs() function in main() function of main.py file.
- Analysis.ipynb is used for data analysis and has no dependency on main.py
- pds_project.ipynb is used for testing of features and models before implementing in model.py
- To train and run the model independently you need main.py and model.py 

## Dataset

The dataset comprises 218,734 rows, each representing a student's profile with the following columns:

- **Numerical Columns**: Includes `No. of DSA questions`, `CGPA`, `No. of backlogs`, `Interview Room Temperature`, and `Age of Candidate`.
- **Yes/No Columns**: Captures binary skills and activities like `Knows ML`, `Knows DSA`, `Knows Python`, `Knows HTML`, `Knows Cricket`, `Participated in College Fest`, and more.
- **Categorical Column**: `Branch of Engineering` represents the student's engineering branch.
- **Target Variable**: `Placement Package`, the placement package offered, serves as the prediction target.

This diverse dataset enables a comprehensive analysis of factors influencing placement outcomes.

---

## Model Details

- **Algorithm**: Gradient Boosting Regression
- **Hyperparameters**:
  - Learning rate: `0.1`
  - Number of estimators: `120`
  - Maximum depth: `4`
  - max_features: `6`

---

## Results

The model achieves the following metrics on the test set:
- **Mean Squared Error (MSE)**: 2.6415  
- **Mean Absolute Error (MAE)**: 0.6132  
- **R-squared (R²)**: 0.9414  
- **Cross-validated MSE**: 0.0602  
- **Bias Squared**: 0.0001  
- **Variance**: 40.7749  

---

## Technologies Used

- **Python**: The core programming language for data processing and model development.  
- **Gradient Boosting (scikit-learn)**: Utilized for implementing the regression model.  
- **Data Visualization**: Libraries like Matplotlib and Seaborn were used for exploratory data analysis and visualizations.  
- **Jupyter Notebook**: Facilitated interactive development and experimentation with the model.  
- **Streamlit**: Used to create a user-friendly web interface for model predictions and real-time interactions.  
- **Joblib**: Employed for saving and loading the trained model efficiently, ensuring ease of deployment.

---

## Contributing

Contributions are welcome! Follow these steps to contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with detailed changes.

---
