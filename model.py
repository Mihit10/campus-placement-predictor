import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
pd.set_option('future.no_silent_downcasting', True)

def save_evaluation_metrics(metrics, filename='evaluation_metrics.json'):
    with open(filename, 'w') as f:
        json.dump(metrics, f)

evaluation_mat = {"Mean Squared Error" : 77.304,
                "Mean Absolute Error" : 5.760,
                "R-squared" : 0.681,
                "Cross-validated MSE" : 76.772,
                "Bias Squared": 0.011,
                "Variance": 165.81}


def load_and_preprocess_data():
    dataset = pd.read_csv("Placements_Dataset.csv")
    # dataset = dataset.iloc[:50000, :]
    dataset = dataset.drop(columns=['Name of Student', 'Roll No.'])
    dataset = dataset.dropna()
    dataset['Placement Package'] -= 4
    dataset['Placement Package'] = dataset['Placement Package'].apply(lambda x: np.random.uniform(1, 2) if x <= 0 else x)
    
    # Convert Yes/No columns to binary
    yes_no_columns = ['Knows ML', 'Knows DSA', 'Knows Python', 'Knows JavaScript', 
                      'Knows HTML', 'Knows CSS', 'Knows Cricket', 'Knows Dance', 
                      'Participated in College Fest', 'Was in Coding Club']
    dataset[yes_no_columns] = dataset[yes_no_columns].replace({'Yes': 1, 'No': 0})
    
    # Remove outliers based on Placement Package
    Q1 = dataset['Placement Package'].quantile(0.25)
    Q3 = dataset['Placement Package'].quantile(0.75)
    IQR = Q3 - Q1
    dataset = dataset[~((dataset['Placement Package'] < (Q1 - 1.5 * IQR)) | (dataset['Placement Package'] > (Q3 + 1.5 * IQR)))]
    # print(f"Count of zeros in the 'Placement Package' column: {(dataset['Placement Package'] == 0).sum()}")
    y = dataset.iloc[:, -1].values  # Target variable
    X = dataset.iloc[:, :-1].values  # Features

    # # Set the style of seaborn
    # sns.set(style="whitegrid")

    # # Create a histogram
    # plt.figure(figsize=(10, 6))
    # sns.histplot(dataset['Placement Package'], bins=20, kde=True)  # kde=True adds a kernel density estimate
    # plt.title('Distribution of Placement Package')
    # plt.xlabel('Placement Package')
    # plt.ylabel('Frequency')
    # plt.show()

    
    return X, y


def train_model(X, y):
    global evaluation_mat
    # Missing values
    imputer_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer_mode.fit(X[:, 2:12])
    imputer_mean.fit(X[:, [0, 1, 12, 13, 14]])
    joblib.dump(imputer_mode, 'imp_mode.joblib')
    joblib.dump(imputer_mean, 'imp_mean.joblib')
    
    
    # one hot encoding
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [15])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

    # Step 2: Scale features
    columns_to_scale = [0, 1, 2, 3, 4]  # Adjust based on your feature indices
    xsc = StandardScaler()
    X_train[:, columns_to_scale] = xsc.fit_transform(X_train[:, columns_to_scale])
    X_test[:, columns_to_scale] = xsc.transform(X_test[:, columns_to_scale])
    joblib.dump(xsc, 'x_scaler.joblib')

    # Step 3: Scale target variable
    ysc = StandardScaler()
    y_train = ysc.fit_transform(y_train.reshape(-1, 1)).flatten()  # Flatten to 1D
    y_test = ysc.transform(y_test.reshape(-1, 1)).flatten()  # Flatten to 1D
    joblib.dump(ysc, 'y_scaler.joblib')

    # Step 4: Train the model
    best_gbr = GradientBoostingRegressor(n_estimators=80, min_samples_split=10, min_samples_leaf=4, max_depth=3, learning_rate=0.1, max_features=None, random_state=100)
    best_gbr.fit(X_train, y_train)  # y_train is now 1D

    # Step 5: Make predictions
    y_pred_scaled = best_gbr.predict(X_test)


    # train_scores = []
    # val_scores = []

    # for n_estimators in range(50, 200, 2):
    #     best_gbr.n_estimators = n_estimators
    #     best_gbr.fit(X_train, y_train)
    #     train_scores.append(mean_squared_error(y_train, best_gbr.predict(X_train)))
    #     val_scores.append(mean_squared_error(y_test, best_gbr.predict(X_test)))
    #     print(train_scores[-1], n_estimators)

    # # Plot the training and validation scores
    # import matplotlib.pyplot as plt

    # plt.plot(range(50, 200, 2), train_scores, label='Training MSE')
    # plt.plot(range(50, 200, 2), val_scores, label='Validation MSE')
    # plt.xlabel('Number of Estimators')
    # plt.ylabel('Mean Squared Error')
    # plt.legend()
    # plt.show()

    # Step 6: Inverse transform the predictions to get them back to the original scale
    y_pred = ysc.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten() 
    y_test = ysc.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    bias_squared = (np.mean(y_test) - np.mean(y_pred)) ** 2
    bias_squared_formatted = f"{1000000 * bias_squared:.3f} x 10^-6"

    variance = np.var(y_pred)
    cv_scores = cross_val_score(best_gbr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

    evaluation_mat = {"Mean Squared Error" : float(mse),
                    "Mean Absolute Error" : float(mae),
                    "R-squared" : float(r2),
                    "Cross-validated MSE" : float(-np.mean(cv_scores)),
                    "Bias Squared": str(bias_squared_formatted),
                    "Variance": float(variance)}

    import matplotlib.pyplot as plt

    feature_importances = best_gbr.feature_importances_
    plt.barh(range(len(feature_importances)), feature_importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Index')
    plt.title('Feature Importance from Gradient Boosting')
    plt.show()

    # Save the model
    joblib.dump(best_gbr, 'placement_model.joblib')

    return evaluation_mat


def main():
    save_evaluation_metrics(evaluation_mat)
    X, y = load_and_preprocess_data()
    evaluation_metrics = train_model(X, y)
    save_evaluation_metrics(evaluation_metrics)

    # print(X[0])
    print(evaluation_metrics)
    return evaluation_metrics

if __name__ == "__main__":
    main()
