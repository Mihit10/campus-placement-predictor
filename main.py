import pandas as pd
import numpy as np
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from time import sleep
from joblib import load
import json
pd.set_option('future.no_silent_downcasting', True)



def load_evaluation_metrics(filename='evaluation_metrics.json'):
    with open(filename, 'r') as f:
        metrics = json.load(f)
    return metrics

# evaluation_mat = {"Mean Squared Error" : 77.304,
#                 "Mean Absolute Error" : 5.760,
#                 "R-squared" : 0.681,
#                 "Cross-validated MSE" : 76.772,
#                 "Bias Squared": 0.011,
#                 "Variance": 165.81}

# def load_and_preprocess_data():
#     dataset = pd.read_csv("Placements_Dataset.csv")
#     dataset = dataset.iloc[:50000, :]
#     dataset = dataset.drop(columns=['Name of Student', 'Roll No.'])
#     dataset = dataset.dropna()
#     yes_no_columns = ['Knows ML', 'Knows DSA', 'Knows Python', 'Knows JavaScript', 
#                       'Knows HTML', 'Knows CSS', 'Knows Cricket', 'Knows Dance', 
#                       'Participated in College Fest', 'Was in Coding Club']
#     dataset[yes_no_columns] = dataset[yes_no_columns].replace({'Yes': 1, 'No': 0})
#     Q1 = dataset['Placement Package'].quantile(0.25)
#     Q3 = dataset['Placement Package'].quantile(0.75)
#     IQR = Q3 - Q1
#     # Filtering out the outliers
#     dataset = dataset[~((dataset['Placement Package'] < (Q1 - 1.5 * IQR)) | (dataset['Placement Package'] > (Q3 + 1.5 * IQR)))]
#     y = dataset.iloc[:, -1].values
#     X = dataset.iloc[:, :-1].values
#     return X, y
gbr_model = load('placement_model.joblib')
evaluation_mat = load_evaluation_metrics()

def getGraphs():
    dataset = pd.read_csv("Optimized_Placement_Dataset_with_Noise.csv")
    yes_no_columns = [
        'Knows ML', 'Knows DSA', 'Knows Python', 'Knows JavaScript', 
        'Knows HTML', 'Knows CSS', 'Knows Cricket', 'Knows Dance', 
        'Participated in College Fest', 'Was in Coding Club'
    ]
    dataset[yes_no_columns] = dataset[yes_no_columns].replace({'Yes': 1, 'No': 0})
    # Drop unnecessary columns if any
    if 'Name of Student' in dataset.columns and 'Roll No.' in dataset.columns:
        dataset = dataset.drop(columns=['Name of Student', 'Roll No.'])
    # Remove outliers based on Placement Package
    Q1 = dataset['Placement Package'].quantile(0.25)
    Q3 = dataset['Placement Package'].quantile(0.75)
    IQR = Q3 - Q1
    dataset = dataset[~((dataset['Placement Package'] < (Q1 - 1.5 * IQR)) | (dataset['Placement Package'] > (Q3 + 1.5 * IQR)))]

    numeric_dataset = dataset.select_dtypes(include=['float64', 'int64'])

    #heatmaps 
    plt.figure(figsize=(10, 6))
    correlation_matrix = numeric_dataset.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("heatmap.png")
    plt.close()

    # Scatterplot for CGPA vs Placement Package
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=dataset, x="CGPA", y="Placement Package", alpha=0.6)
    plt.title("Scatterplot: CGPA vs Placement Package")
    plt.xlabel("CGPA")
    plt.ylabel("Placement Package")
    plt.savefig("scatterplot_cgpa_vs_package.png")
    plt.close()


    # Scatterplot for No. of DSA questions vs Placement Package
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=dataset, x="No. of DSA questions", y="Placement Package", alpha=0.6)
    plt.title("Scatterplot: No. of DSA Questions vs Placement Package")
    plt.xlabel("No. of DSA Questions")
    plt.ylabel("Placement Package")
    plt.savefig("scatterplot_dsa_vs_package.png")
    plt.close()

    # Scatterplot: No. of Backlogs vs Placement Package
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=dataset, x="No. of backlogs", y="Placement Package", alpha=0.6)
    plt.title("Scatterplot: No. of Backlogs vs Placement Package")
    plt.xlabel("No. of Backlogs")
    plt.ylabel("Placement Package")
    plt.savefig("scatterplot_backlogs_vs_package.png")
    plt.close()


    # Boxplot: Branch of Engineering vs Placement Package
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataset, x="Branch of Engineering", y="Placement Package")
    plt.title("Boxplot: Branch of Engineering vs Placement Package")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("boxplot_branch_vs_package.png")
    plt.close()

    # Countplot: Students in Coding Club by Branch of Engineering
    plt.figure(figsize=(10, 6))
    sns.countplot(data=dataset, x="Was in Coding Club", hue="Branch of Engineering")
    plt.title("Countplot: Students in Coding Club by Branch of Engineering")
    plt.xlabel("Was in Coding Club (1 = Yes, 0 = No)")
    plt.ylabel("Count")
    plt.legend(title="Branch", loc="upper right")
    plt.tight_layout()
    plt.savefig("countplot_coding_club_by_branch.png")
    plt.close()

    # Histogram: Distribution of Placement Package
    plt.figure(figsize=(8, 6))
    sns.histplot(dataset['Placement Package'], bins=20, kde=True)
    plt.title("Histogram: Distribution of Placement Package")
    plt.xlabel("Placement Package")
    plt.ylabel("Frequency")
    plt.savefig("histogram_package_distribution.png")
    plt.close()

    cleaned_data = numeric_dataset.apply(pd.to_numeric, errors='coerce').dropna()  # Ensure clean data
    sns.pairplot(cleaned_data, diag_kind="kde")
    plt.savefig("pairplot_numerical_features.png")
    plt.close()


def input_and_ml():
    # X, y = load_and_preprocess_data()
    
    st.title("Campus Placement Predictor ")
    st.markdown("**Welcome to Campus Placement Predictor**  \nEnter some information about you to get an estimate about you annual package which you can expect!!!")

    encoding_dict = {"Computer Science" : [[0.0, 1.0, 0.0, 0.0]],
                "Mechanical Engineering" : [[0.0, 0.0, 0.0, 1.0]],
                "Electrical Engineering" : [[0.0, 0.0, 1.0, 0.0]],
                "Civil Engineering" : [[1.0, 0.0, 0.0, 0.0]]}
    
    set_none = st.checkbox("Do you know the interview room temperature?")
    # Create a form
    with st.form(key='my_form'):
        # Input fields
        name_of_candidate = st.text_input("Name of candidate", value=None)
        rollno_of_candidate = st.text_input("Roll No of candidate", value=None)
        age_of_candidate = st.number_input("Age of Candidate", min_value=1, max_value=120, value=None, step=1, format="%d")
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, format="%.2f", value=None)
        num_dsa_questions = st.number_input("No. of DSA questions", min_value=0, value=None)
        num_backlogs = st.number_input("No. of backlogs", min_value=0, max_value=5, value=None)
        if set_none:
            interview_room_temp = st.slider("Interview Room Temperature (°C)",min_value=10.0, max_value=40.0, value=22.0, step=0.1, format="%.1f")
        else:
            interview_room_temp = None
        
        knows_ml = st.checkbox("Know ML")
        knows_dsa = st.checkbox("Know DSA")
        knows_python = st.checkbox("Know Python")
        knows_JavaScript = st.checkbox("Know JavaScript")
        knows_HTML = st.checkbox("Know HTML")
        knows_CSS = st.checkbox("Know CSS")
        knows_Cricket = st.checkbox("Know Cricket")
        knows_Dance = st.checkbox("Know Dance")
        Participated_in_College_Fest = st.checkbox("Participated in College Fest")
        Was_in_Coding_Club = st.checkbox("Was in Coding Club")

        branch_of_engineering = st.selectbox(
            "Branch of Engineering",
            ['Select a branch', 'Computer Science', 'Mechanical Engineering', 'Electrical Engineering', 'Civil Engineering'])
        
        # Submit button
        submit_button = st.form_submit_button(label='Submit')

    # After form submission
    if submit_button:
        # Set default values to NULL if inputs are empty
        cgpa = cgpa if cgpa != 0.0 else np.nan
        num_dsa_questions = num_dsa_questions if num_dsa_questions != -1 else np.nan
        num_backlogs = num_backlogs if num_backlogs != -1 else np.nan
        interview_room_temp = interview_room_temp if interview_room_temp is not None else np.nan
        age_of_candidate = age_of_candidate if age_of_candidate != 0 else np.nan
        
        # Convert checkboxes to boolean values 1/0
        knows_ml = int(knows_ml) if knows_ml else 0
        knows_dsa = int(knows_dsa) if knows_dsa else 0
        knows_python = int(knows_python) if knows_python else 0
        knows_JavaScript = int(knows_JavaScript) if knows_JavaScript else 0
        knows_HTML = int(knows_HTML) if knows_HTML else 0
        knows_CSS = int(knows_CSS) if knows_CSS else 0
        knows_Cricket = int(knows_Cricket) if knows_Cricket else 0
        knows_Dance = int(knows_Dance) if knows_Dance else 0
        Participated_in_College_Fest = int(Participated_in_College_Fest) if Participated_in_College_Fest else 0
        Was_in_Coding_Club = int(Was_in_Coding_Club) if Was_in_Coding_Club else 0

        if name_of_candidate == None:
            st.error("Please enter the Name of the candidate.")
        else:
            st.write("Name of candidate:", name_of_candidate)

        if rollno_of_candidate == None:
            st.error("Please enter the Roll No of the candidate.")
        else:
            st.write("Roll No of candidate:", rollno_of_candidate)

        if branch_of_engineering == 'Select a branch':
            st.error("Please enter the Branch of the candidate.")

        # displaying values
        if name_of_candidate != None and rollno_of_candidate != None and branch_of_engineering != 'Select a branch':
            input = {
                "No. of DSA questions": num_dsa_questions,
                "CGPA": cgpa,
                "Knows ML": knows_ml,
                "Knows DSA": knows_dsa,
                "Knows Python": knows_python,
                "Knows JavaScript": knows_JavaScript,  
                "Knows HTML": knows_HTML,                
                "Knows CSS": knows_CSS,
                "Knows Cricket": knows_Cricket,
                "Knows Dance": knows_Dance,
                "Participated in College Fest": Participated_in_College_Fest, 
                "Was in Coding Club": Was_in_Coding_Club,              
                "No. of backlogs": num_backlogs,
                "Interview Room Temperature (°C)": interview_room_temp,
                "Age of Candidate": age_of_candidate,
                "Branch of Engineering": branch_of_engineering
            }

            input_df = pd.DataFrame([input])
            st.dataframe(input_df, hide_index=True)
            progressBar = st.progress(0, "Initiating process...")
            sleep(0.4)

            # Convert input values to a 2D NumPy array
            x_values = np.array(list(input.values()), dtype=object).reshape(1, -1)  # Reshape to 2D array

            progressBar.progress(10, "Handling missing values...")
            sleep(0.5)
            # Handling null values 
            mean_imp = load('imp_mean.joblib')
            mode_imp = load('imp_mode.joblib')
            
            # imputer_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            # imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
            # imputer_mode.fit(X[:, 2:12])
            # imputer_mean.fit(X[:, [0, 1, 12, 13, 14]])
            x_values = np.where(x_values == None, np.nan, x_values)
            x_values[:, 2:12] = mode_imp.transform(x_values[:, 2:12])
            x_values[:, [0, 1, 12, 13, 14]] = mean_imp.transform(x_values[:, [0, 1, 12, 13, 14]])

            progressBar.progress(20, "Encoding data...")
            sleep(0.5)

            # one hot encoding of dataset and input
            # ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [15])], remainder='passthrough')
            # X = np.array(ct.fit_transform(X))

            # last_column = x_values[0, -1]
            # if last_column in encoding_dict:
            #     one_hot_encoded = np.array(encoding_dict[last_column])
            #     x_values_reshaped = x_values.reshape(1, -1)
            #     x_values = np.concatenate((one_hot_encoded, x_values_reshaped), axis=1)
            #     x_values = x_values[:, :-1]

            ct = load('column_transformer.joblib')
            x_values = np.array(ct.transform(x_values))

            progressBar.progress(30, "Performing Feature Scaling...")
            sleep(0.5)

            # feature scaling
                # on X
            # x_values = x_values.astype(float)
            columns_to_scale = [4, 5, 16, 17, 18]
            # sc = StandardScaler()
            # X_to_scale = X[:, columns_to_scale]
            # X_rest = np.delete(X, columns_to_scale, axis=1)
            # X_scaled = sc.fit_transform(X_to_scale)
            # X[:, columns_to_scale] = X_scaled
                # on input
            
            # x_values_to_scale = x_values[:, columns_to_scale]
            # x_values_rest = np.delete(x_values, columns_to_scale, axis=1)
            # x_values_scaled = sc.transform(x_values_to_scale)
            # x_values[:, columns_to_scale] = x_values_scaled
            scale_x = load('x_scaler.joblib')
            x_values[:, columns_to_scale] = scale_x.transform(x_values[:, columns_to_scale])


            # print("input - ", x_values[0])
            # print("original - ", X[0])

            progressBar.progress(50, "Creating ML algorithm...")
            sleep(0.2)
            # gradient boosting 
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


            progressBar.progress(70, "Calculating values...")

            #test
            # best_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_features=5, random_state=100)
            # best_gbr.fit(X_train, y_train)

            # y_pred = best_gbr.predict(X_test)
            # y_train_pred = best_gbr.predict(X_train)

            # mse = mean_squared_error(y_test, y_pred)
            # mse_train = mean_squared_error(y_train, y_train_pred)

            # bias_squared = (np.mean(y_test) - np.mean(y_pred)) ** 2
            # variance = np.var(y_pred)

            # mae = mean_absolute_error(y_test, y_pred)
            # r2 = r2_score(y_test, y_pred)
            # cv_scores = cross_val_score(best_gbr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

            # st.write(evaluation_mat)

            progressBar.progress(85, "Almost there...")
            sleep(0.2)

            # evaluation_mat = {"Mean Squared Error" : mse,
            #                 "Mean Absolute Error" : mae,
            #                 "R-squared" : r2,
            #                 "Cross-validated MSE" : -np.mean(cv_scores),
            #                 "Bias Squared": bias_squared,
            #                 "Variance": variance}
            
            # st.write(f'Mean Squared Error: {mse}')
            # st.write(f'Mean Absolute Error: {mae}')
            # st.write(f'R-squared: {r2}')
            # st.write(f'Cross-validated MSE: {-np.mean(cv_scores)}')
            # st.write(f'Bias Squared: {bias_squared}')
            # st.write(f'Variance: {variance}')

            progressBar.progress(100, "DONE!!!...")

            # st.write("Shape of x_values before prediction:", x_values.shape)
            # st.write("x_val after hot encoding", x_values[0])

            # Predicting Value 
            scale_y = load("y_scaler.joblib")
            predicted_y = gbr_model.predict(x_values)
            predicted_y = scale_y.inverse_transform(predicted_y.reshape(-1, 1)).flatten()
            predicted_y = np.round(predicted_y, 2)
            pred_package = predicted_y[0]

            st.markdown("<h3 style='text-align: center;'>Your expected annual package is:</h3>", unsafe_allow_html=True)

            # Display the predicted value in a larger font
            st.markdown(f"<h1 style='text-align: center; color: green;'>{pred_package} LPA</h1>", unsafe_allow_html=True)

            # Display the warning message in smaller text
            st.markdown(f"<p style='text-align: center; color: red; font-size: 14px;'>The values calculated can/may contain an error margin of ± {evaluation_mat['Mean Absolute Error']:.2f}.</p>", unsafe_allow_html=True)

def info_tab():
    st.title("Model Information")
    st.markdown(f"""
    ## **Campus Placement Predictor Model**

    Our **Campus Placement Predictor** model is a data-driven tool designed to forecast a student’s expected annual placement package by analyzing a comprehensive set of features. This model factors in academic performance, technical proficiency (such as familiarity with Machine Learning, Python, and DSA skills), extracurricular participation, and even personal attributes like age and interview conditions. Using these inputs, the model generates a customized estimate of the annual package a student might expect in campus placements, offering valuable insights that can guide students in career planning and skill development.

    ### **Why Gradient Boosting?**
    To select the most accurate and reliable model, I evaluated multiple algorithms, including **Random Forest**, **Support Vector Regression (SVR)**, **K-Nearest Neighbors (KNN)**, **Decision Tree**, and **Linear Regression**. Each model was tested on essential evaluation metrics: **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, **R-squared**, and **Cross-Validation MSE**. Among these, **Gradient Boosting** performed the best, delivering the lowest MSE and a higher R-squared score compared to other models, indicating greater accuracy and consistency. Here’s a brief comparison:

    - **Gradient Boosting**: Cross-validation MSE of {evaluation_mat['Mean Squared Error']:.3f}, MAE of {evaluation_mat['Mean Absolute Error']:.3f}, and R-squared of {evaluation_mat['R-squared']:.3f}
    - **Random Forest**: Cross-validation MSE of 3.105, MAE of 0.762, and R-squared of 0.937
    - **Support Vector Regression, KNN, Decision Tree,** and **Linear Regression** models had significantly higher MSE and lower R-squared scores, showing lesser predictive power for this dataset.

        #### **How Gradient Boosting Works**

        Gradient Boosting is an ensemble learning method that builds on weak learners, typically decision trees, to form a strong predictive model. It works iteratively, with each tree correcting the errors of the previous one. This approach minimizes bias and variance, creating a highly accurate model well-suited for complex datasets. Gradient Boosting also offers **tuning flexibility** with parameters like learning rate and the number of estimators, allowing fine-tuning to further enhance model performance. 

    ### **Model Evaluation and Performance**
    The **Campus Placement Predictor** model has been evaluated on multiple metrics to ensure its reliability and accuracy. Key performance indicators include **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **R-squared**, along with **Cross-Validated MSE**, **Bias Squared**, and **Variance**.

    - **Mean Squared Error (MSE)**: `{evaluation_mat['Mean Squared Error']:.3f}` - measures average squared differences between predictions and actual values, indicating the overall error.
    - **Mean Absolute Error (MAE)**: `{evaluation_mat['Mean Absolute Error']:.3f}` - reflects the average absolute difference, providing insight into prediction accuracy.
    - **R-squared**: `{evaluation_mat['R-squared']:.3f}` - shows the proportion of variance explained by the model, with a value closer to 1 indicating better fit.
    - **Cross-validated MSE**: `{evaluation_mat['Cross-validated MSE']:.3f}` - confirms model consistency and generalizability across multiple folds.
    - **Bias Squared**: `{evaluation_mat['Bias Squared']}` - captures model accuracy relative to the true data.
    - **Variance**: `{evaluation_mat['Variance']:.3f}` - reflects the model’s ability to adapt to the data without overfitting.

    These metrics underscore the model’s effectiveness, with a low bias and a high R-squared, suggesting it accurately predicts placement packages with minimal error. The low variance also highlights its robustness across different samples, making it a dependable tool for real-world applications.

    """)

    st.markdown("""
    ### **Visualizations**

    Below are the graphs generated to better understand the dataset and its relationships. Each graph is accompanied by a brief explanation, the generated visualization, and insights drawn from it.

    #### **1. Heatmap**
    Heatmaps are effective for visualizing the strength of correlations between numerical variables in a dataset. Correlation coefficients range from -1 to 1, where:
    - **1** indicates a perfect positive correlation,
    - **-1** indicates a perfect negative correlation,
    - **0** indicates no correlation.

    **Generated Heatmap:**
    """)
    st.image("heatmap.png", caption="Correlation Heatmap")
    st.markdown("""
    **Conclusion:**
    - Variables like `CGPA` and `No. of DSA questions` show strong positive correlations with `Placement Package`, indicating their importance in predicting placement outcomes.
    - Features such as `No. of backlogs` have a weaker correlation, suggesting a lesser impact on placements.

    #### **2. Scatterplots**
    Scatterplots help visualize the relationship between two variables. Below are key scatterplots for this dataset:

    ##### **Scatterplot: CGPA vs Placement Package**
    """)
    st.image("scatterplot_cgpa_vs_package.png", caption="Scatterplot: CGPA vs Placement Package")
    st.markdown("""
    **Conclusion:**
    - Higher CGPA generally correlates with higher placement packages.
    - However, outliers indicate CGPA alone isn't the sole determinant.

    ##### **Scatterplot: No. of DSA Questions vs Placement Package**
    """)
    st.image("scatterplot_dsa_vs_package.png", caption="Scatterplot: No. of DSA Questions vs Placement Package")
    st.markdown("""
    **Conclusion:**
    - Students solving more DSA questions tend to secure better packages, emphasizing the importance of technical skills.

    ##### **Scatterplot: No. of Backlogs vs Placement Package**
    """)
    st.image("scatterplot_backlogs_vs_package.png", caption="Scatterplot: No. of Backlogs vs Placement Package")
    st.markdown("""
    **Conclusion:**
    - An increasing number of backlogs tends to correlate with lower placement packages, highlighting the importance of academic consistency.

    #### **3. Boxplot**
    Boxplots display the distribution of placement packages across engineering branches, helping identify branch-specific trends.

    **Generated Boxplot:**
    """)
    st.image("boxplot_branch_vs_package.png", caption="Boxplot: Branch of Engineering vs Placement Package")
    st.markdown("""
    **Conclusion:**
    - Certain branches (e.g., Computer Science) show higher median packages.
    - Some branches exhibit a wider range, suggesting varied opportunities based on specialization.

    #### **4. Histogram**
    Histograms provide an overview of the distribution of placement packages in the dataset.

    **Generated Histogram:**
    """)
    st.image("histogram_package_distribution.png", caption="Histogram: Distribution of Placement Package")
    st.markdown("""
    **Conclusion:**
    - Most placement packages fall within a specific range, indicating a general trend or norm.
    - A few outliers reflect exceptionally high or low packages.

    #### **5. Countplot**
    Countplots visualize the distribution of categorical variables, such as participation in the coding club.

    **Generated Countplot:**
    """)
    st.image("countplot_coding_club_by_branch.png", caption="Countplot: Students in Coding Club by Branch of Engineering")
    st.markdown("""
    **Conclusion:**
    - Participation in coding clubs varies significantly across branches, reflecting branch-specific interests and opportunities.

    #### **6. Pairplot**
    Pairplots provide a holistic view of relationships among multiple numerical variables in the dataset.

    **Generated Pairplot:**
    """)
    st.image("pairplot_numerical_features.png", caption="Pairplot: Numerical Features and Placement Package")
    st.markdown("""
    **Conclusion:**
    - The pairplot confirms that certain features (e.g., CGPA, DSA questions) are positively correlated with placement packages, while others (e.g., backlogs) show a negative trend.

    ### **Final Insights**
    - The dataset highlights the importance of technical skills, academic performance, and participation in extracurricular activities in securing better placement packages.
    - Some features (e.g., CGPA, DSA questions) emerge as stronger predictors, while others have limited impact.
    - The visualizations emphasize that while numerical performance is critical, other factors also play a role in determining placement outcomes.
    """)



def main():
    # Create tabs
    tabs = st.tabs(["Model", "Info"])

    with tabs[0]:
        input_and_ml()

    with tabs[1]:
        # getGraphs()   (uncomment if you want to generate graphs again)
        info_tab()

if __name__ == "__main__":
    main()