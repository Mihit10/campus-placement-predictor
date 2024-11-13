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
from time import sleep
pd.set_option('future.no_silent_downcasting', True)


# dataset processing
dataset = pd.read_csv("Placements_Dataset.csv")
dataset = dataset.iloc[:50000,:]
dataset = dataset.drop(columns=['Name of Student', 'Roll No.'])
dataset = dataset.dropna()
yes_no_columns = ['Knows ML', 'Knows DSA', 'Knows Python', 'Knows JavaScript', 
                  'Knows HTML', 'Knows CSS', 'Knows Cricket', 'Knows Dance', 
                  'Participated in College Fest', 'Was in Coding Club']
dataset[yes_no_columns] = dataset[yes_no_columns].replace({'Yes': 1, 'No': 0})
y = dataset.iloc[:,-1].values
X = dataset.iloc[:,:-1].values

y_min = y.min()
y_max = y.max()
y_mean = y.mean()

st.write(f"y min : {y_min}")
st.write(f"y max : {y_max}")
st.write(f"y mean : {y_mean}")

encoding_dict = {"Computer Science" : [[0.0, 1.0, 0.0, 0.0]],
                 "Mechanical Engineering" : [[0.0, 0.0, 0.0, 1.0]],
                 "Electrical Engineering" : [[0.0, 0.0, 1.0, 0.0]],
                 "Civil Engineering" : [[1.0, 0.0, 0.0, 0.0]]}


st.title("Campus Placement Predictor ")
st.markdown("**Welcome to Campus Placement Predictor**  \nEnter some information about you to get an estimate about you annual package which you can expect!!!")


X_content = ["No. of DSA questions","CGPA","Knows ML","Knows DSA","Knows Python","Knows JavaScript","Knows HTML","Knows CSS","Knows Cricket","Knows Dance","Participated in College Fest","Was in Coding Club","No. of backlogs","Interview Room Temperature","Age of Candidate","Branch of Engineering"]

set_none = st.checkbox("Do you know the interview room temperature?")
# Create a form
with st.form(key='my_form'):
    # Input fields
    name_of_candidate = st.text_input("Name of candidate", value=None)
    rollno_of_candidate = st.text_input("Roll No of candidate", value=None)
    age_of_candidate = st.number_input("Age of Candidate", min_value=1, max_value=120, value=None, step=1, format="%d")
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, format="%.2f", value=None)
    num_dsa_questions = st.number_input("No. of DSA questions", min_value=0, value=None)
    num_backlogs = st.number_input("No. of backlogs", min_value=0, value=None)

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
        imputer_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer_mode.fit(X[:, 2:12])
        imputer_mean.fit(X[:, [0, 1, 12, 13, 14]])
        x_values = np.where(x_values == None, np.nan, x_values)
        x_values[:, 2:12] = imputer_mode.transform(x_values[:, 2:12])
        x_values[:, [0, 1, 12, 13, 14]] = imputer_mean.transform(x_values[:, [0, 1, 12, 13, 14]])

        progressBar.progress(20, "Encoding data...")
        sleep(0.5)

        # one hot encoding of dataset and input
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [15])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))
        last_column = x_values[0, -1]
        if last_column in encoding_dict:
            one_hot_encoded = np.array(encoding_dict[last_column])
            x_values_reshaped = x_values.reshape(1, -1)
            x_values = np.concatenate((one_hot_encoded, x_values_reshaped), axis=1)
            x_values = x_values[:, :-1]

        progressBar.progress(30, "Performing Feature Scaling...")
        sleep(0.5)

        # feature scaling
            # on X
        x_values = x_values.astype(float)
        columns_to_scale = [4, 5, 16, 17, 18]
        sc = StandardScaler()
        X_to_scale = X[:, columns_to_scale]
        X_rest = np.delete(X, columns_to_scale, axis=1)
        X_scaled = sc.fit_transform(X_to_scale)
        X[:, columns_to_scale] = X_scaled
            # on input
        x_values_to_scale = x_values[:, columns_to_scale]
        x_values_rest = np.delete(x_values, columns_to_scale, axis=1)
        x_values_scaled = sc.transform(x_values_to_scale)
        x_values[:, columns_to_scale] = x_values_scaled


        # print("input - ", x_values[0])
        # print("original - ", X[0])

        progressBar.progress(50, "Creating ML algorithm...")
        sleep(0.2)
        # gradient boosting 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


        progressBar.progress(70, "Calculating values...")

        #test
        best_gbr = GradientBoostingRegressor(n_estimators=100, min_samples_split=10, min_samples_leaf=4, max_depth=3, learning_rate=0.1, max_features=None, random_state=100)
        best_gbr.fit(X_train, y_train)

        y_pred = best_gbr.predict(X_test)
        y_train_pred = best_gbr.predict(X_train)

        mse = mean_squared_error(y_test, y_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)

        bias_squared = (np.mean(y_test) - np.mean(y_pred)) ** 2
        variance = np.var(y_pred)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(best_gbr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

        progressBar.progress(85, "Almost there...")
        sleep(0.2)

        evaluation_mat = {"Mean Squared Error" : mse,
                          "Mean Absolute Error" : mae,
                          "R-squared" : r2,
                          "Cross-validated MSE" : -np.mean(cv_scores),
                          "Bias Squared": bias_squared,
                          "Variance": variance}
        

        st.write(f'Mean Squared Error: {mse}')
        st.write(f'Mean Absolute Error: {mae}')
        st.write(f'R-squared: {r2}')
        st.write(f'Cross-validated MSE: {-np.mean(cv_scores)}')
        st.write(f'Bias Squared: {bias_squared}')
        st.write(f'Variance: {variance}')

        progressBar.progress(100, "DONE!!!...")

        # Predicting Value 
        predicted_y = best_gbr.predict(x_values)
        predicted_y = np.round(predicted_y, 2)
        pred_package = predicted_y[0]

        st.markdown("<h3 style='text-align: center;'>Your expected annual package is:</h3>", unsafe_allow_html=True)

        # Display the predicted value in a larger font
        st.markdown(f"<h1 style='text-align: center; color: green;'>{pred_package} LPA</h1>", unsafe_allow_html=True)

        # Display the warning message in smaller text
        st.markdown(f"<p style='text-align: center; color: red; font-size: 14px;'>The values calculated can/may contain an error margin of ± {mae:.2f}.</p>", unsafe_allow_html=True)