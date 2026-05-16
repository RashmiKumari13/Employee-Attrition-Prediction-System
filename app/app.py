import streamlit as st
import pickle
import pandas as pd
from catboost import CatBoostClassifier

# Load the trained model and unique values from the pickle file
with open('model_and_key_components.pkl', 'rb') as file:
    saved_components = pickle.load(file)

model = saved_components['model']
unique_values = saved_components['unique_values']

# Define the Streamlit app
def main():
    st.title("Employee Attrition Prediction App")
    st.sidebar.title("Model Settings")

    # Sidebar inputs
    with st.sidebar.beta_expander("View Unique Values"):
        st.write("Unique values for each feature:")
        for column, values in unique_values.items():
            st.write(f"- {column}: {values}")

    # Main content
    st.write("This app predicts employee attrition using a trained CatBoost model.")

    # Add inputs for user to input data
    age = st.slider("Age", min_value=18, max_value=70, value=30)
    distance_from_home = st.slider("Distance From Home", min_value=1, max_value=30, value=10)
    environment_satisfaction = st.slider("Environment Satisfaction", min_value=1, max_value=4, value=2)
    hourly_rate = st.slider("Hourly Rate", min_value=30, max_value=100, value=65)
    job_involvement = st.slider("Job Involvement", min_value=1, max_value=4, value=2)
    job_level = st.slider("Job Level", min_value=1, max_value=5, value=3)
    job_satisfaction = st.slider("Job Satisfaction", min_value=1, max_value=4, value=2)
    monthly_income = st.slider("Monthly Income", min_value=1000, max_value=20000, value=5000)
    num_companies_worked = st.slider("Number of Companies Worked", min_value=0, max_value=10, value=2)
    over_time = st.checkbox("Over Time")
    percent_salary_hike = st.slider("Percent Salary Hike", min_value=10, max_value=25, value=15)
    stock_option_level = st.slider("Stock Option Level", min_value=0, max_value=3, value=1)
    training_times_last_year = st.slider("Training Times Last Year", min_value=0, max_value=6, value=2)
    work_life_balance = st.slider("Work Life Balance", min_value=1, max_value=4, value=2)
    years_since_last_promotion = st.slider("Years Since Last Promotion", min_value=0, max_value=15, value=3)
    years_with_curr_manager = st.slider("Years With Current Manager", min_value=0, max_value=15, value=3)
    
    # Create a DataFrame to hold the user input data
    input_data = pd.DataFrame({
        'Age': [age],
        'DistanceFromHome': [distance_from_home],
        'EnvironmentSatisfaction': [environment_satisfaction],
        'HourlyRate': [hourly_rate],
        'JobInvolvement': [job_involvement],
        'JobLevel': [job_level],
        'JobSatisfaction': [job_satisfaction],
        'MonthlyIncome': [monthly_income],
        'NumCompaniesWorked': [num_companies_worked],
        'OverTime': [over_time],
        'PercentSalaryHike': [percent_salary_hike],
        'StockOptionLevel': [stock_option_level],
        'TrainingTimesLastYear': [training_times_last_year],
        'WorkLifeBalance': [work_life_balance],
        'YearsSinceLastPromotion': [years_since_last_promotion],
        'YearsWithCurrManager': [years_with_curr_manager]
    })
    
    # Suggestions for retaining the employee
    if predicted_to_leave:
        st.subheader("Suggestions for Retaining the Employee:")
        st.markdown("- Invest in orientation programs and career development for entry-level staff to contribute to higher retention.")
        st.markdown("- Implement mentorship programs and career development initiatives aimed at engaging and retaining younger employees.")
        st.markdown("- Offer robust training and development programs and regular promotion to foster career growth. This investment in skills and career advancement can contribute to higher job satisfaction and retention.")
        st.markdown("- Recognize the diverse needs of employees based on marital status and consider tailoring benefits or support programs accordingly.")
        st.markdown("- Consider offering benefits that cater to the unique needs of married, single, and divorced employees.")
        st.markdown("- Introduce or enhance policies that support work-life balance for employees with families.")
        st.markdown("- Recognize the unique challenges and opportunities within each department and tailor retention strategies accordingly.")

    # Make predictions
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]

    # Display prediction
    if prediction[0] == 0:
        st.success("Employee is predicted to stay (Attrition = No)")
    else:
        st.error("Employee is predicted to leave (Attrition = Yes)")

        # Offer recommendations for retaining the employee
        st.subheader("Suggestions for retaining the employee:")
        st.markdown("- Invest in orientation programs and career development for entry-level staff, which could contribute to higher retention.")
        st.markdown("- Implement mentorship programs and career development initiatives aimed at engaging and retaining younger employees.")
        st.markdown("- Offer robust training and development programs and regular promotions to foster career growth. This investment in skills and career advancement can contribute to higher job satisfaction and retention.")
        st.markdown("- Recognize the diverse needs of employees based on marital status and consider tailoring benefits or support programs accordingly.")
        st.markdown("- Consider offering benefits that cater to the unique needs of married, single, and divorced employees.")
        st.markdown("- Introduce or enhance policies that support work-life balance for employees with families.")
        st.markdown("- Recognize the unique challenges and opportunities within each department and tailor retention strategies accordingly.")

    # Display probability
    st.write(f"Probability of Attrition: {probability[0]:.2f}")

if __name__ == "__main__":
    main()
