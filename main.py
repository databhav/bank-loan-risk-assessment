# Importing all the required libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import plotly.express as px

# setting page view to landscape instead of vertical
st.set_page_config(layout="wide")

# loading dataset
df = pd.read_csv("/BCC.csv")
df.head()

# creating df2 to work with
df2 = df

# creating tenure risk and tenure risk encoded columns
df2['tenure_risk'] = np.where(df2['tenure']<=3, 'high risk',
                              np.where(df2['tenure']<=6, 'medium risk', 'low risk'))
df2['tenure_risk_enc'] = np.where(df2['tenure_risk']=='high risk',1,
                              np.where(df2['tenure_risk']=='medium risk', 2, 3))

# creating credit risk and credit risk encoded columns
df2['credit_risk'] = np.where(df2['credit_score']<=579, 'high risk', 
                                np.where(df2['credit_score']<=749,'medium risk','low risk'))
df2['credit_risk_enc'] = np.where(df2['credit_risk']=='high risk',1,
                              np.where(df2['credit_risk']=='medium risk', 2, 3))
df2.head()

# creating estimated salary risk and estimated salary risk columns
df2['salary_risk'] = np.where(df2['estimated_salary']<=30000,'high risk',
                              np.where(df2['estimated_salary']<=70000, 'medium risk', 'low risk'))

df2['salary_risk_enc'] = np.where(df2['salary_risk']=='high risk',1,
                              np.where(df2['salary_risk']=='medium risk', 2, 3))

# creating a total risk column based on the other risk columns
df2['total_risk_enc'] = df2['credit_risk_enc']+df2['tenure_risk_enc']+df2['salary_risk_enc']

df2['total_risk'] = np.where(df2['total_risk_enc']<=4,'high risk',
                             np.where(df2['total_risk_enc']<=6, 'medium risk', 'low risk'))

# mapping colors and risks to numerical values to plot later
color_mapping = {'low risk': 3, 'medium risk': 2, 'high risk': 1}
colors = df2['total_risk'].map(color_mapping)

## MODEL TRAINING
# selecting features and target values
X = df2[['credit_score','tenure','estimated_salary']]
Y = df2['total_risk_enc']
# splitting into training and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3, random_state=42, shuffle=False)
#training the model
model = DecisionTreeClassifier()
model.fit(X_train,Y_train)


## PAGE SHOWN ATTRIBUTES
st.title('Bank Loan Risk Assessment Analysis & Model:')

# Dividing page into two columns
with st.container():
    col1, col2 = st.columns((2,1)) # selecting the ratios to be 2:1 between columns

# Working with col1    
with col1:
    # Plotting the scatter plot between estimated salary and credit score
    fig = px.scatter(df2, x='estimated_salary', y='credit_score',color=colors,color_continuous_scale='blugrn',opacity=0.8)
    fig.update_layout(
    title='Salary vs Credit Score',
    title_font=dict(size=24),
    width=900,  # Set the width of the plot
    height=600  # Set the height of the plot
)
    # Display the interactive plot in Streamlit
    st.plotly_chart(fig)   
    # adding some description
    st.markdown("##### The following plot gives us the visualization on the insights of the data. You can either select User_id and see their information or you can use a trained model and input tenure, salary and credit score to calculate the risk.")
  
# Working with col2
with col2:
    # creating a search query to search customer id
    search_query = st.text_input('Search by customer ID')
    if search_query.strip().isdigit():
        # Perform the search operation based on the user's input
        search_result = df2[df2['customer_id'] == int(search_query)]
        st.table(search_result[['customer_id','gender','country','tenure','credit_score','estimated_salary']].T)
        st.write(f"customer score: **{search_result['total_risk_enc'].iloc[0]}/10**: **{search_result['total_risk'].iloc[0]}**")
        # making a progress bar to show the risk level
        progress_bar_1 = st.progress(search_result['total_risk_enc'].iloc[0]/10)
    
    st.subheader('OR')
    
    # creating inputs for customer salary, credit score and tenure
    customer_salary = st.text_input('Enter customer salary: ')
    customer_credit_score = st.text_input('Enter Credit score: ')
    customer_tenure = st.text_input('Enter Customer Tenure: ')
    # if all inputs given then plotting progress bar
    if customer_credit_score and customer_tenure and customer_salary:
        pred = pd.DataFrame([[customer_credit_score, customer_tenure, customer_salary]], columns=['credit_score', 'tenure', 'estimated_salary'])
        pred[['credit_score', 'tenure', 'estimated_salary']] = pred[['credit_score', 'tenure', 'estimated_salary']].astype(float)
        Y_pred = model.predict(pred)
        Y_pred = Y_pred.astype(float)
        # selecting conditions to show the risk level based on predicted value
        if Y_pred[0] <=3:
            pred_risk = 'high risk'
        elif Y_pred[0] <=6:
            pred_risk = 'medium risk'
        else:
            pred_risk = 'low_risk'
            
        st.write(f"customer score:   **{Y_pred[0]}/10** : **{pred_risk}**")
        progress_bar = st.progress(Y_pred[0]/10)
    
    # plotting dataframe to select customer ids later while testing
    st.subheader('Raw Data')
    st.dataframe(df2[['customer_id','gender','tenure','credit_score','estimated_salary']],height=300)

    


