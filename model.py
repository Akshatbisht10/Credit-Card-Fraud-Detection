# importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st 

# loading data
data = pd.read_csv(r"C:\Users\AKSHAT_BISHT\OneDrive\Desktop\Project\creditcard.csv")
data.Class.value_counts()

# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# balancing the legitimate data
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)
legit_sample.shape

# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluating model performance
accuracy = accuracy_score(model.predict(X_test), y_test)
accuracy

# create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the details of transaction:")

# create input fields for user to enter feature values
input_data = st.text_input('Input All features')
input_data_sp = input_data.split(',')

# create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    # get input feature values
    features = np.array(input_data_sp, dtype=np.float64)
    # make prediction
    prediction = model.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")
        
# 406,-2.312226542,1.951992011,-1.609850732,3.997905588,-0.522187865,-1.426545319,-2.537387306,1.391657248,-2.770089277,-2.772272145,3.202033207,-2.899907388,-0.595221881,-4.289253782,0.38972412,-1.14074718,-2.830055675,-0.016822468,0.416955705,0.126910559,0.517232371,-0.035049369,-0.465211076,0.320198199,0.044519167,0.177839798,0.261145003,-0.143275875,0	
# 15,1.4929359769862,-1.02934573189487,0.45479473374366,-1.43802587991702,-1.55543410136344,-0.720961147043557,-1.08066413038614,-0.0531271179483221,-1.9786815953872,1.63807603690446,1.07754241162743,-0.63204651464934,-0.41695716661602,0.0520105153724404,-0.0429789228232019,-0.166432496451972,0.304241418614353,0.554432499062278,0.0542295152184719,-0.387910172646258,-0.177649846438814,-0.175073809074822,0.0400022190621329,0.295813862676508,0.33293059939425,-0.220384850672322,0.0222984359135846,0.00760225559997897,5