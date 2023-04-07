import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import base64
import pickle

from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
warnings.filterwarnings("ignore")
st.write("""
# Bondora Classifier
### Here we go!
""")
Data=pd.read_csv("MIC.csv")
def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
#add_bg_from_local("back.jpg")        
im=Image.open("back2.jpg")
st.sidebar.image(im)
st.sidebar.header("Borrower Input data")
st.sidebar.write("*Kindly answer the following questions to check the eligibility of the applied loan's :*")
def InputData():
    
    selectL=st.sidebar.selectbox("Select language:",["English","Estonian","Finnish","German","Russian","Slovakian","Spanish","Other"])
    if(selectL=="English"): Language=0
    elif(selectL=="Estonian"): Language=1
    elif(selectL=="Finnish"): Language=2
    elif(selectL=="German"): Language=3
    elif(selectL=="Other"): Language=4
    elif(selectL=="Russian"): Language=5
    elif(selectL=="Slovakian"): Language=6
    else: Language =7

    selectCountry=selectL=st.sidebar.selectbox("Select Country:",["EE","ES","FI","SK"])
    if(selectCountry=="EE"): selectCountry=0
    elif(selectCountry=="ES"): selectCountry=1
    elif(selectCountry=="FI"): selectCountry=2
    else: selectCountry=3   
    appliedAmount=st.sidebar.number_input("Enter the amount applied for originally:")
    amount=st.sidebar.number_input("Enter the amount provided to borrower:")
    interest=st.sidebar.number_input("Enter the interest accepted in the application:")
    principalBalance=st.sidebar.number_input("Enter the principal balance for borrower:")
    prevLoans=st.sidebar.number_input("Enter the value of previous loans:")
    monthlyPayment=st.sidebar.number_input("Enter borrower's monthly payment:")
    principalPayments=st.sidebar.number_input("Enter the amount of principal payments made:")
    penaltypayments=st.sidebar.number_input("Enter the amount of repaid penalties:")
    penaltyBalance=st.sidebar.number_input("Enter the value of unpaid interest and penalties:")
    Mgr=st.sidebar.number_input("Enter the amount of investment offers made by Portfolio Managers:")
    rating=st.sidebar.selectbox("Enter the rating issued by the rating Model:",["A","AA","B","C","D", "E","F","HR", "Empty"])
    if(rating=="A"): Rating=0
    elif(rating=="AA"): Rating=1
    elif(rating=="B"): Rating=2
    elif(rating=="C"): Rating=3
    elif(rating=="D"): Rating=4
    elif(rating=="E"): Rating=5
    elif(rating=="Empty"): Rating=6
    elif(rating=="F"): Rating=7
    elif(rating=="HR"): Rating=8
    else: Rating =9
    data={"InterestAndPenaltyBalance":penaltyBalance,
         "PrincipalPaymentsMade":principalPayments,
         "PrincipalBalance":principalBalance,
         
         "InterestAndPenaltyPaymentsMade":penaltypayments,
         "Interest":interest,
         "Amount":amount,
         "AppliedAmount":appliedAmount,
        
         "MonthlyPayment":monthlyPayment,
         "Rating":Rating,
          "PreviousRepaymentsBeforeLoan":prevLoans,
          "BidsPortfolioManager":Mgr,
          "LanguageCode":Language,
          "Country": selectCountry
         }
    features=pd.DataFrame(data, index=[0])
    return features

df=InputData()
df=df
st.subheader("Borrower Input Parameters:")
st.write(df)
GB=GradientBoostingClassifier()

def Modeling():
    le = LabelEncoder()
    def EncodingCategoricals(columns ,df):
        for c in columns:
            df[c]=le.fit_transform(df[c])
    objects=Data.select_dtypes("object")
    EncodingCategoricals(Data.columns,Data)
    y=Data["DefaultLoan"]
    X=Data.drop("DefaultLoan", axis=1)
    sc=StandardScaler()
    X=sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=0)
    GB.fit(X_train,y_train)

st.subheader("Loan Status:")
def PredictAns():
        Modeling()
        y_predict=GB.predict(df)
        return y_predict
if( st.sidebar.button("Predict")):
    ans=PredictAns()
    if(ans==0):
        st.write("Congratulations! The selected loan has been admitted :)")
    else:
        st.write("Unfortunately, The selected loan has been denied :(")


