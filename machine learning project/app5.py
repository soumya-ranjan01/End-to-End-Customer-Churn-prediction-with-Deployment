import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
import pickle
from flask import Flask,render_template,redirect,request,url_for

app5=Flask(__name__)
model= pickle.load(open("model.sav","rb"))
df=pd.read_csv("C:/Users/SOUMYA/New folder (3)/myenv/machine learning project/first_telc.csv")

@app5.route("/")
def homepage():
    return render_template("home.html")



@app5.route("/predict",methods=['POST','GET'])
def predict():
    if request.method=='POST':
        input_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])}
        
    
    input_df = pd.DataFrame([input_data])
    input_df["TotalCharges"]=input_df["TotalCharges"].replace(' ',np.nan)
    input_df.dropna(how='any',inplace=True)
    input_df["TotalCharges"]=input_df["TotalCharges"].astype("float")

    #new_df=pd.get_dummies(input_df)
    from sklearn.preprocessing import LabelEncoder
   

   # Perform label encoding for categorical features
    label_encoders = {}
    for column in input_df.columns:
        if input_df[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            input_df[column] = label_encoders[column].fit_transform(input_df[column])


    prediction = model.predict(input_df.tail(1))
    probability = model.predict_proba(input_df.tail(1))[:,1]


    if prediction==1:
        result="This customer is likely to be churned!!"

    else:
        result="This customer is likely to be not churned!!"

    return render_template("home.html",result=result)

if __name__=='__main__':
    app5.run(debug=True)
