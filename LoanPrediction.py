import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#Binarization of categories as scikit can only be implemented with inetgers  
df=pd.read_csv("train.csv")#importing csv file using pandas
df['Gender'].replace(['Male','Female'],[1,0],inplace=True)
df['Married'].replace(['Yes','No'],[1,0],inplace=True)
df['Education'].replace(['Graduate','Not Graduate'],[1,0],inplace=True)
df['Property_Area'].replace(['Urban','Rural','Semiurban'],[1,2,3],inplace=True)
df['Self_Employed'].replace(['Yes','No'],[1,0],inplace=True)
df['Dependents'].replace(['3+'],[3],inplace=True)

#filling the blank spaces with 0 so it doesn't affect the prediction 
df['Gender']=df['Gender'].fillna(0)
df['Married']=df['Married'].fillna(0)
df['Dependents']=df['Dependents'].fillna(0)
df['Self_Employed']=df['Self_Employed'].fillna(0)
df['LoanAmount']=df['LoanAmount'].fillna(0)
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(0)
df['Credit_History']=df['Credit_History'].fillna(0)

header=['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']

X=df[header]
y=df.Loan_Status
model = KNeighborsClassifier() #LogisticRegression() 
model.fit(X, y)#training the data

#test data
d=pd.read_csv("test.csv")
d['Gender'].replace(['Male','Female'],[1,0],inplace=True)
d['Married'].replace(['Yes','No'],[1,0],inplace=True)
d['Education'].replace(['Graduate','Not Graduate'],[1,0],inplace=True)
d['Property_Area'].replace(['Urban','Rural','Semiurban'],[1,2,3],inplace=True)
d['Self_Employed'].replace(['Yes','No'],[1,0],inplace=True)
d['Dependents'].replace(['3+'],[3],inplace=True)
d['Gender']=d['Gender'].fillna(0)
d['Married']=d['Married'].fillna(0)
d['Dependents']=d['Dependents'].fillna(0)
d['Self_Employed']=d['Self_Employed'].fillna(0)
d['LoanAmount']=d['LoanAmount'].fillna(0)
d['Loan_Amount_Term']=d['Loan_Amount_Term'].fillna(0)
d['Credit_History']=d['Credit_History'].fillna(0)

X_test=d[header]
y_pred=model.predict(X_test)
d['Loan_Status']=y_pred
print(d)

