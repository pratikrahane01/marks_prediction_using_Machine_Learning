import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("students_db.csv")
print(df.head())

print(df.info())
print(df.describe())
print(df.isnull())
#encoading

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
df["Gender_Encoaded"]=le.fit_transform(df["Gender"])
df["Internet_Access_Encoaded"]=le.fit_transform(df["Internet_Access"])


print(df[["Gender_Encoaded",'Internet_Access_Encoaded']].head())



education_order=[["School","Diploma","Graduate"]]

from sklearn.preprocessing import OrdinalEncoder

oe=OrdinalEncoder(categories=education_order)

df["Parental_Education_encoaded"]=oe.fit_transform(df[["Parental_Education"]])

print(df["Parental_Education_encoaded"].head())

#visualizing trend

df.drop(columns=["Gender","Internet_Access","Parental_Education"],inplace=True)
sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
plt.show()

# train-test splitt

x=df.drop("Score",axis=1) # all column except score
y=df["Score"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print(X_train.shape) 
print(X_test.shape)  
                     
                    
#model training
from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(X_train,y_train)

print(model.coef_)
print(pd.Series(model.coef_, index= x.columns))

#prediction

hours = float(input("Enter study hours: "))
sleep = float(input("Enter sleep hours: "))
attendance = float(input("Enter attendance: "))

gender = input("Enter gender (Male/Female): ")
internet = input("Internet access (Yes/No): ")
education = input("Parental education (School/Diploma/Graduate): ")

#encoading

if gender == "Male":
    gender_val = 1
else:
    gender_val = 0

if internet == "Yes":
    internet_val = 1
else:
    internet_val = 0

if education == "School":
    edu_val = 0
elif education == "Diploma":
    edu_val = 1
else:
    edu_val = 2

new_data = [[
    hours,
    sleep,
    attendance,
    gender_val,
    internet_val,
    edu_val
]]
y_pred = model.predict(X_test)
prediction=model.predict(new_data)[0]
print(f"Predicted Score: {prediction}")

# Model Evaluation

from sklearn.metrics import mean_squared_error, r2_score


mse = mean_squared_error(y_test,y_pred)
#lower is better


r2 = r2_score(y_test,y_pred)
#closer to 1 is better 

print("MSE:", mse)
print("R2 Score:", r2)