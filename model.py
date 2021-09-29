import pandas as pd

import warnings
warnings.simplefilter("ignore")

import pickle
from sklearn.linear_model import LinearRegression 


data = pd.read_csv(r'C:\Users\Prasanna\Desktop\model deployment\Admission_Predict.csv')
data.columns

X = data.drop('Chance of Admit ', axis = 1).copy()

y =  data['Chance of Admit '].copy()


model= LinearRegression()  
model.fit(X,y)
model.score(X,y)


pickle.dump(model,open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


