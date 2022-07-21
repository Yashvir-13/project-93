import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv("penguin.csv")
df.dropna(inplace=True)
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})
df['sex'] = df['sex'].map({'Male':0,'Female':1})
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})
features=df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
x_train,x_test,y_train,y_test=train_test_split(features,df["label"],train_size=0.67,random_state=42)
svc_model = SVC(kernel = 'linear')
svc_model.fit(x_train, y_train)
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(x_train, y_train)
@st.cache(suppress_st_warning=True)
def prediction(model,x):
	y_pred=model.predict([x])
	y_pred=y_pred[0]
	if y_pred==0:
		st.write("The species's name is Adelie.")
	elif y_pred==1:
		st.write("The species's name is Chinstrap.")
	elif y_pred==2:
	    st.write("The species's name is Gentoo.")
st.title("Penguin's species prediction")	
island_=st.sidebar.selectbox("Select the island's name",('Biscoe','Dream','Torgersen'))
if island_=="Biscoe":
	island_=0
elif island_=="Dream":
	island_=1
elif island_=="Torgersen":
    island_=2	
model_=st.sidebar.selectbox("Select the model",("Support vector Machine","Logistic Regression","Random Forest Classifier"))  
sex_=st.sidebar.selectbox("Is the penguin a male or female",("Male","Female"))
if sex_=="Male":
	_sex=0
else:
	_sex=1	
bill_lengthmm=st.sidebar.slider("Select the bill length(mm)",df["bill_length_mm"].min(),df["bill_length_mm"].max(),1.0)
bill_depthmm=st.sidebar.slider("Select the bill depth(mm)",df["bill_depth_mm"].min(),df["bill_depth_mm"].max(),1.0)
flipper_lengthmm=st.sidebar.slider("Select the flipper length(mm)",df["flipper_length_mm"].min(),df["flipper_length_mm"].max(),1.0)
body_massg=st.sidebar.slider("Select the body mass(mm)",df["body_mass_g"].min(),df["body_mass_g"].max(),1.0)
if st.sidebar.button("Classify"):
    if model_=="Random Forest Classifier":
        ps=prediction(rf_clf,[island_,bill_lengthmm,bill_depthmm,flipper_lengthmm,body_massg,_sex])
        score=rf_clf.score(x_train,y_train)    

    elif model_=="Support vector Machine" :
        ps=prediction(svc_model,[island_,bill_lengthmm,bill_depthmm,flipper_lengthmm,body_massg,_sex])
        score=svc_model.score(x_train,y_train) 
    elif model_=="Logistic Regression":
        ps=prediction(lr_model,[island_,bill_lengthmm,bill_depthmm,flipper_lengthmm,body_massg,_sex])
        score=lr_model.score(x_train,y_train)
    
    st.write("The score of the model is",score)

	
		
