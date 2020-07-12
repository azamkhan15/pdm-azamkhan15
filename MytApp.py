# Core Pkgs
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 



import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

X1 = pd.DataFrame()
y1 = pd.DataFrame()
def main():
	
	#if st.Button('Open a Dataset")
	#activities = ["Open Data Set","Data Visualizations","Prediction","About"]	
	#choice = st.sidebar.selectbox("Select Activities",activities)
		"""Predictive Maintenance System """
		st.write("Click below to open a dataset")
		if st.checkbox("Open Data Set"):
				st.subheader("Click browse file option in the center to upload a data set")
				data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
				if data is not None:
					df = pd.read_csv(data)
					X1=df
					y1=df
					st.dataframe(X1.head(2))
					all_columns = df.columns.to_list()
				#if st.checkbox("Show Detail"):
					st.write('Numbers of Rows and Col found',df.shape)
					#st.write(df.shape)
				#st.write(all_columns)
				#st.write(df.describe())
			
			#X = dff.drop('TTFord', axis=1)
			#y = dff['TTFord']
			
				
					if st.checkbox("Select features"):
						st.subheader("Click [Choose an option] below to select Features ")
						selected_columns = st.multiselect("Select features",all_columns)
						X1 = df[selected_columns]
				
				#st.dataframe(X)
				#st.write(X.iloc[:,-1].value_counts())
				
					if st.checkbox("Select Target"):
								st.subheader("Click [Choose an option] below to select  Target varible ")
								selected_columns2 = st.multiselect("Select Target",all_columns)
								y1=df[selected_columns2]
				#st.dataframe(y)
				#st.write(y.iloc[:,-1].value_counts())	
				
				#"""Semi Automated ML App with Streamlit """
				#s.write(X.shape)
				#print('x shape : ',X.shape)
				#print('y shape : ',y.shape)
				#"""Semi Automated ML App with Streamlit """
				#s.write(y.shape)


	
	#if choice == '':
		#st.subheader("Exploratory Data Analysis2")
		#data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		#if data is not None:
			#df = pd.read_csv(data)
			
			#X = df.iloc[:,0:-1] 
			#Y = df.iloc[:,-1]
			# Model Building
			#X = df.iloc[:,0:-1] 
			#Y = df.iloc[:,-1]
								if st.checkbox("Prediction"):
									st.subheader("Please scroll down to see the results")
									X = X1
									y=y1
									seed = 7
									models = []
									models.append(('LR', LogisticRegression()))
									models.append(('LDA', LinearDiscriminantAnalysis()))
									models.append(('KNN', KNeighborsClassifier()))
									models.append(('CART', DecisionTreeClassifier()))
									models.append(('NB', GaussianNB()))
									models.append(('SVM', SVC()))
									model_names = []
									model_mean = []
									model_std = []
									all_models = []
									scoring = 'accuracy'
									for name, model in models:
										kfold = model_selection.KFold(n_splits=10, random_state=seed)
										cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
										model_names.append(name)
										model_mean.append(cv_results.mean())
										model_std.append(cv_results.std())
										accuracy_results = {"model name":name,"model_accuracy":cv_results.mean(),"standard deviation":cv_results.std()}
										all_models.append(accuracy_results)
				
									st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Algo","Mean of Accuracy","Std"]))	


				
if __name__ == '__main__':
		main()