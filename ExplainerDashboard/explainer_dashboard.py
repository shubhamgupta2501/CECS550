from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard, ExplainerHub
from sklearn.model_selection import train_test_split
import pandas as pd
from dash import html
from dash import dcc
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from explainerdashboard.datasets import titanic_survive, feature_descriptions

dataset = pd.read_csv("heart.csv")
predictors = dataset.drop("target",axis=1)
target = dataset["target"]

X_train,X_test,y_train,y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

model = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)
explainer1 = ClassifierExplainer(model, X_test, y_test, descriptions=feature_descriptions, labels=['Unhealthy', 'Healthy'])

model2 = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)
explainer2 = ClassifierExplainer(model2, X_test, y_test, descriptions=feature_descriptions, labels=['Unhealthy', 'Healthy'])

model3 = GradientBoostingClassifier(max_depth=2, n_estimators=3, learning_rate=1.0).fit(X_train, y_train)
explainer3 = ClassifierExplainer(model3, X_test, y_test, descriptions=feature_descriptions, labels=['Unhealthy', 'Healthy'])

db1 = ExplainerDashboard(explainer1, title="Random Forest", name="db1")
db2 = ExplainerDashboard(explainer2, title="Logistic Regression", name="db2")
db3 = ExplainerDashboard(explainer3, title="Light GBM", name="db3")

hub = ExplainerHub([db1, db2, db3], title="Heart Disease Prediction",
description="A dashboards for predective model comparision ")
hub.run(port=8089)