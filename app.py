import pandas as pd
import joblib
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix

from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier, XGBRegressor

import shap
import lime
from lime import lime_tabular

from dotenv import load_dotenv
import os

load_dotenv()

ID = os.getenv('ID')
ID2 = os.getenv('ID2')

df = pd.read_csv("https://raw.githubusercontent.com/daniloui001/datasci_9_data_prep/main/HHS_COVID-19_Small_Area_Estimations_Survey_-_Updated_Bivalent_Vaccine_Audience_-_Wave_28.csv")

df1 = pd.read_csv("https://raw.githubusercontent.com/daniloui001/datasci_9_data_prep/main/Percentage_of_Hospitals_Reporting_Data_to_HHS_by_State.csv")

df.size
df1.size
df.dtypes

df.columns
df1.columns

X = df.drop(['FIPS','enthusiast_se', 'audience_se','no_outreach_se'], axis = 1)
y = df['no_outreach_est']
df['audience_est'] = df['audience_est'] * 1000 # to make numbers more than 0 otherwise won't work
df['audience_est'] = df['audience_est'].round(2)
df['enthusiast_est'] = df['enthusiast_est'] * 1000 # to make numbers more than 0 otherwise won't work
df['enthusiast_est'] = df['enthusiast_est'].round(2)
df.head(5)


scaler = StandardScaler()
scaler.fit(X)
pickle.dump(scaler, open(ID, 'scaler_100k.sav', 'wb'))

X_scaled = scaler.transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.7, random_state=42)

(X_train.shape, X_val.shape, X_test.shape)


dummy = DummyClassifier(strategy='most_frequent')

dummy.fit(X_train, y_train)
dummy_acc = dummy.score(X_val, y_val)




log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

y_val_pred = log_reg.predict(X_val)

log_reg_acc = log_reg.score(X_val, y_val)
log_reg_mse = mean_squared_error(y_val, y_val_pred)
log_reg_r2 = r2_score(y_val, y_val_pred)

print(confusion_matrix(y_val, y_val_pred))

print(classification_report(y_val, y_val_pred))

print('Baseline accuracy:', dummy_acc)
print('Logistic Regression accuracy:', log_reg_acc)
print('Logistic Regression MSE:', log_reg_mse)
print('Logistic Regression R2:', log_reg_r2)





xgboost = XGBClassifier()

xgboost.fit(X_train, y_train)

y_val_pred = xgboost.predict(X_val)

xgboost_acc = xgboost.score(X_val, y_val)
xgboost_mse = mean_squared_error(y_val, y_val_pred)
xgboost_r2 = r2_score(y_val, y_val_pred)

print(confusion_matrix(y_val, y_val_pred))

print(classification_report(y_val, y_val_pred))

print('Baseline accuracy:', dummy_acc)
print('Logistic Regression accuracy:', log_reg_acc)
print('Logistic Regression MSE:', log_reg_mse)
print('Logistic Regression R2:', log_reg_r2)
print('XGBoost accuracy:', xgboost_acc)
print('XGBoost MSE:', xgboost_mse)
print('XGBoost R2:', xgboost_r2)




param_grid = {

    'learning_rate': [0.1, 0.01, 0.001], 
    'n_estimators': [100, 200, 300], 
    'max_depth': [3, 4, 5],
}


xgboost = XGBClassifier()

grid_search = GridSearchCV(estimator=xgboost, param_grid=param_grid, cv=3, n_jobs=-1)

grid_search.fit(X_train, y_train)

y_val_pred = grid_search.predict(X_val)

df_results = pd.DataFrame({'actual': y_val, 'predicted': y_val_pred})
grid_search_acc = grid_search.score(X_val, y_val)
grid_search_mse = mean_squared_error(y_val, y_val_pred)
grid_search_r2 = r2_score(y_val, y_val_pred)

print(confusion_matrix(y_val, y_val_pred))

print(classification_report(y_val, y_val_pred))

print('Baseline accuracy:', dummy_acc)
print('Logistic Regression accuracy:', log_reg_acc)
print('XGBoost Model 1 accuracy:', xgboost_acc)
print('XGBoost Model 2 accuracy:', grid_search_acc)




print(grid_search.best_params_)
print(grid_search.best_score_)




xgboost = XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=200)

xgboost.fit(X_train, y_train)

y_test_pred = xgboost.predict(X_test)

xgboost_acc = xgboost.score(X_test, y_test)


explainer = shap.TreeExplainer(xgboost)
explanation = explainer(X_test)
shape_vlaues = explanation.values
shap.summary_plot(explanation, X_test, plot_type="bar")

explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=X.columns,
    class_names=['audience', 'outreach', 'enthusiast'],
    mode='classification',
)


exp = explainer.explain_instance(X_val, xgboost.predict_proba, num_features=9)
exp.save_to_file('observation_1.html')

pickle.dump(xgboost, open(ID2, 'xgboost_100k.sav', 'wb'))


## df1

X1 = df.drop(['State','Total', 'Percentage_reporting'], axis = 1)
y1 = df['Percentage_reporting']


X1_scaled = scaler.transform(X)

X1_train, X1_temp, y1_train, y1_temp = train_test_split(X1_scaled, y1, test_size1=0.5, random_state1=42)
X1_val, X1_test, y1_val, y1_test = train_test_split(X1_temp, y1_temp, test_size1=0.7, random_state1=42)

(X1_train.shape, X1_val.shape, X1_test.shape)


dummy = DummyClassifier(strategy='most_frequent')

dummy.fit(X1_train, y1_train)
dummy_acc = dummy.score(X1_val, y1_val)




log_reg = LogisticRegression()

log_reg.fit(X1_train, y1_train)

y1_val_pred = log_reg.predict(X1_val)

log_reg_acc = log_reg.score(X1_val, y1_val)
log_reg_mse = mean_squared_error(y1_val, y1_val_pred)
log_reg_r2 = r2_score(y1_val, y1_val_pred)

print(confusion_matrix(y1_val, y1_val_pred))

print(classification_report(y1_val, y1_val_pred))

print('Baseline accuracy:', dummy_acc)
print('Logistic Regression accuracy:', log_reg_acc)
print('Logistic Regression MSE:', log_reg_mse)
print('Logistic Regression R2:', log_reg_r2)





xgboost = XGBClassifier()

xgboost.fit(X1_train, y1_train)

y1_val_pred = xgboost.predict(X1_val)

xgboost_acc = xgboost.score(X1_val, y_val)
xgboost_mse = mean_squared_error(y1_val, y1_val_pred)
xgboost_r2 = r2_score(y1_val, y1_val_pred)

print(confusion_matrix(y1_val, y1_val_pred))

print(classification_report(y1_val, y1_val_pred))

print('Baseline accuracy:', dummy_acc)
print('Logistic Regression accuracy:', log_reg_acc)
print('Logistic Regression MSE:', log_reg_mse)
print('Logistic Regression R2:', log_reg_r2)
print('XGBoost accuracy:', xgboost_acc)
print('XGBoost MSE:', xgboost_mse)
print('XGBoost R2:', xgboost_r2)




param_grid = {

    'learning_rate': [0.1, 0.01, 0.001], 
    'n_estimators': [100, 200, 300], 
    'max_depth': [3, 4, 5],
}


xgboost = XGBClassifier()

grid_search = GridSearchCV(estimator=xgboost, param_grid=param_grid, cv=3, n_jobs=-1)

grid_search.fit(X1_train, y1_train)

y1_val_pred = grid_search.predict(X1_val)

df_results = pd.DataFrame({'actual': y1_val, 'predicted': y1_val_pred})
grid_search_acc = grid_search.score(X1_val, y1_val)
grid_search_mse = mean_squared_error(y1_val, y1_val_pred)
grid_search_r2 = r2_score(y1_val, y1_val_pred)

print(confusion_matrix(y1_val, y1_val_pred))

print(classification_report(y1_val, y1_val_pred))

print('Baseline accuracy:', dummy_acc)
print('Logistic Regression accuracy:', log_reg_acc)
print('XGBoost Model 1 accuracy:', xgboost_acc)
print('XGBoost Model 2 accuracy:', grid_search_acc)




print(grid_search.best_params_)
print(grid_search.best_score_)




xgboost = XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=200)

xgboost.fit(X_train, y_train)

y_test_pred = xgboost.predict(X1_test)

xgboost_acc = xgboost.score(X1_test, y1_test)


explainer = shap.TreeExplainer(xgboost)
explanation = explainer(X1_test)
shape_vlaues = explanation.values
shap.summary_plot(explanation, X_test, plot_type="bar")

explainer = lime_tabular.LimeTabularExplainer(
    training_data=X1_train,
    feature_names=X.columns,
    class_names=['audience', 'outreach', 'enthusiast'],
    mode='classification',
)


exp = explainer.explain_instance(X_val, xgboost.predict_proba, num_features=9)
exp.save_to_file('observation_2.html')

pickle.dump(xgboost, open(ID2, 'xgboost_100k.sav', 'wb'))