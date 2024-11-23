import streamlit as st
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import warnings

# Load the trained XGBoost model
with open('model/XGB_regressor.pkl', 'rb') as model_file:
    model_xgb = pickle.load(model_file)

# Load and preprocess the dataset
dataset = pd.read_excel("data/HousePricePrediction.xlsx")
dataset.drop(['Id'], axis=1, inplace=True)
dataset['SalePrice'].fillna(dataset['SalePrice'].mean(), inplace=True)
new_dataset = dataset.dropna()

# One-hot encode categorical features
s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
OH_encoder = OneHotEncoder(sparse_output=False)
OH_cols = OH_encoder.fit_transform(new_dataset[object_cols])
OH_cols_df = pd.DataFrame(OH_cols, columns=OH_encoder.get_feature_names_out(object_cols))
OH_cols_df.index = new_dataset.index
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols_df], axis=1)

selected_features = [
     'LotArea', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'SalePrice'
]
df_selected = df_final[selected_features]

# Split the dataset for training and testing
X = df_selected.drop(['SalePrice'], axis=1)
Y = df_selected['SalePrice']
X_train, X_valid, Y_train, Y_valid = train_test_split(
	X, Y, train_size=0.8, test_size=0.2, random_state=0)

# Suppress warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
# Create a Streamlit app
st.set_page_config(page_title="House Price Prediction App")

st.write("""
# House Price Prediction App
""")

st.sidebar.header('Specify Input Parameters')

def user_input_features():
    LotArea = st.sidebar.slider('LotArea', float(X['LotArea'].min()), float(X['LotArea'].max()), float(X['LotArea'].mean()))
    YearBuilt = st.sidebar.slider('YearBuilt', int(X['YearBuilt'].min()), int(X['YearBuilt'].max()), int(X['YearBuilt'].mean()))
    YearRemodAdd = st.sidebar.slider('YearRemodAdd', int(X['YearRemodAdd'].min()), int(X['YearRemodAdd'].max()), int(X['YearRemodAdd'].mean()))
    TotalBsmtSF = st.sidebar.slider('TotalBsmtSF', float(X['TotalBsmtSF'].min()), float(X['TotalBsmtSF'].max()), float(X['TotalBsmtSF'].mean()))

    data = {
            'LotArea': LotArea,
            'YearBuilt': YearBuilt,
            'YearRemodAdd': YearRemodAdd,
            'TotalBsmtSF': TotalBsmtSF}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.header('Specified Input parameters')
st.write(df)
st.write('---')

prediction = model_xgb.predict(df)

st.header('Predicted Price:')
st.write(prediction)
st.write('---')

explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig)
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig)

