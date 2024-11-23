# House-Price-Predictor
This project predicts house prices based on various features using machine learning. The application is built using XGBoost regression model and deployed via Streamlit for an interactive user interface.

---

## Features
- **Machine Learning Model**: Utilizes the XGBoost regression algorithm for accurate predictions.
- **Streamlit Web App**: Provides an interactive interface for users to input parameters and view predictions.
- **Feature Importance Visualization**: Displays SHAP values to explain model predictions.
- **Data Preprocessing**: Includes data cleaning, feature selection, and encoding for categorical variables.


## File Structure
```
House Price Prediction/
│
├── App/
│   └── Streamlit_App.py                # Streamlit application
│
├── data/
│   └── HousePricePrediction.xlsx       # Dataset used for training and prediction
│
├── model/
│   └── XGB_regressor.pkl               # Trained XGBoost model
│  
├── House_Price_Prediction.ipynb        # Jupyter notebook for model training
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```


## How to Run the Project

### Prerequisites
1. **Python 3.7+** must be installed on your system.
2. Install the required Python libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
1. Open your terminal and navigate to the project directory:
   ```bash
   cd path/to/project
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run App/Streamlit_App.py
   ```
3. Open the URL provided in the terminal (usually `http://localhost:8501`) in your web browser.


## Usage
1. Use the sliders in the app to specify input parameters:
   - Lot Area
   - Year Built
   - Year Remodel/Add
   - Total Basement Area
2. View the **predicted house price** based on the inputs.
3. Explore **feature importance** visualizations to understand the model's decision-making.


## Dependencies
- **Streamlit**: For deploying the web app.
- **Pandas**: For data manipulation and preprocessing.
- **SHAP**: For model explainability.
- **Seaborn & Matplotlib**: For visualization.
- **Scikit-learn**: For data splitting and metrics.
- **XGBoost**: For training the regression model.
- **OpenPyXL**: For handling Excel files.

To install all dependencies, use:
```bash
pip install -r requirements.txt
```


## Future Improvements
- Add additional model options (e.g., Random Forest, Linear Regression).
- Containerize the app using **Docker**.
- Deploy the app on a cloud platform like **Streamlit Cloud**, **Heroku**, or **AWS**.
- Enhance the dataset with more features to improve predictions.


## License
This project is licensed under the MIT License. See the LICENSE file for details.
