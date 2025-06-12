import dash
from dash import html

defaultStyle = {'display' : 'flex', 'justifyContent' : 'center', 'marginBottom' : '1px', 'fontFamily' : 'Monospace', 'fontSize' : '22px'}

layout = html.Div([
    html.P("Demand forecasting is a process where businesses", style = defaultStyle),
    html.P("predicts the future demand for their products or services", style = defaultStyle),
    html.P("it involves using various techniques including", style = defaultStyle),
    html.P("statistical modelling and historical data", style = defaultStyle),
    html.P("to estimate how much products or services consumers ", style = defaultStyle),
    html.P("likely purchase over a specific period", style = defaultStyle),
])

steps = [
    {
        "title": "What is Demand Forecasting?",
        "content": [
            "Demand forecasting is a process where businesses predicts the future demand for their products or services it involves using various techniques including statistical modelling and historical data to estimate how much products or services consumers likely purchase over a specific period"
        ]
    },
     {
         "title": "Working of Linear Regression",
         "content": [
             "Data loading : Loads a cleaned retail sales dataset and removes non-predictive fields Weekly_Sales and Date to isolate features and the target variable",
             "",
             "Train-test split : (with Date tracking) Splits the data into training and test sets 80 and 20, while preserving Date to retain temporal context",
             "",
             "Model Training : Linear Regression model is fit using scikit-learn on training features and target values",
             "",
             "Prediction : Generates predictions, computes residuals, and evaluates performance using mean abs error, root mean square error, r2 score",
             "",
             "Result : Exports a CSV with actual vs predicted values, sorted by date, saves evaluation metrics as a JSON file and outputs a csv of feature coefficients for interpretability."

         ]
     },
     {
         "title": "Working of XGBoost",
         "content": [
             "Data loading : Loads a cleaned dataset and discards Weekly_Sales (target) and Date. Splits features and target into train/test sets while preserving Date",
             "",
             "Model config  :  Initializes an XGBRegressor with: n_estimators=100: number of boosting rounds learning_rate=0.1: step size shrinkage to control overfitting max_depth=5: depth of individual trees, balancing performance and complexity ",
             "",
             "Training : Fits the model on training data, learning non-linear relationships via gradient boosting.",
             "",
             "Prediction and evaluation : Predicts weekly sales on the test set. Computes MAE and RMSE, which measure average error and penalize large deviations respectively. Saves metrics to a JSON file for easy downstream tracking or reporting.",
             "",
             "Time-Based Output Logging Combines Date, actual, and predicted values into a tidy DataFrame for chronological validation. Sorted CSV export supports easy plotting or audit trails.",
             "",
             "Feature importance extraction : Outputs a ranked list of feature importances — a key strength of XGBoost — helping you identify what drives weekly sales. Great for interpretability or feature pruning in future models"
         ]
     },
     {
         "title": "Working of LSTM model",
         "content": [
             "Data Preprocessing : Loads retail weekly sales and extracts the target variable. Applies MinMaxScaler to normalize sales data between 0 and 1 — essential for LSTM convergence",
             "",
             "Sequence Creation:  Uses a sliding window (seq_len = 10) to build input sequences and their corresponding next-step labels Transforms the 1D series into a supervised learning structure: past 10 weeks → predict week 11.",
             "",
             "Train-Test Split:  Splits the sequences chronologically to avoid data leakage: 80% for training, 20% for testing",
             "",
             "Model Architecture : LSTM(50 units) with ReLU activation captures temporal dependencies.  Dense(1) output layer returns the next predicted value. Compiled with Adam optimizer and MSE loss — a common, effective setup for time series regression.",
             "",
             "Training & Evaluation : Trains silently over 20 epochs, using validation to monitor performance. Predicts and then inverse-scales results back to the original range for interpretability.  Calculates MAE and RMSE, offering error metrics in the original scale",
             "",
             "Results & Logging :  Saves predictions with timestamps, providing traceability for time-based trends. Exports metrics and a prediction timeline for downstream analysis or visualization."

         ]
     },
     {
         "title": "Working of SARIMA",
         "content": [
             "Data loading : Model begins by reading 5000 rows of data, the date columsn is parsed to datetime format",
             "",
             "Stationary testing : The Augmented Dickey Fuller test checks whether the data is stationary",
             "",
             "Automatic parameter selection : auto_arima() is used to find the optimal p, d, q parameters which minimize the forecasting error, eliminating manual trial and error",
             "",
             "Model training : The dataset is split into 80 percent training and 20 percent testing, the arima model is trained on this traning data using optimal parameters",
             "",
             "Forecasting : Model forecasts for test period and 4 future weeks, allowing comparison",
             "",
             "Performance evaluation : mean abs error and root mean square error are saved in json for later analysis",
             "",
             "Output generation : Three csv files are generated for actual vs predicted test results, future week forecast and model evaluation"
         ]
     }
]

columns = [
    html.Div([
        html.H4(step["title"]),
        html.P([elem for line in step["content"] for elem in (line, html.Br())])
    ], style={
        'flex': '1',
        'minWidth': '200px',
        'margin': '10px',
        'padding': '10px',
        #'border': '2px solid #000',
        #'borderRadius': '7px',
        'backgroundColor': '#fff',
        'fontFamily' : 'Monospace',
        'fontSize' : '18px'
    }) for step in steps
]

layout = html.Div([
    html.H2("know more", style={'textAlign': 'center', 'fontFamily' : 'Monospace', 'fontSize' : '24px'}),
    html.Div(columns, style={
        'display': 'flex',
        'flexDirection': 'row',
        'overflowX': 'auto',
        'padding': '20px',
        'fontFamily' : 'Verdana',
        'fontSize' : '18px'
    })
])
