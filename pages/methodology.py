from dash import html

steps = [
    {
        "title": "Data collection",
        "content": [
            "The dataset used in this project is pulled from Kaggle.",
            "",
            "It contains information on 45 Walmart stores located in different regions.",
            "",
            "Store - the store number",
            "",
            "Dept - the department number",
            "",
            "Date - the week of the sales record",
            "",
            "Weekly_Sales - sales for the given department in the given week",
            "",
            "IsHoliday - whether the week included a holiday",
            "",
            "Temperature - average temperature in the region during that week",
            "",
            "Fuel_Price - cost of fuel in the region",
            "",
            "MarkDown1-5 - promotional markdowns",
            "",
            "CPI - consumer price index",
            "",
            "Unemploylment - rate of unemployment",
        ]
    },
    {
        "title": "Data Preprocessing",
        "content": [
            "Dropped missing values per feature",
            "",
            "Dropped unnecessary instances",
            "",
            "Null values were replaced",
            "",
            "Features with continous values were detected",
            "",
            "Outliers were handled using log transformation",
            "",
            "Non numeric features were encoded"
        ]
    },
    {
        "title": "Exploratory Data Analysis",
        "content": [
            "Visualized trends, seasonality, and anomalies",
            "",
            "Used heatmap do display correlation matrix",
            "",
            "Used histograms to detect continous values",
            "",
            "Used boxplot to detect outlier presence",
            "",
            "Used area plot to visualize change over time"
        ]
    },
    {
        "title": "Model Selection",
        "content": [
            "Multiple models are used in this project",
            "",
            "ARIMA, LSTM, XGBoost, Linear Regression",
            "",
            "The plots for respective models were displayed",
        ]
    },
    {
        "title": "Model Evaluation",
        "content": [
            "Evaluated with MAE, RMSE",
            "",
            "For ARIMA MAE measures the abs difference between predicted and actual, RMSE shows how much large forecasting mistake occur with time series",
            "",
            "For LSTM MAE helps assess overall accuray, RMSE allows the model to capture spikes or abrupt changes",
        ]
    },
    {
        "title": "Forecasting and visualization",
        "content": [
            "Plotly graphs were used to visualize the forecasts",
            "",
            "Actual vs forecasted predictions were compared using graphs"
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
    html.H2("methodology", style={'textAlign': 'center', 'fontFamily' : 'Monospace', 'fontSize' : '24px'}),
    html.Div(columns, style={
        'display': 'flex',
        'flexDirection': 'row',
        'overflowX': 'auto',
        'padding': '20px',
        'fontFamily' : 'Verdana',
        'fontSize' : '18px'
    })
])
