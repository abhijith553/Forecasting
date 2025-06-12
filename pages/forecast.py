from dash import html, dcc, dash_table, ctx, Output, Input
import io
import plotly.express as px
import json
import pandas as pd
import plotly.graph_objects as go
import base64
import plotly.graph_objs as go
import os
import requests
from io import StringIO

df = pd.read_csv(StringIO(requests.get("https://drive.google.com/file/d/1cNzCXXzy_jfKd1Ip3CVhs4WB8RZtaJp6/view?usp=drive_link").text))

layout = html.Div([
    html.Div([
        html.H3("Available models", style={
            'fontFamily': 'Monospace',
            'fontSize': '18px',
            'color': '#000','display': 'flex',
                    'justifyContent': 'center',
                    'alignItems': 'center',
                    'marginBottom': '25px'
        }),
        html.Div([
            html.Button("xg-boost", id="btn-xg", n_clicks=0, style={'fontSize': '24px', 'fontFamily': 'Monospace', 'backgroundColor': '#fff', 'border': '2.5px solid #000', 'borderRadius': '7px'}),
            html.Button("lstm", id="btn-lstm", n_clicks=0, style={'fontSize': '24px', 'fontFamily': 'Monospace', 'backgroundColor': '#fff', 'border': '2.5px solid #000', 'borderRadius': '7px'}),
            html.Button("arima", id="btn-arima", n_clicks=0, style={'fontSize': '24px', 'fontFamily': 'Monospace', 'backgroundColor': '#fff', 'border': '2.5px solid #000', 'borderRadius': '7px'}),
            html.Button("linear regression", id="btn-regr", n_clicks=0, style={'fontSize': '24px', 'fontFamily': 'Monospace', 'backgroundColor': '#fff', 'border': '2.5px solid #000', 'borderRadius': '7px'})
        ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '15px'})
    ], style={'width': '20%', 'padding': '30px'}),

    html.Div(id='forecast-output-area', style={'width': '70%', 'padding': '50px'})
], style={'display': 'flex', 'justifyContent': 'space-between','backgroundColor': '#fff'})

def load_arima_output():
    df_results = pd.read_csv(StringIO(requests.get("https://drive.google.com/file/d/1AXHXgJYK-SYVoUUZdCp3kY6SbXCQsYrY/view?usp=drive_link")))
    future_df = pd.read_csv(StringIO(requests.get("https://drive.google.com/file/d/1_nfM-qqcThbZmJs50Oq1TzBUs5OCG1ZV/view?usp=drive_link")))
    df_results['Date'] = pd.to_datetime(df_results['Date'])
    future_df['Date'] = pd.to_datetime(future_df['Date'])

    with open("model_outputs/arima_metrics.json", "r") as f:
        metrics = json.load(f)

    df_results.set_index('Date', inplace=True)
    weekly_df = df_results.resample('W').mean().dropna().reset_index()

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=weekly_df['Date'], y=weekly_df['Actual'],
        mode='lines+markers', name='Actual (Weekly Avg)', line=dict(color='blue', width=2)
    ))

    fig1.add_trace(go.Scatter(
        x=weekly_df['Date'], y=weekly_df['ARIMA_Prediction'],
        mode='lines+markers', name='Predicted (Weekly Avg)', line=dict(color='red', dash='dash', width=2)
    ))

    if 'Upper_CI' in df_results.columns and 'Lower_CI' in df_results.columns:
        weekly_df['Upper_CI'] = df_results['Upper_CI'].resample('W').mean().dropna().values
        weekly_df['Lower_CI'] = df_results['Lower_CI'].resample('W').mean().dropna().values

        fig1.add_trace(go.Scatter(
            x=weekly_df['Date'], y=weekly_df['Upper_CI'],
            mode='lines', name='Upper Bound', line=dict(width=0), showlegend=False
        ))
        fig1.add_trace(go.Scatter(
            x=weekly_df['Date'], y=weekly_df['Lower_CI'],
            mode='lines', name='Confidence Interval',
            fill='tonexty', fillcolor='rgba(255,165,0,0.2)', line=dict(width=0), showlegend=True
        ))

    forecast_start = future_df['Date'].min()
    fig1.add_vline(
        x=forecast_start.to_pydatetime(),
        line=dict(color="gray", dash="dash")
    )
    fig1.add_annotation(
        x=forecast_start.to_pydatetime(),
        y=weekly_df['Actual'].max(),
        text="Forecast Start",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40,
        bgcolor="lightgray"
    )

    fig1.update_layout(
        title="ARIMA: Actual vs Predicted Weekly Sales (Aggregated)",
        xaxis_title="Date",
        yaxis_title="Weekly Sales",
        template='plotly_white',
        xaxis=dict(tickformat="%b %Y")
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=future_df['Date'], y=future_df['Forecast'],
        mode='lines+markers', name='4-Week Forecast',
        line=dict(color='green', width=2)
    ))
    fig2.update_layout(
        title="ARIMA: 4-Week Forecast",
        xaxis_title="Date",
        yaxis_title="Weekly Sales",
        template='plotly_white',
        xaxis=dict(tickformat="%b %d")
    )

    return html.Div([
        html.H4("ARIMA Forecast Results", style={'fontFamily': 'Monospace'}),
        html.P(f"MAE: {metrics['MAE']:.3f}", style={'fontSize': '16px', 'fontFamily': 'Monospace'}),
        html.P(f"RMSE: {metrics['RMSE']:.3f}", style={'fontSize': '16px', 'fontFamily': 'Monospace'}),
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2),
    ])



def load_lstm_output():
    df_results = pd.read_csv(StringIO(requests.get("https://drive.google.com/file/d/1rgRP9oYakuEOqcT0e4VqY7P9yBYGKWKy/view?usp=drive_link").text))
    df_forecast = pd.read_csv(StringIO(requests.get("https://drive.google.com/file/d/1Uzt8UEEmZdxOBnWzJuDyFdaCYUjEeaZp/view?usp=drive_link").text))

    with open(StringIO(requests.get("https://drive.google.com/file/d/1eKJ9xPEGEJWgfZ6RGDXGidQzw7MrcQvy/view?usp=drive_link")), "r") as f:
        metrics = json.load(f)
    df_results['Date'] = pd.to_datetime(df_results['Date'])

    df_grouped = df_results.groupby('Date').agg({'Actual': 'mean', 'Predicted': 'mean'}).reset_index()
    fig1 = go.Figure()

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=df_grouped['Date'], y=df_grouped['Actual'],
        mode='lines', name='Actual',
        line=dict(color='blue', width=2)
    ))

    fig1.add_trace(go.Scatter(
        x=df_grouped['Date'], y=df_grouped['Predicted'],
        mode='lines', name='Predicted',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig1.add_trace(go.Scatter(
        x=df_grouped['Date'].tolist() + df_grouped['Date'][::-1].tolist(),
        y=df_grouped['Actual'].tolist() + df_grouped['Predicted'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    fig1.update_layout(
        title='LSTM: Actual vs Predicted Weekly Sales (Averaged)',
        xaxis_title='Date',
        yaxis_title='Average Weekly Sales',
        template='plotly_white'
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_forecast['Date'], y=df_forecast['Forecast'], mode='lines+markers', name='4-Week Forecast'))
    fig2.update_layout(title='LSTM: 4-Week Forecast', xaxis_title='Date', yaxis_title='Sales', template='plotly_white')

    df_loss = pd.read_csv(StringIO(requests.get("https://drive.google.com/file/d/1605qJ-KoWZRquuAR50o2Y1MzAGeijIWo/view?usp=drive_link").text))
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=df_loss['loss'], mode='lines+markers', name='Training Loss'))
    fig3.update_layout(title='LSTM: Training Loss Over Epochs', xaxis_title='Epoch', yaxis_title='Loss', template = 'plotly_white')

    return html.Div([
        html.H4("LSTM Forecast Results", style={'fontFamily': 'Monospace'}),
        html.P(f"MAE: {metrics['MAE']}", style={'fontSize': '16px', 'fontFamily': 'Monospace'}),
        html.P(f"RMSE: {metrics['RMSE']}", style={'fontSize': '16px', 'fontFamily': 'Monospace'}),
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2),
        dcc.Graph(figure=fig3)
    ])

def load_xgboost_output():

    df_results = pd.read_csv(StringIO(requests.get("https://drive.google.com/file/d/14fXSTGaPgDdIZur2UdRe9mKz8SuZo39J/view?usp=drive_link").text))

    with open(StringIO(requests.get("https://drive.google.com/file/d/1QYaFtQRBx9GXHqlmrru-Mi0a3RqZMIU3/view?usp=drive_link")), "r") as f:
        metrics = json.load(f)

    importance_df = pd.read_csv(StringIO(requests.get("https://drive.google.com/file/d/1QRRTfRlJYbGvF9J5c4WO_beAYmgUIArJ/view?usp=drive_link").text))

    df_results = df_results.groupby('Date').mean().reset_index()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_results['Date'], y=df_results['Actual'],
        mode='lines+markers', name='Actual', line=dict(color='blue')
    ))
    fig1.add_trace(go.Scatter(
        x=df_results['Date'], y=df_results['XGBoost_Prediction'],
        mode='lines+markers', name='XGBoost Prediction', line=dict(color='red')
    ))
    fig1.update_layout(
        title="Aggregated Weekly Sales: Actual vs XGBoost Prediction",
        xaxis_title="Date",
        yaxis_title="Weekly Sales",
        height=500,
        width=1000,
        template='plotly_white'
    )

    fig2 = px.bar(
        importance_df.head(10),
        x='Importance', y='Feature',
        orientation='h',
        title="Top 10 Feature Importances (XGBoost)"
    )
    fig2.update_layout(yaxis=dict(categoryorder='total ascending'), template='plotly_white')

    return html.Div([
        html.H4("XGBoost Forecast Results", style={'fontFamily': 'Monospace'}),
        html.P(f"MAE: {metrics['MAE']}", style={'fontSize': '16px', 'fontFamily': 'Monospace'}),
        html.P(f"RMSE: {metrics['RMSE']}", style={'fontSize': '16px', 'fontFamily': 'Monospace'}),
        dcc.Graph(figure=fig1),
        html.Br(),
        html.H4("XGBoost Feature Importance", style={'fontFamily': 'Monospace'}),
        dcc.Graph(figure=fig2)
    ])


def load_linear_regression_output():
    df_results = pd.read_csv(StringIO(requests.get("https://drive.google.com/file/d/1U6-6-48pi27GvRtOYfPUa0M1IGMqz1rz/view?usp=drive_link").text))
    df_results["Date"] = pd.to_datetime(df_results["Date"])
    df_results.sort_values("Date", inplace=True)

    with open(StringIO(requests.get("https://drive.google.com/file/d/1EQJti0NJltfEpFOA90KTuYKWXebSMQng/view?usp=drive_link")) "r") as f:
        metrics = json.load(f)

    coeff_df = pd.read_csv(StringIO(requests.get("https://drive.google.com/file/d/1UKeL3aJUZ3o8pnaRkvp3aSFyh6TxQ1SA/view?usp=drive_link").text))
    coeff_df_sorted = coeff_df.reindex(coeff_df["Coefficient"].abs().sort_values(ascending=False).index)

    df_results.set_index("Date", inplace=True)
    df_weekly = df_results.resample("W").mean().reset_index()
    df_weekly["Residuals"] = df_weekly["Actual"] - df_weekly["Predicted"]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_weekly["Date"], y=df_weekly["Actual"],
        mode="lines", name="Actual", line=dict(color='blue', width=2)
    ))
    fig1.add_trace(go.Scatter(
        x=df_weekly["Date"], y=df_weekly["Predicted"],
        mode="lines", name="Predicted", line=dict(color='orange', dash="dot", width=2)
    ))
    fig1.update_layout(
        title="Actual vs Predicted Weekly Sales (Linear Regression)",
        xaxis_title="Date",
        yaxis_title="Weekly Sales",
        template="plotly_white"
    )

    df_weekly["Residual_Trend"] = df_weekly["Residuals"].rolling(window=4, min_periods=1).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_weekly["Date"], y=df_weekly["Residuals"],
        mode="markers", name="Residuals", marker=dict(color="red", size=6, opacity=0.5)
    ))
    fig2.add_trace(go.Scatter(
        x=df_weekly["Date"], y=df_weekly["Residual_Trend"],
        mode="lines", name="Residual Trend", line=dict(color="black", dash="dash")
    ))
    fig2.update_layout(
        title="Residual Plot (Linear Regression)",
        xaxis_title="Date",
        yaxis_title="Residuals",
        template="plotly_white"
    )

    coeff_df_sorted["Impact"] = coeff_df_sorted["Coefficient"].apply(lambda x: "Positive" if x > 0 else "Negative")
    fig3 = px.bar(
        coeff_df_sorted,
        x="Coefficient", y="Feature",
        orientation="h",
        title="Feature Effects on Weekly Sales (Linear Regression)",
        color="Impact",
        color_discrete_map={"Positive": "green", "Negative": "crimson"},
        text="Coefficient"
    )
    fig3.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        template='plotly_white',
        showlegend=False
    )
    fig3.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    return html.Div([
        html.H4("Linear Regression Forecast Results", style={'fontFamily': 'Monospace'}),
        html.P(f"MAE: {metrics['MAE']:.3f}", style={'fontSize': '16px', 'fontFamily': 'Monospace'}),
        html.P(f"RMSE: {metrics['RMSE']:.3f}", style={'fontSize': '16px', 'fontFamily': 'Monospace'}),
        html.P(f"R² Score: {metrics['R2']:.3f}", style={'fontSize': '16px', 'fontFamily': 'Monospace'}),

        html.Hr(),
        dcc.Graph(figure=fig1),
        html.Hr(),
        dcc.Graph(figure=fig2),
        html.Hr(),
        dcc.Graph(figure=fig3),
    ])

def register_callbacks(app):
    @app.callback(
        Output('forecast-output-area', 'children'),
        Input('btn-xg', 'n_clicks'),
        Input('btn-lstm', 'n_clicks'),
        Input('btn-arima', 'n_clicks'),
        Input('btn-regr', 'n_clicks')
    )
    def update_output(n1, n2, n3, n4):
        triggered_id = ctx.triggered_id
        if triggered_id == "btn-xg":
            return load_xgboost_output()
        elif triggered_id == "btn-lstm":
            return load_lstm_output()
        elif triggered_id == "btn-arima":
            return load_arima_output()
        elif triggered_id == "btn-regr":
            return load_linear_regression_output()
        return html.Div("No model currently selected available(4)", style = {'fontSize': '18px', 'fontFamily': 'Monospace',})