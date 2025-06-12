from dash import html, dcc, dash_table, ctx, Output, Input
import pandas as pd
import io
import plotly.express as px
import requests
from io import StringIO


df = pd.read_csv(StringIO(requests.get("https://drive.google.com/uc?export=download&id=1cNzCXXzy_jfKd1Ip3CVhs4WB8RZtaJp6").text))

layout = html.Div([
    html.Div([
        html.H3("Applied EDA methods", style={
            'fontFamily': 'Monospace',
            'fontSize': '24px',
            'fontWeight': 'bold',
            'color': '#000'
        }),
        html.Div([
            html.Button("Instances", id="btn-instances", n_clicks=0, style={'fontSize': '24px', 'fontFamily': 'Monospace', 'backgroundColor': '#fff', 'border': '2.5px solid #000', 'borderRadius': '7px'}),
            html.Button("Info", id="btn-info", n_clicks=0, style={'fontSize': '24px', 'fontFamily': 'Monospace', 'backgroundColor': '#fff', 'border': '2.5px solid #000', 'borderRadius': '7px'}),
            html.Button("Correlation matrix", id="corr-mat", n_clicks=0, style={'fontSize': '24px', 'fontFamily': 'Monospace', 'backgroundColor': '#fff', 'border': '2.5px solid #000', 'borderRadius': '7px'}),
            html.Button("Change over time", id="change-time", n_clicks=0, style={'fontSize': '24px', 'fontFamily': 'Monospace', 'backgroundColor': '#fff', 'border': '2.5px solid #000', 'borderRadius': '7px'}),
            html.Button("Shape", id="shape", n_clicks=0, style={'fontSize': '24px', 'fontFamily': 'Monospace', 'backgroundColor': '#fff', 'border': '2.5px solid #000', 'borderRadius': '7px'}),
            html.Button("Datatypes", id="dtypes", n_clicks=0, style={'fontSize': '24px', 'fontFamily': 'Monospace', 'backgroundColor': '#fff', 'border': '2.5px solid #000', 'borderRadius': '7px'})
        ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '15px'})
    ], style={'width': '25%', 'padding': '30px'}),

    html.Div(id='eda-output-area', style={'width': '70%', 'padding': '50px'})
], style={'display': 'flex', 'justifyContent': 'space-between', 'backgroundColor': '#fff'})

def register_callbacks(app):
    @app.callback(
        Output('eda-output-area', 'children'),
        Input('btn-instances', 'n_clicks'),
        Input('btn-info', 'n_clicks'),
        Input('shape', 'n_clicks'),
        Input('dtypes', 'n_clicks'),
        Input('corr-mat', 'n_clicks'),
        Input('change-time', 'n_clicks')
    )
    def update_output(n1, n2, n3, n4, n5, n6):
        triggered_id = ctx.triggered_id

        if triggered_id == "btn-instances":
            df_head = df.head(10)
            return dash_table.DataTable(
                data=df_head.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'}
            )
        elif triggered_id == "btn-info":
            buffer = io.StringIO()
            df.info(buf=buffer)
            return html.Pre(buffer.getvalue(), style={'whiteSpace': 'pre-wrap'})
        elif triggered_id == "shape":
            return html.Div(f"shape of the dataframe: {df.shape}")
        elif triggered_id == "dtypes":
            return html.Pre(df.dtypes.to_string(), style={'whiteSpace': 'pre-wrap'})
        elif triggered_id == "corr-mat":
            corr = df.select_dtypes(include=['number']).corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale='IceFire', title="Correlation Heatmap")
            fig.update_layout(margin=dict(l=30, r=20, t=50, b=30))
            return dcc.Graph(figure=fig)
        elif triggered_id == "change-time":
            change = ["week", "month", "year"]
            graphs = []
            for j in change:
                fig = px.area(df, x = j, y = "Weekly_Sales")
                fig.update_layout(margin=dict(l=30, r=20, t=50, b=30))
                graphs.append(dcc.Graph(figure = fig))
            return html.Div(graphs)
        return html.Div("Select an option from the menu on left")