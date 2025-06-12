import dash
from dash import html
import dash_bootstrap_components as dbc


defaultStyle = {'display' : 'flex', 'justifyContent' : 'center', 'marginBottom' : '1px', 'fontFamily' : 'Monospace', 'fontSize' : '22px'}

layout = html.Div([
    html.H3("This dashboard provides insights and forecast for walmart sales open dataset", style = {'display': 'flex',
                                                                                                     'justifyContent': 'center',
                                                                                                     'alignItems': 'center',
                                                                                                     'marginBottom': '25px',
                                                                                                     'fontFamily': 'Monospace',
                                                                                                     'fontSize': '24px'
                                                                                                     }),
    html.H4("Core technologies used in this project are : Python, Tensorflow, Plotly dash", style = {'display': 'flex',
                                                                                                     'justifyContent': 'center',
                                                                                                     'alignItems': 'center',
                                                                                                     'marginBottom': '25px',
                                                                                                     'fontFamily': 'Monospace',
                                                                                                     'fontSize': '24px'
                                                                                                     }),

    html.Div([
        html.H4("Source code", style={'textAlign': 'center', 'fontFamily': 'Monospace', 'fontSize': '18px'}),
        html.Div(
            html.A("github", href="https://github.com/abhijith553/Forecasting", target="_blank", style={
                                                                                'color': 'white',
                                                                                'backgroundColor':'#000',
                                                                                'padding': '10px 20px',
                                                                                'borderRadius': '5px',
                                                                                'fontFamily': 'Monospace',
                                                                                'fontSize': '18px'
                                                                                }),
            style={'display': 'flex', 'justifyContent': 'center'}
        )
    ], style={
                'position': 'fixed',
                'bottom': '70px',
                'left': '50%',
                'transform': 'translateX(-50%)',
                'textAlign': 'center'
            }),
    html.Div([
        html.Img(src='/assets/image1.png', style={
                                                    'width': '100px', 
                                                    'margin': '10px'
        }),
        html.Img(src='/assets/image2.png', style={
                                                    'width': '200px',
                                                    'margin': '10px'
        })
    ], style={
                'display': 'flex',
                'justifyContent': 'center',
                'alignItems': 'center',
                'marginBottom': '30px'
    })
])