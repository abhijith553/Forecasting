from dash import Dash, dcc, html, Input, Output
from pages import eda, forecast, about_the_project, methodology, more_insights

app = Dash(__name__, suppress_callback_exceptions=True)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #fff !important;
                margin: 0;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.title = "Retail Sales Dashboard"

link_style = {
    'fontSize': '14px',
    'fontFamily': 'Monospace',
    'padding': '6px 10px',
    'marginBottom': '8px',
    'paddingBotom' : '80px',
    'textDecoration': 'none',
    'border': '2px solid #000',
    'backgroundColor': '#000',
    'color' : '#fff'
}

app.layout = html.Div([
    dcc.Location(id='url'),
    html.Div([
        html.H3("retail store market trends and predictions", style={
                                                                        'display': 'flex',
                                                                        'justifyContent': 'flex-start',
                                                                        'alignItems': 'flex-start',
                                                                        'width': '100%',
                                                                        'fontSize': '24px',
                                                                        'fontFamily': 'Monospace',
                                                                        'color' : '#fff'

                                                                    }
),
        html.Details([
            html.Summary("Pages", style={
                                            'cursor': 'pointer',
                                            'fontSize': '21px',
                                            'fontFamily': 'Consolas',
                                            'backgroundColor': '#000',
                                            'color': '#fff',
                                            'border': '1.5px solid #fff',
                                            'padding': '8px 12px',
                                            'listStyleType': 'none',
                                            'position': 'fixed',       
                                            'top': '20px',            
                                            'right': '30px',           
                                            'zIndex': '1000',
                                            'borderRadius': '7px'
            }),
            html.Div([
                dcc.Link("Home", href='/', style=link_style),
                dcc.Link("EDA", href='/eda', style=link_style),
                dcc.Link("More Insights", href='/more_insights', style=link_style),
                dcc.Link("Forecast", href='/forecast', style=link_style),
                dcc.Link("Methodology", href='/methodology', style=link_style),
                dcc.Link("About", href='/about_the_project', style=link_style),
            ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'flexWrap': 'wrap',
                        'padding': '10px',
                        'border': '3px solid #000',
                        'backgroundColor': '#000',
                        'position': 'absolute',
                        'right': '0.1px',
                        'top': '100px',
                        'zIndex': '1000',
                        'borderRadius': '7px'
            })
        ], style={'position': 'relative', 'backgroundColor': '#000'})
    ], style={
                'display': 'flex',
                'justifyContent': 'flex-end',
                'padding': '2px 20px',
                'backgroundColor': '#000',
                'color' : '#fff'
    }),

    html.Div(id='page-content')
], style = {'backgroundColor': '#fff'})

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def render_page(pathname):
    if pathname == '/eda':
        return eda.layout
    elif pathname == '/forecast':
        return forecast.layout
    elif pathname == '/methodology':
          return methodology.layout
    elif pathname == '/more_insights':
          return more_insights.layout
    elif pathname == '/about_the_project':
        return about_the_project.layout
    else:
        return html.Div("welcome to the dash app", style={
                                                            'fontFamily': 'Monospace',
                                                            'fontSize' : '24px',
                                                            'display' : 'flex',
                                                            'justifyContent': 'center',
                                                            'alignItems':'center',
                                                            'padding' : '80px'
                                                        }
                        )


eda.register_callbacks(app)
forecast.register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)