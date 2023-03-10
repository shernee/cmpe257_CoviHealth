import dash
from dash import html
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from dash import dcc
from pipeline import *
import plotly.graph_objs as go

app = dash.Dash(__name__, external_stylesheets=['assets/style.css'])
min_Value = 65
min_value = 26
max_value = 88

high_slider_label = "Infection Rate High:"
high_slider =dcc.RangeSlider(
        id='high-slider',
        min=0,
        max=100,
        step=10,
        value=[0, 0],
        marks={i: f'{i}%' for i in range(0, 110, 10)},
        className='slider-style',
    )
high_slider_input = dbc.Col(
    html.Div(children=[
        html.Label(high_slider_label, htmlFor=high_slider_label),
        high_slider
    ])
)

mid_slider_label = "Infection Rate Mid:" 
mid_slider= dcc.RangeSlider(
        id='mid-slider',
        min=0,
        max=100,
        step=10,
        value=[0, 0],
        marks={i: f'{i}%' for i in range(0, 110, 10)},
        className='slider-style',       
)
mid_slider_input = dbc.Col(
    html.Div(children=[
        html.Label(mid_slider_label, htmlFor=mid_slider_label) ,
        mid_slider
    ])
)

dataset_dropdown_label = "Select a Dataset to see the class distribution and the top features"
dataset_dropdown = dcc.Dropdown(
    id='dataset-dropdown',
    options=[
        {'label': 'Dataset1', 'value': 'viz_1.csv'},
        {'label': 'Dataset1 & Dataset2', 'value': 'viz_2.csv'},
        {'label': 'Dataset1 & Dataset2 & Dataset3', 'value': 'viz_3.csv'}
    ],
    className='dropdown-style',
    value='viz_1.csv'
)


input_row_1 = html.Div(children=[
    html.Label(dataset_dropdown_label, htmlFor=dataset_dropdown_label),
    dataset_dropdown
])
input_row_2 = html.Div(children=[
    dbc.Row([high_slider_input, mid_slider_input])
])

# output_row_1 = dcc.Graph(id='class-balance-graph')
output_row_1 = html.Div(children=[
    dcc.Graph(id='hist-plot')
], style={'width': '50%'})



app.layout = html.Div([
    input_row_1,
    input_row_2,
    output_row_1
])


@app.callback(
    dash.dependencies.Output('hist-plot', 'figure'),
    # dash.dependencies.Output('class-balance-graph', 'children'),
    [dash.dependencies.Input('high-slider', 'value'),
     dash.dependencies.Input('mid-slider', 'value'),
     dash.dependencies.Input('dataset-dropdown', 'value')])
def update_output(infection_rate_high, infection_rate_mid, selected_dataset):
    range1_min = round((infection_rate_high[0] / 100) * (max_value - min_Value) + min_Value, 2)
    range1_max = round((infection_rate_high[1] / 100) * (max_value - min_Value) + min_Value, 2)
    range2_min = round((infection_rate_mid[0] / 100) * (max_value - min_value) + min_value, 2)
    range2_max = round((infection_rate_mid[1] / 100) * (max_value - min_value) + min_value, 2)
    X_res, y_res = oversample(range1_max, range2_max, selected_dataset)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=y_res, nbinsx=3))
    fig.update_layout(title='Class Distribution')
    
    return fig
    # return f'infection_rate_high: {range1_min}% to {range1_max}% \ninfection_rate_mid: {range2_min}% to {range2_max}%'

if __name__ == "__main__":
    app.run_server(debug=True)