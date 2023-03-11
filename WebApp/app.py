import dash
from dash import html
from dash import Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from dash import dcc
from pipeline import *
import plotly.graph_objs as go
# import dash_table
import seaborn as sns
import matplotlib.pyplot as plt

app = dash.Dash(__name__, external_stylesheets=['assets/style.css'])
min_Value = 65
min_value = 26
max_value = 88
DATAFRAME = pd.read_csv('viz_1.csv')

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


feature_dropdown_label = "Select feature to see its distribution"
feature_dropdown = dcc.Dropdown(
    id='feature-dropdown',
    options=[
        'protein_per_capita_milk_excluding_butter',
        'fat_per_capita_animal_fats',
        'calorie_per_capita_milk_excluding_butter',
    ],
    className='dropdown-style',
    # value='protein_per_capita_milk_excluding_butter'
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
    dcc.Graph(id='class-balance-histogram')
], style={'width': '50%'})

output_row_2 = html.Div(
    children=[
    html.Label(feature_dropdown_label, htmlFor=feature_dropdown_label),
    feature_dropdown
    ]
)

output_row_3 = html.Div(children=[
    dcc.Graph(id='feature-distribution-histogram')
], style={'width': '50%'})

confusion_matrix_label = "The confusion matrix of the selected dataset and features"
confusion_matrix_output = html.Div([
        html.Table(id='confusion-matrix', children=[
        ])
])


classifier_output = html.Div(children=[
    html.Div(id='output_classifier')
])

app.layout = html.Div([
    input_row_1,
    input_row_2,
    output_row_1,
    output_row_2,
    output_row_3,
    classifier_output,
    confusion_matrix_output
    
])


@app.callback(
    [
        Output('class-balance-histogram', 'figure'),
        Output('feature-dropdown', 'options'),
        Output('feature-dropdown', 'value'),
        # Output('confusion-matrix', 'data'),
        # Output('confusion-matrix', 'columns'),
        Output('output_classifier','children'),
        Output('confusion-matrix','children')
    ],
    [
        Input('high-slider', 'value'),
        Input('mid-slider', 'value'),
        Input('dataset-dropdown', 'value')
    ])
def update_output(infection_rate_high, infection_rate_mid, selected_dataset):
    range1_min = round((infection_rate_high[0] / 100) * (max_value - min_Value) + min_Value, 2)
    range1_max = round((infection_rate_high[1] / 100) * (max_value - min_Value) + min_Value, 2)
    range2_min = round((infection_rate_mid[0] / 100) * (max_value - min_value) + min_value, 2)
    range2_max = round((infection_rate_mid[1] / 100) * (max_value - min_value) + min_value, 2)

    df = pd.read_csv(selected_dataset)
    DATAFRAME = df

    f1, cf, y_counts, y, top3, classifier = controller(range1_max, range2_max, df)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=y, nbinsx=3))
    fig.update_layout(title='Class Distribution')
    
    children=[  html.Table([html.Tr([html.Th('Confusion Matrix')]),
                html.Tr([html.Td('True Class'), html.Td('Predicted Class'), html.Td('Count')]),
                html.Tr([html.Td('Class low'), html.Td('Class low'), html.Td(cf[0][0])]),
                html.Tr([html.Td('Class low'), html.Td('Class mid'), html.Td(cf[0][1])]),
                html.Tr([html.Td('Class low'), html.Td('Class high'), html.Td(cf[0][2])]),
                html.Tr([html.Td('Class mid'), html.Td('Class low'), html.Td(cf[1][0])]),
                html.Tr([html.Td('Class mid'), html.Td('Class mid'), html.Td(cf[1][1])]),
                html.Tr([html.Td('Class mid'), html.Td('Class high'), html.Td(cf[1][2])]),
                html.Tr([html.Td('Class high'), html.Td('Class low'), html.Td(cf[2][0])]),
                html.Tr([html.Td('Class high'), html.Td('Class mid'), html.Td(cf[2][1])]),
                html.Tr([html.Td('Class high'), html.Td('Class high'), html.Td(cf[2][2])])
            ])
        ]
    

    return fig, top3, top3[0], f'Best Classifier: {classifier} with F1- weighted score {f1}', children

    # return f'infection_rate_high: {range1_min}% to {range1_max}% \ninfection_rate_mid: {range2_min}% to {range2_max}%'

@app.callback(
        Output('feature-distribution-histogram', 'figure'),
        Input('feature-dropdown', 'value')
)
def update_output(selected_feature):
    x = DATAFRAME[selected_feature]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x))
    fig.update_layout(title='Feature Distribution')
    
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
