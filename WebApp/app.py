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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
min_Value = 65
min_value = 26
max_value = 88
DATAFRAME = pd.read_csv('viz_1.csv')

# components for output row 1
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
high_slider_input = dbc.Col([
    html.Label(high_slider_label, htmlFor=high_slider_label),
    high_slider
], className='component-style')

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
mid_slider_input = dbc.Col([
    html.Label(mid_slider_label, htmlFor=mid_slider_label) ,
    mid_slider
], className='component-style')

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
dataset_dropdown_input = dbc.Col([
    html.Label(dataset_dropdown_label, htmlFor=dataset_dropdown_label),
    dataset_dropdown
], className='component-style')

input_col = dbc.Col([
    dataset_dropdown_input,
    mid_slider_input,
    high_slider_input
])
class_distribution_graph = html.Div(children=[
    dcc.Graph(id='class-balance-histogram')
])
output_row_1 = dbc.Row([
        dbc.Col(input_col, width=5),
        dbc.Col(class_distribution_graph, width=5)
], className='row-style')


# components for output row 2
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

feature_dropdown_input = dbc.Col([
    html.Label(feature_dropdown_label, htmlFor=feature_dropdown_label),
    feature_dropdown
], className='component-style')
feature_distribution_graph = html.Div(children=[
    dcc.Graph(id='feature-distribution-histogram')
])
output_row_2 = dbc.Row([
    dbc.Col(feature_dropdown_input, width=5),
    dbc.Col(feature_distribution_graph, width=5)
], className='row-style')


# componenets for output row 3
classifier_conclusion = html.Div(children=[
    html.Div(id='classifier-output')
], className='component-style')

classifier_conf_matrix = html.Div(children=[
    dcc.Graph(id='confusion-matrix')
])

output_row_3 = dbc.Row([
    dbc.Col(classifier_conclusion, width=5),
    dbc.Col(classifier_conf_matrix, width=5)
], className='row-style')


# page layout
app.layout = html.Div([
    output_row_1,
    output_row_2,
    output_row_3,
])


# callbacks

# callback 1
@app.callback(
    [
        Output('class-balance-histogram', 'figure'),
        Output('feature-dropdown', 'options'),
        Output('feature-dropdown', 'value'),
        Output('classifier-output', 'children'),
        Output('confusion-matrix','figure')
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

    f1, conf_matrix, class_feature, top3_features, classifier = controller(range1_max, range2_max, df)
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=class_feature, nbinsx=3))
    hist_fig.update_layout(title='Class Distribution')

    classes = ['low', 'mid', 'high']
    heatmap_fig = px.imshow(
        conf_matrix, 
        text_auto=True, 
        labels=dict(x="predicted", y="actual"), 
        x=classes, 
        y=classes,
        title="Confusion matrix"
    )

    return hist_fig, top3_features, top3_features[0], f'Best Classifier: {classifier} with F1- weighted score {f1:.2f}', heatmap_fig


# callback 2
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
