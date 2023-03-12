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
from scipy import stats
import plotly.figure_factory as ff

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
MIN_VALUE_HIGH = 65
MIN_VALUE_MID = 26
MAX_VALUE = 88
DATAFRAME = pd.read_csv('viz_1.csv')
NUM_FEATURES_MAP = {
    'viz_1.csv': 6,
    'viz_2.csv': 8,
    'viz_3.csv': 10
}
CLASS_LABELS = ['Low', 'Moderate', 'High']

# components for output row 1
high_slider_label = "Infection Rate High:"
high_slider = dcc.RangeSlider(
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

mid_slider_label = "Infection Rate Moderate :" 
mid_slider = dcc.RangeSlider(
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

dataset_dropdown_label = "Select a Dataset"
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
)

feature_dropdown_input = dbc.Col([
    html.Label(feature_dropdown_label, htmlFor=feature_dropdown_label),
    feature_dropdown
], className='component-style')
feature_distribution_graph = html.Div(children=[
    dcc.Graph(id='feature-distribution-curve')
])
output_row_2 = dbc.Row([
    dbc.Col(feature_dropdown_input, width=5),
    dbc.Col(feature_distribution_graph, width=5)
], className='row-style')


# componenets for output row 3
classifier_conclusion = html.Div(children=[
], id='classifier-output', className='component-style')

classifier_conf_matrix = html.Div(children=[
    dcc.Graph(id='confusion-matrix')
])

output_row_3 = dbc.Row([
    dbc.Col(classifier_conclusion, width=5),
    dbc.Col(classifier_conf_matrix, width=5)
], className='row-style')


# components for output row 4
classifier_geoplot = html.Div(children=[
    dcc.Graph(id='classification-geoplot')
])

output_row_4 = dbc.Row([
    dbc.Col(classifier_geoplot, width=5)
], className='row-style')


# page layout
app.layout = html.Div([
    output_row_1,
    output_row_2,
    output_row_3,
    output_row_4
])


# callbacks

# callback 1
@app.callback(
    [
        Output('class-balance-histogram', 'figure'),
        Output('feature-dropdown', 'options'),
        Output('feature-dropdown', 'value'),
        Output('classifier-output', 'children'),
        Output('confusion-matrix','figure'),
        Output('classification-geoplot', 'figure')
    ],
    [
        Input('high-slider', 'value'),
        Input('mid-slider', 'value'),
        Input('dataset-dropdown', 'value')
    ])

def update_output(infection_rate_high, infection_rate_mid, selected_dataset):
    range1_min = round((infection_rate_high[0] / 100) * (MAX_VALUE - MIN_VALUE_HIGH) + MIN_VALUE_HIGH, 2)
    range1_max = round((infection_rate_high[1] / 100) * (MAX_VALUE - MIN_VALUE_HIGH) + MIN_VALUE_HIGH, 2)
    range2_min = round((infection_rate_mid[0] / 100) * (MAX_VALUE - MIN_VALUE_MID) + MIN_VALUE_MID, 2)
    range2_max = round((infection_rate_mid[1] / 100) * (MAX_VALUE - MIN_VALUE_MID) + MIN_VALUE_MID, 2)

    df = pd.read_csv(selected_dataset)
    DATAFRAME = df
    n = NUM_FEATURES_MAP[selected_dataset]

    f1, conf_matrix, class_feature, top3_features, classifier, predictions = controller(range1_max, range2_max, df, n)
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=class_feature, nbinsx=3))
    hist_fig.update_layout(title='Class Distribution')

    heatmap_fig = px.imshow(
        conf_matrix, 
        text_auto=True, 
        labels=dict(x="Predicted label", y="True label"), 
        x=CLASS_LABELS, 
        y=CLASS_LABELS,
        title='Confusion matrix'
    )

    # Classification geoplot
    predictions['pred'] = predictions['pred'].replace({'Low':0, 'Moderate':1, 'High':2})
    data = dict(type = 'choropleth', 
            locations = predictions['area'], 
            z = predictions['pred'], 
            locationmode='country names',
            colorbar = dict(title='Infection levels', tickvals=[0, 1, 2], ticktext=['Low', 'Moderate', 'High']),)
    layout = dict(title = 'Infection level classification', height=700, width= 1000)
    geo_fig = go.Figure(data = [data], 
              layout = layout)

    return (
        hist_fig, 
        top3_features, 
        top3_features[0], 
        f'Best Classifier: {classifier} with F1- weighted score {f1:.2f}', 
        heatmap_fig,
        geo_fig
    )


# callback 2
@app.callback(
        Output('feature-distribution-curve', 'figure'),
        Input('feature-dropdown', 'value')
)
def update_output(selected_feature):
    x = DATAFRAME[selected_feature]
    fig = ff.create_distplot([x], ['Feature Distribution'], bin_size=0.5, curve_type='kde')
    fig.update_layout(title='Feature Distribution')
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
