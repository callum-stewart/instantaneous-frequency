
import numpy as np
import seaborn as sns
import colorednoise as cn

# dash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
from plotly import subplots

from AdvEMDpy import AdvEMDpy, emd_hilbert

time = np.linspace(0, 10 * np.pi, 1001)
time_series = np.zeros_like(time)
knots = np.linspace(0, 10 * np.pi, 101)

sns.set(style='darkgrid')

# Dash App - start

app = dash.Dash(__name__)

colors = {
    'background': '#101921',
    'text': '#dc1fff'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[

        html.H1('Time Series Dash Application', style={'text-align': 'center', 'color': colors['text']}),

        html.Div(children=[

            html.Button(id='execute_button', children='Execute')

        ],

            style={'display': 'block', 'horizontal-align': 'centre', 'margin-left': '47.5%'}

        ),

        html.Div(children=[

            html.H2('Trend', style={'text-align': 'center', 'color': colors['text']}),

            dcc.Dropdown(
                id='trend',
                options=[{'label': 'No Trend', 'value': False},
                         {'label': 'Linear', 'value': 'linear'},
                         {'label': 'Quadratic', 'value': 'quadratic'},
                         {'label': 'Exponential', 'value': 'exponential'}],
                value=False
            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3(u'\u03B1', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.Div(children=[

                dcc.Input(id="alpha",
                          type='number',
                          value=0.1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3(u'\u03B2', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.Div(children=[

                dcc.Input(id="beta",
                          type='number',
                          value=0.1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3(u'\u03B3', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.Div(children=[

                dcc.Input(id="gamma",
                          type='number',
                          value=0.1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H2('Additions', style={'text-align': 'center', 'color': colors['text']}),

            dcc.Dropdown(
                id='noise_bool',
                options=[{'label': 'No Noise', 'value': True},
                         {'label': 'Violet Noise', 'value': 'violet noise'},
                         {'label': 'Blue Noise', 'value': 'blue noise'},
                         {'label': 'White Noise', 'value': 'white noise'},
                         {'label': 'Pink Noise', 'value': 'pink noise'},
                         {'label': 'Brown Noise', 'value': 'brown noise'}],
                value=True
            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Noise Mean:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="noise_mean",
                          type='number',
                          value=0.0)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Noise SD:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="noise_sd",
                          type='number',
                          value=1.0)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            dcc.Graph(
                id='time_series_graph'
            )

        ],
            style={'display': 'inline-block', 'width': '40%', 'margin-left': '5%', 'margin-right': '5%',
                   'margin-bottom': '5%', 'vertical-align': 'top'}),

        html.Div(children=[

            html.H2('Fourier Transform', style={'text-align': 'center', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            dcc.Graph(
                id='FT'
            ),

            html.H2('Short-Time Fourier Transform', style={'text-align': 'center', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Which IMFs:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Dropdown(
                    id='which_imfs',
                    options=[{'label': 'All', 'value': 'all'},
                             {'label': 'IMF 1', 'value': [1]},
                             {'label': 'IMF 2', 'value': [2]},
                             {'label': 'IMF 3', 'value': [3]}],
                    value='all'
                )

            ]),

            html.H3('Maximum Frequency:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.Div(children=[

                dcc.Input(id="max_frequency",
                          type='number',
                          value=10.0)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            dcc.Graph(
                id='STFT'
            ),

            html.H2('Wavelet Transform', style={'text-align': 'center', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Recursive HT:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Dropdown(
                    id='recursive_ht',
                    options=[{'label': 'Smooth HT', 'value': True}, {'label': 'No Smooth HT', 'value': False}],
                    value=False
                )

            ]),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Recursive IF:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Dropdown(
                    id='recursive_if',
                    options=[{'label': 'Smooth IF', 'value': True}, {'label': 'No Smooth IF', 'value': False}],
                    value=False
                )

            ]),

            html.Div(children=[], style={'marginBottom': '1em'}),

            dcc.Graph(
                id='MWT'
            ),

            html.H2('Intrinsic Mode Functions', style={'text-align': 'center', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            dcc.Graph(
                id='EMD_IMF'
            ),

            html.H2('Hilbert Transform', style={'text-align': 'center', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            dcc.Graph(
                id='EMD_HT'
            ),

        ],
            style={'display': 'inline-block', 'width': '40%', 'margin-left': '5%', 'margin-right': '5%',
                   'margin-bottom': '5%', 'vertical-align': 'top'})

    ]
)

# connect plotly graphs to dash components

# pca graphs
@app.callback(
    output=[Output(component_id='time_series_graph', component_property='figure'),
            Output(component_id='FT', component_property='figure'),
            Output(component_id='STFT', component_property='figure'),
            Output(component_id='MWT', component_property='figure'),
            Output(component_id='EMD_IMF', component_property='figure'),
            Output(component_id='EMD_HT', component_property='figure')],

    inputs=[Input(component_id='execute_button', component_property='n_clicks')],

    state=[State(component_id='trend', component_property='value'),
           State(component_id='alpha', component_property='value'),
           State(component_id='beta', component_property='value'),
           State(component_id='gamma', component_property='value'),
           State(component_id='noise_bool', component_property='value'),
           State(component_id='noise_mean', component_property='value'),
           State(component_id='noise_sd', component_property='value'),
           State(component_id='max_frequency', component_property='value'),
           State(component_id='which_imfs', component_property='value')])

def update_output(n_click, trend, alpha, beta, gamma, noise_bool, noise_mean, noise_sd, max_frequency, which_imfs):

    time_series = np.zeros_like(time) + np.cos(time) + np.cos(5 * time)

    if not trend:
        pass
    else:
        if trend == 'linear':
            time_series += alpha * np.ones_like(time) + beta * time
        elif trend == 'quadratic':
            time_series += alpha * np.ones_like(time) + beta * time + gamma * time ** 2
        elif trend == 'exponential':
            time_series += alpha * np.exp(beta * time)

    if not noise_bool:
        pass
    else:
        if noise_bool == 'violet noise':
            time_series += cn.powerlaw_psd_gaussian(-2, len(time_series))
        elif noise_bool == 'blue noise':
            time_series += cn.powerlaw_psd_gaussian(-1, len(time_series))
        elif noise_bool == 'white noise':
            time_series += cn.powerlaw_psd_gaussian(0, len(time_series))
        elif noise_bool == 'pink noise':
            time_series += cn.powerlaw_psd_gaussian(1, len(time_series))
        elif noise_bool == 'brown noise':
            time_series += cn.powerlaw_psd_gaussian(2, len(time_series))

    fig_1 = px.line(x=time, y=time_series,
                    labels={"x": "Time", "y": "Displacement"})
    fig_1.update_layout(title={'text': "Time Series", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})

    emd = AdvEMDpy.EMD(time_series=time_series, time=time)
    emd = emd.empirical_mode_decomposition(knots=knots)

    imfs = emd[0]

    fig_2 = subplots.make_subplots(rows=int(np.shape(imfs)[0] - 1), cols=1)

    for imf in range(int(np.shape(imfs)[0] - 1)):

        fig_2.add_trace(go.Scatter(x=time, y=imfs[int(imf + 1), :], mode='lines',
                                   name='IMF {}'.format(str(int(imf + 1)))), row=int(imf + 1), col=1)

    hs = emd_hilbert.hilbert_spectrum(time=time, imf_storage=imfs, ht_storage=emd[1], if_storage=emd[2],
                                      max_frequency=float(max_frequency), plot=False, which_imfs=which_imfs)

    fig_3 = go.Figure(data=go.Heatmap(z=hs[2], y=hs[1][:, 0], x=hs[0][0, :]))

    fig_4 = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{
        "text": "No matching data found",
        "xref": "paper",
        "yref": "paper",
        "showarrow": False,
        "font": {"size": 28}}]}}

    fig_5 = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{
        "text": "No matching data found",
        "xref": "paper",
        "yref": "paper",
        "showarrow": False,
        "font": {"size": 28}}]}}

    fig_6 = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{
        "text": "No matching data found",
        "xref": "paper",
        "yref": "paper",
        "showarrow": False,
        "font": {"size": 28}}]}}

    return fig_1, fig_2, fig_3, fig_4, fig_5, fig_6


app.run_server(debug=True)

# Dash App - end