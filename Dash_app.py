
import numpy as np
import seaborn as sns
import colorednoise as cn
from matplotlib import mlab
from scipy import signal as sig
from scipy.fft import fft, fftfreq

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

# frequency modulation - top


def frequency(time_signal, frequency_period, min_frequency, max_frequency):

    end_time = time_signal[-1]  # last time point value
    # time over which frequency changes from minimum frequency to maximum frequency or vice versa
    freq_half_mod = frequency_period / 2  # half period
    time_points = len(time_signal)  # total number of time points
    time_diff = np.diff(time_signal)  # difference between time points

    increments = int(end_time / freq_half_mod)

    increment_length = int(max((time_points - 1) / increments, 1))  # minimum set to 1 as trivial

    new_time_signal = np.zeros_like(time_signal)
    new_time_signal[0] = time_signal[0]

    for i in range(1, time_points):
        if np.mod(i / increment_length, 1) == 0:
            temp_mod = 1
        else:
            temp_mod = np.mod(i / increment_length, 1)
        if ((i - 1) // increment_length) % 2 == 0:
            new_time_signal[i] = new_time_signal[i - 1] + (min_frequency + temp_mod * (max_frequency - min_frequency)) \
                                 * time_diff[i - 1]
        else:
            new_time_signal[i] = new_time_signal[i - 1] + (max_frequency - temp_mod * (max_frequency - min_frequency)) \
                                 * time_diff[i - 1]

    return new_time_signal

# frequency modulation - bottom

# Dash App - start

app = dash.Dash(__name__)
server = app.server

colors = {
    'background': '#101921',
    'text': '#03e1ff'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[

        html.H1('Time-Frequency Analysis Dash Application', style={'text-align': 'center', 'color': colors['text']}),

        html.Div(children=[

            html.Button(id='execute_button', children='Execute')

        ],

            style={'display': 'block', 'horizontal-align': 'centre', 'margin-left': '47.5%'}

        ),

        html.Div(children=[

            html.H1('Additions', style={'text-align': 'center', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

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

            html.H3('Linear Trend:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3(u'f(t) = \u03B1 + \u03B2t', style={'text-align': 'center', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Quadratic Trend:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3(u'f(t) = \u03B1 + \u03B2t + \u03B3t^2', style={'text-align': 'center', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Exponential Trend:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3(u'f(t) = \u03B1 + e^(\u03B2t)', style={'text-align': 'center', 'color': colors['text']}),

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

            html.H2('Noise', style={'text-align': 'center', 'color': colors['text']}),

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
                id='power_spectral_density'
            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H2('Discontinuities', style={'text-align': 'center', 'color': colors['text']}),

            html.H3('Number of Discontinuities:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="discontinuities",
                          type='float',
                          value=0)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Discontinuity SD:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="discontinuity_sd",
                          type='number',
                          value=1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            dcc.Graph(
                id='discontinuity'
            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H2('Time Series', style={'text-align': 'center', 'color': colors['text']}),

            html.H3('Minimum frequency component 1:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="min_freq_mod_1",
                          type='number',
                          value=1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Maximum frequency component 1:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="max_freq_mod_1",
                          type='number',
                          value=1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Frequency period component 1:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="freq_period_1",
                          type='number',
                          value=1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Minimum amplitude component 1:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="min_amp_mod_1",
                          type='number',
                          value=1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Maximum amplitude component 1:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="max_amp_mod_1",
                          type='number',
                          value=1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Amplitude modulation frequency component 1:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="amp_freq_1",
                          type='number',
                          value=1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Minimum frequency component 2:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="min_freq_mod_2",
                          type='number',
                          value=4)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Maximum frequency component 2:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="max_freq_mod_2",
                          type='number',
                          value=4)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Frequency period component 2:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="freq_period_2",
                          type='number',
                          value=1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Minimum amplitude component 2:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="min_amp_mod_2",
                          type='number',
                          value=1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Maximum amplitude component 2:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="max_amp_mod_2",
                          type='number',
                          value=1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Amplitude modulation frequency component 2:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="amp_freq_2",
                          type='number',
                          value=1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Minimum frequency component 3:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="min_freq_mod_3",
                          type='number',
                          value=8)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Maximum frequency component 3:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="max_freq_mod_3",
                          type='number',
                          value=8)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Frequency period component 3:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="freq_period_3",
                          type='number',
                          value=1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Minimum amplitude component 3:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="min_amp_mod_3",
                          type='number',
                          value=1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Maximum amplitude component 3:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="max_amp_mod_3",
                          type='number',
                          value=1)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Amplitude modulation frequency component 3:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[

                dcc.Input(id="amp_freq_3",
                          type='number',
                          value=1)

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

            html.H1('Time-Frequency Analysis', style={'text-align': 'center', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H2('Fourier Transform', style={'text-align': 'center', 'color': colors['text']}),

            html.H3('Fourier Transform Maximum Frequency:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.Div(children=[

                dcc.Input(id="ft_max_frequency",
                          type='number',
                          value=10)

            ],

                style={'text-align': 'right'}

            ),

            html.H3('Fourier Transform Window:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            dcc.Dropdown(
                id='ft_window',
                options=[{'label': 'No Window', 'value': True},
                         {'label': 'Window', 'value': 'window'}],
                value=True
            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            dcc.Graph(
                id='FT'
            ),

            html.H2('Short-Time Fourier Transform', style={'text-align': 'center', 'color': colors['text']}),

            html.H3('Short-Time Fourier Transform Maximum Frequency:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.Div(children=[

                dcc.Input(id="stft_max_frequency",
                          type='number',
                          value=10)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Short-Time Fourier Transform Window Width:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.Div(children=[

                dcc.Input(id="stft_window_width",
                          type='number',
                          value=256)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            dcc.Graph(
                id='STFT'
            ),

            html.H2('Wavelet Transform', style={'text-align': 'center', 'color': colors['text']}),

            html.H3('Morlet Wavelet Maximum Frequency:',
                    style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.Div(children=[

                dcc.Input(id="mwt_max_frequency",
                          type='number',
                          value=10)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.H3('Morlet Wavelet Window Width:', style={'text-align': 'left', 'color': colors['text']}),

            html.Div(children=[], style={'marginBottom': '1em'}),

            html.Div(children=[

                dcc.Input(id="mwt_window_width",
                          type='number',
                          value=256)

            ],

                style={'text-align': 'right'}

            ),

            html.Div(children=[], style={'marginBottom': '1em'}),

            dcc.Graph(
                id='MWT'
            ),

            html.H2('Intrinsic Mode Functions', style={'text-align': 'center', 'color': colors['text']}),

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
    output=[Output(component_id='power_spectral_density', component_property='figure'),
            Output(component_id='discontinuity', component_property='figure'),
            Output(component_id='time_series_graph', component_property='figure'),
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
           State(component_id='discontinuities', component_property='value'),
           State(component_id='discontinuity_sd', component_property='value'),

           State(component_id='min_freq_mod_1', component_property='value'),
           State(component_id='max_freq_mod_1', component_property='value'),
           State(component_id='freq_period_1', component_property='value'),
           State(component_id='min_amp_mod_1', component_property='value'),
           State(component_id='max_amp_mod_1', component_property='value'),
           State(component_id='amp_freq_1', component_property='value'),

           State(component_id='min_freq_mod_2', component_property='value'),
           State(component_id='max_freq_mod_2', component_property='value'),
           State(component_id='freq_period_2', component_property='value'),
           State(component_id='min_amp_mod_2', component_property='value'),
           State(component_id='max_amp_mod_2', component_property='value'),
           State(component_id='amp_freq_2', component_property='value'),

           State(component_id='min_freq_mod_3', component_property='value'),
           State(component_id='max_freq_mod_3', component_property='value'),
           State(component_id='freq_period_3', component_property='value'),
           State(component_id='min_amp_mod_3', component_property='value'),
           State(component_id='max_amp_mod_3', component_property='value'),
           State(component_id='amp_freq_3', component_property='value'),

           State(component_id='ft_max_frequency', component_property='value'),
           State(component_id='ft_window', component_property='value'),
           State(component_id='stft_max_frequency', component_property='value'),
           State(component_id='stft_window_width', component_property='value'),
           State(component_id='mwt_max_frequency', component_property='value'),
           State(component_id='mwt_window_width', component_property='value'),
           State(component_id='max_frequency', component_property='value'),
           State(component_id='which_imfs', component_property='value')])

def update_output(n_click, trend, alpha, beta, gamma, noise_bool, noise_mean, noise_sd, discontinuities,
                  discontinuity_sd, min_freq_mod_1, max_freq_mod_1, freq_period_1, min_amp_mod_1, max_amp_mod_1, amp_freq_1,
                  min_freq_mod_2, max_freq_mod_2, freq_period_2, min_amp_mod_2, max_amp_mod_2, amp_freq_2,
                  min_freq_mod_3, max_freq_mod_3, freq_period_3, min_amp_mod_3, max_amp_mod_3, amp_freq_3,
                  ft_max_frequency, ft_window, stft_max_frequency, stft_window_width,
                  mwt_max_frequency, mwt_window_width, max_frequency, which_imfs):

    freq_time = frequency(time, freq_period_1, min_freq_mod_1, max_freq_mod_1)
    amp_time_series = ((max_amp_mod_1 - min_amp_mod_1) / 2) * np.sin(amp_freq_1 * time) + \
                      ((max_amp_mod_1 - min_amp_mod_1) / 2) + min_amp_mod_1
    time_series = amp_time_series * np.cos(freq_time)

    freq_time = frequency(time, freq_period_2, min_freq_mod_2, max_freq_mod_2)
    amp_time_series = ((max_amp_mod_2 - min_amp_mod_2) / 2) * np.sin(amp_freq_2 * time) + \
                      ((max_amp_mod_2 - min_amp_mod_2) / 2) + min_amp_mod_2
    time_series += amp_time_series * np.cos(freq_time)

    freq_time = frequency(time, freq_period_3, min_freq_mod_3, max_freq_mod_3)
    amp_time_series = ((max_amp_mod_3 - min_amp_mod_3) / 2) * np.sin(amp_freq_3 * time) + \
                      ((max_amp_mod_3 - min_amp_mod_3) / 2) + min_amp_mod_3
    time_series += amp_time_series * np.cos(freq_time)

    if not trend:
        pass
    else:
        if trend == 'linear':
            time_series += alpha * np.ones_like(time) + beta * time
        elif trend == 'quadratic':
            time_series += alpha * np.ones_like(time) + beta * time + gamma * time ** 2
        elif trend == 'exponential':
            time_series += alpha * np.exp(beta * time)

    if noise_bool:
        fig_1_optional_1 = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{
                          "text": "No matching data found",
                          "xref": "paper",
                          "yref": "paper",
                          "showarrow": False,
                          "font": {"size": 28}}]}}
    if noise_bool == 'violet noise':
        noise = noise_mean * np.ones_like(time_series) + noise_sd * cn.powerlaw_psd_gaussian(-2, len(time_series))
        time_series += noise
        dt = 0.04
        s, f = mlab.psd(noise, Fs=1 / dt)
        fig_1_optional_1 = px.line(x=np.log10(f), y=np.log10(s), labels={"x": 'log10(Frequency)', "y": 'log10(Power)'})
        fig_1_optional_1.update_layout(title={'text': "Violet Noise Power Spectral Density", 'y': 0.95, 'x': 0.5,
                                            'xanchor': 'center', 'yanchor': 'top'})
        fig_1_optional_1.add_trace(go.Scatter(x=np.log10(f),
                                            y=np.log10(f ** 2) - np.mean(np.log10(f ** 2)[1:] - np.log10(s)[1:]),
                                            mode='lines', name='P \u221D f^2', line=dict(color='#101921')))
        fig_1_optional_1.data[0].line.color = '#8F00FF'
    elif noise_bool == 'blue noise':
        noise = noise_mean * np.ones_like(time_series) + noise_sd * cn.powerlaw_psd_gaussian(-1, len(time_series))
        time_series += noise
        dt = 0.04
        s, f = mlab.psd(noise, Fs=1 / dt)
        fig_1_optional_1 = px.line(x=np.log10(f), y=np.log10(s), labels={"x": 'log10(Frequency)', "y": 'log10(Power)'})
        fig_1_optional_1.update_layout(title={'text': "Blue Noise Power Spectral Density", 'y': 0.95, 'x': 0.5,
                                            'xanchor': 'center', 'yanchor': 'top'})
        fig_1_optional_1.add_trace(go.Scatter(x=np.log10(f),
                                            y=np.log10(f) - np.mean(np.log10(f)[1:] - np.log10(s)[1:]),
                                            mode='lines', name='P \u221D f', line=dict(color='#101921')))
        fig_1_optional_1.data[0].line.color = '#0000FF'
    elif noise_bool == 'white noise':
        noise = noise_mean * np.ones_like(time_series) + noise_sd * cn.powerlaw_psd_gaussian(0, len(time_series))
        time_series += noise
        dt = 0.04
        s, f = mlab.psd(noise, Fs=1 / dt)
        fig_1_optional_1 = px.line(x=np.log10(f), y=np.log10(s),
                                 labels={"x": 'log10(Frequency)', "y": 'log10(Power)'})
        fig_1_optional_1.update_layout(title={'text': "White Noise Power Spectral Density", 'y': 0.95, 'x': 0.5,
                                            'xanchor': 'center', 'yanchor': 'top'})
        fig_1_optional_1.add_trace(go.Scatter(x=np.log10(f),
                                            y=np.ones_like(f) - np.mean(np.ones_like(f)[1:] - np.log10(s)[1:]),
                                            mode='lines', name='P \u221D k', line=dict(color='#101921')))
        fig_1_optional_1.data[0].line.color = '#808080'
    elif noise_bool == 'pink noise':
        noise = noise_mean * np.ones_like(time_series) + noise_sd * cn.powerlaw_psd_gaussian(1, len(time_series))
        time_series += noise
        dt = 0.04
        s, f = mlab.psd(noise, Fs=1 / dt)
        fig_1_optional_1 = px.line(x=np.log10(f), y=np.log10(s),
                                 labels={"x": 'log10(Frequency)', "y": 'log10(Power)'})
        fig_1_optional_1.update_layout(title={'text': "Pink Noise Power Spectral Density", 'y': 0.95, 'x': 0.5,
                                            'xanchor': 'center', 'yanchor': 'top'})
        fig_1_optional_1.add_trace(go.Scatter(x=np.log10(f),
                                            y=np.log10(1 / f) - np.mean(np.log10(1 / f)[1:] - np.log10(s)[1:]),
                                            mode='lines', name='P \u221D 1/f', line=dict(color='#101921')))
        fig_1_optional_1.data[0].line.color = '#FFC0CB'
    elif noise_bool == 'brown noise':
        noise = noise_mean * np.ones_like(time_series) + noise_sd * cn.powerlaw_psd_gaussian(2, len(time_series))
        time_series += noise
        dt = 0.04
        s, f = mlab.psd(noise, Fs=1 / dt)
        fig_1_optional_1 = px.line(x=np.log10(f),
                                 y=np.log10(s),
                                 labels={"x": 'log10(Frequency)', "y": 'log10(Power)'})
        fig_1_optional_1.update_layout(title={'text': "Brown Noise Power Spectral Density", 'y': 0.95, 'x': 0.5,
                                            'xanchor': 'center', 'yanchor': 'top'})
        fig_1_optional_1.add_trace(go.Scatter(x=np.log10(f),
                                            y=np.log10(1 / (f ** 2)) - np.mean(np.log10(1 / (f ** 2))[1:] - np.log10(s)[1:]),
                                            mode='lines', name='P \u221D 1/f^2', line=dict(color='#101921')))
        fig_1_optional_1.data[0].line.color = '#964B00'

    if discontinuities == 0:
        fig_1_optional_2 = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{
            "text": "No matching data found",
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 28}}]}}
    else:
        discontinuity_vector = np.zeros_like(time_series)
        for discontinue in range(int(discontinuities)):
            discontinue_location_bool = time > np.random.uniform(0, 1) * time[-1]
            discontinuity_vector[discontinue_location_bool] = \
                np.random.normal(0, discontinuity_sd) * np.ones_like(time_series)[discontinue_location_bool]
        fig_1_optional_2 = px.line(x=time, y=discontinuity_vector,
                        labels={"x": "Time", "y": "Displacement"})
        fig_1_optional_2.update_layout(title={'text': "Discontinuity Vector", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
        time_series += discontinuity_vector

    fig_1 = px.line(x=time, y=time_series,
                    labels={"x": "Time", "y": "Displacement"})
    fig_1.update_layout(title={'text': "Time Series", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})

    f_time = fftfreq(int(len(time_series) - 1), time[1] - time[0])[:int(len(time_series) - 1) // 2] * 2 * np.pi

    if ft_window:
        ft_window_multiply = np.ones_like(time_series)
    if ft_window == 'window':
        ft_window_multiply = sig.get_window(window='hann', Nx=len(time_series), fftbins=False)
    f_time_series = fft(ft_window_multiply * time_series)

    x_hs, y, z = [0, 5 * np.pi], f_time, (2.0 / int(len(time_series) - 1) *
                                          np.abs(f_time_series[0:int(len(time_series) - 1) // 2])).reshape(-1, 1)
    fig_2 = go.Figure(data=go.Heatmap(z=np.abs(z), y=y, x=x_hs))
    fig_2.update_layout(yaxis_range=[0, ft_max_frequency])

    hilbert = emd_hilbert.Hilbert(time=time, time_series=time_series)

    x_hs, y, z = hilbert.stft_custom(window_width=stft_window_width, window='hann')
    fig_3 = go.Figure(data=go.Heatmap(z=np.abs(z), y=y, x=x_hs))
    fig_3.update_layout(yaxis_range=[0, stft_max_frequency])

    x_hs, y, z = hilbert.morlet_wavelet_custom(window_width=mwt_window_width)
    fig_4 = go.Figure(data=go.Heatmap(z=np.abs(z), y=y, x=x_hs))
    fig_4.update_layout(yaxis_range=[0, mwt_max_frequency])

    emd = AdvEMDpy.EMD(time_series=time_series, time=time)
    emd = emd.empirical_mode_decomposition(knots=knots)

    imfs = emd[0]

    fig_5 = subplots.make_subplots(rows=int(np.shape(imfs)[0] - 1), cols=1)

    for imf in range(int(np.shape(imfs)[0] - 1)):

        fig_5.add_trace(go.Scatter(x=time, y=imfs[int(imf + 1), :], mode='lines',
                                   name='IMF {}'.format(str(int(imf + 1)))), row=int(imf + 1), col=1)

    hs = emd_hilbert.hilbert_spectrum(time=time, imf_storage=imfs, ht_storage=emd[1], if_storage=emd[2],
                                      max_frequency=float(max_frequency), plot=False, which_imfs=which_imfs)

    fig_6 = go.Figure(data=go.Heatmap(z=hs[2], y=hs[1][:, 0], x=hs[0][0, :]))

    return fig_1_optional_1, fig_1_optional_2, fig_1, fig_2, fig_3, fig_4, fig_5, fig_6

if __name__ == '__main__':
    app.run_server(debug=True)

# Dash App - end
