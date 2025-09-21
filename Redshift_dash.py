#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JWST Spectral Lab - Plotly Dash Version
Converted from matplotlib to web-deployable Dash app

@author: jansen (converted to Dash)
"""

import sys
import os
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# Try to import astropy, provide fallback if not available
try:
    from astropy.io import fits as pyfits
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    print("Warning: astropy not available. Using mock data for demo.")

nan = float('nan')
pi = np.pi
e = np.e
c = 3.*10**8

class Redshift_dash:
    def __init__(self):
        """Initialize the Dash application"""
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "JWST Spectral Lab - Redshift"
        
        # Initialize data variables
        self.data_wave = None
        self.data_flux = None
        self.data_error = None
        self.ztrue = 9.436
        self.z = 1
        self.target = 'generic'
        self.score = 0
        self.show_score_flag = False
        
        # Data file configurations
        self.data_files = {
            'SF943': {'file': '10058975_prism_clear_v5.0_1D.fits', 'ztrue': 9.436, 'target': 'generic'},
            'QC_galaxy': {'file': '199773_prism_clear_v5.0_1D.fits', 'ztrue': 2.820, 'target': 'generic'},
            'SF_galaxy': {'file': '001882_prism_clear_v5.0_1D.fits', 'ztrue': 5.4431, 'target': 'generic'},
            'GSz14': {'file': '183348_prism_clear_v5.0_1D.fits', 'ztrue': 14.18, 'target': 'GSz14'},
            'COS30': {'file': '007437_prism_clear_v3.1_1D.fits', 'ztrue': 6.856, 'target': 'generic'},
            'SF2': {'file': '001927_prism_clear_v5.0_1D.fits', 'ztrue': 3.6591, 'target': 'generic'},
            'PSB': {'file': '023286_prism_clear_v5.1_1D.fits', 'ztrue': 1.781, 'target': 'generic'},
            'zhig': {'file': '066585_prism_clear_v5.1_1D.fits', 'ztrue': 7.1404, 'target': 'low_snr'},
            'zhig2': {'file': '003991_prism_clear_v5.1_1D.fits', 'ztrue': 10.603, 'target': 'gnz11'}
        }
        
        # Load initial data
        self.load_data('SF943')
        
        # Setup app layout and callbacks
        self.setup_layout()
        self.setup_callbacks()
    
    def load_data(self, dataset_key):
        """Load data from FITS files or create mock data"""
        config = self.data_files[dataset_key]
        self.ztrue = config['ztrue']
        self.target = config['target']
        
        if ASTROPY_AVAILABLE:
            try:
                pth = sys.path[0] if sys.path[0] else '.'
                filepath = os.path.join(pth, 'Data', config['file'])
                
                with pyfits.open(filepath) as hdu:
                    self.data_wave = hdu['WAVELENGTH'].data * 1e6
                    self.data_flux = hdu['DATA'].data * 1e-7
                    self.data_error = hdu['ERR'].data * 1e-7
                    
                # Special case for COS30
                if dataset_key == 'COS30':
                    self.data_wave = np.append(self.data_wave, np.linspace(5.32, 5.5, 32))
                    self.data_flux = np.append(self.data_flux, np.zeros(32))
                    self.data_error = np.append(self.data_error, np.ones(32) * 0.001e-18)
                    
            except Exception as e:
                print(f"Error loading {config['file']}: {e}")
                self.create_mock_data()
        else:
            self.create_mock_data()
    
    def create_mock_data(self):
        """Create mock spectral data for demonstration"""
        self.data_wave = np.linspace(0.5, 5.3, 1000)
        # Create a mock spectrum with some emission lines
        self.data_flux = (np.exp(-(self.data_wave - 2.5)**2 / 0.5) * 0.01 + 
                         np.random.normal(0, 0.001, len(self.data_wave)) + 0.005)
        self.data_error = np.ones_like(self.data_flux) * 0.001
    
    def get_emission_lines(self):
        """Get emission line data with redshift applied"""
        emlines = {
            r'C⁺⁺': (1907., 'red'),
            r'Mg⁺': (2797., 'blue'),
            r'[O⁺]': (3728., 'green'),
            r'[Ne⁺⁺]': (3869.860, 'purple'),
            'Hδ': (4102.860, 'orange'),
            'Hγ': (4341.647, 'pink'),
            'Hβ': (4862.647, 'brown'),
            r'[O⁺⁺]': (4960.0, 'red'),  # Simplified to single line
            r'[O⁰]': (6302.0, 'green'),  # Simplified to single line
            'Na': (5891.583, 'yellow'),
            'Hα': (6564.522, 'red'),
            r'[S⁺]': (6725, 'blue'),
            r'[S⁺⁺]': (9070.0, 'purple'),  # Simplified to single line
            'HeI': (10832.1, 'orange'),
            'Paγ': (10940.978, 'pink'),
            r'[Fe⁺]': (12570.200, 'brown'),
            'Paβ': (12821.432, 'green'),
            'Paα': (18755.80357, 'red'),
        }
        
        visible_lines = {}
        for line_name, (rest_wave, color) in emlines.items():
            obs_wave = rest_wave * (1 + self.z) / 1.e4
            if 0.5 * 1.001 < obs_wave < 5.3 * 0.999:
                visible_lines[line_name] = (obs_wave, color)
        
        return visible_lines
    
    def calculate_score(self):
        """Calculate redshift score"""
        return (self.z - self.ztrue) / (1 + self.ztrue) * 3e5
    
    def setup_layout(self):
        """Setup the Dash app layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("JWST Spectral Lab - Redshift", className="text-center mb-4"),
                ], width=12)
            ]),
            
            # Control buttons
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("SF943", id="btn-SF943", color="info", size="sm"),
                        dbc.Button("QC_galaxy", id="btn-QC_galaxy", color="info", size="sm"),
                        dbc.Button("SF_galaxy", id="btn-SF_galaxy", color="info", size="sm"),
                        dbc.Button("GSz14", id="btn-GSz14", color="info", size="sm"),
                        dbc.Button("COS30", id="btn-COS30", color="info", size="sm"),
                        dbc.Button("SF2", id="btn-SF2", color="info", size="sm"),
                        dbc.Button("PSB", id="btn-PSB", color="info", size="sm"),
                        dbc.Button("zhig", id="btn-zhig", color="info", size="sm"),
                        dbc.Button("zhig2", id="btn-zhig2", color="info", size="sm"),
                    ], className="mb-3")
                ], width=12)
            ]),
            
            # Main plot
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="main-plot", style={'height': '600px'})
                ], width=12)
            ]),
            
            # Redshift slider
            dbc.Row([
                dbc.Col([
                    html.Label("Redshift", className="fw-bold"),
                    dcc.Slider(
                        id="redshift-slider",
                        min=1,
                        max=15,
                        step=0.01,
                        value=1,
                        marks={i: str(i) for i in range(1, 16)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], width=10),
                dbc.Col([
                    dbc.Button("Show Score", id="show-score-btn", color="success", className="mt-4")
                ], width=2)
            ], className="mt-3"),
            
            # Score display
            dbc.Row([
                dbc.Col([
                    html.Div(id="score-display", className="mt-3")
                ], width=12)
            ])
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup Dash callbacks"""
        
        # Combined callback for all interactions
        @self.app.callback(
            [Output("main-plot", "figure"),
             Output("score-display", "children")],
            [Input("redshift-slider", "value"),
             Input("show-score-btn", "n_clicks")] + 
            [Input(f"btn-{dataset_key}", "n_clicks") for dataset_key in self.data_files.keys()],
            prevent_initial_call=False
        )
        def update_app(*args):
            ctx = callback_context
            if not ctx.triggered:
                # Initial load
                self.z = args[0]  # redshift slider value
                return self.create_plot(), ""
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Handle redshift slider
            if trigger_id == "redshift-slider":
                self.z = args[0]
                return self.create_plot(), dash.no_update
            
            # Handle dataset buttons
            else:
                for i, dataset_key in enumerate(self.data_files.keys()):
                    if trigger_id == f"btn-{dataset_key}":
                        n_clicks = args[2 + i]  # Button clicks start from index 2
                        if n_clicks:
                            self.load_data(dataset_key)
                            self.z = args[0]  # Keep current redshift slider value
                            return self.create_plot(), ""
                        break
            
            return dash.no_update, dash.no_update
    
    def create_plot(self):
        """Create the main spectral plot"""
        if self.data_wave is None:
            return go.Figure()
        
        fig = go.Figure()
        
        self.score = (self.z-self.ztrue)/(1+self.ztrue)*3e5
        
        # Add main spectrum
        fig.add_trace(go.Scatter(
            x=self.data_wave,
            y=self.data_flux / 1e-18,
            mode='lines',
            line=dict(color='black', shape='hv'),
            name='Spectrum',
            showlegend=False
        ))
        
        # Add emission lines
        emission_lines = self.get_emission_lines()
        for line_name, (obs_wave, color) in emission_lines.items():
            fig.add_vline(
                x=obs_wave,
                line=dict(color=color, dash='dash', width=2),
                opacity=0.7
            )
            
            # Add line labels
            fig.add_annotation(
                x=obs_wave,
                y=0.95,
                text=line_name,
                textangle=90,
                showarrow=False,
                yref='paper',
                bgcolor='white',
                bordercolor=color,
                font=dict(size=10)
            )
        
        # Set layout
        fig.update_layout(
            title=f"JWST Spectrum - Redshift: {self.z:.3f} (Score: {self.score:.0f})",
            xaxis_title="Wavelength (μm) - blue ← → red",
            yaxis_title="Brightness (×10⁻¹⁸)",
            xaxis=dict(range=[0.5, 5.3]),
            template="plotly_white",
            height=600
        )
        
        # Set y-axis limits based on target type
        if self.target == 'GSz14':
            fig.update_yaxes(range=[-0.00025, 0.01])
        elif self.target == 'gnz11':
            fig.update_yaxes(range=[-0.01, 0.04])
        elif self.target == 'low_snr':
            fig.update_yaxes(range=[-0.01, 0.025])
        
        return fig
    
    def run_server(self, debug=True, port=8050):
        """Run the Dash server"""
        self.app.run(debug=debug, port=port)

