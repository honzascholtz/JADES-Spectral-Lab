#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JWST Stellar Population Lab - Plotly Dash Version
Converted from matplotlib to web-deployable Dash app

@author: jansen (converted to Dash)
"""

import sys
import os
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

import numpy as np
from astropy.modeling.models import Sersic2D
import astropy.io.fits as pyfits
import astropy.stats as stats
nan = float('nan')
pi = np.pi
e = np.e
c = 3.*10**8

class JADES_photo_lab:
    def __init__(self):
        """Initialize the Dash application"""
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "JWST Image Lab"
        
        # Initialize data variables
        self.data_wave = None
        self.data_flux = None
        self.data_error = None
        self.model = None
        self.model_spectrum = None

        # Model parameters
        self.amplitude = 1
        self.radius = 10
        self.index = 4
        self.x = 84
        self.y = 84
        self.ellipticity = 0.5
        self.theta = -1

        self.target = 'generic'
        self.score = 0

        self.image_lim = 20
        
        # Data file configurations
        self.data_files = {
            'Galaxy1': {'file': 'hlsp_jades_jwst_nirspec_goods-s-mediumjwst-00188208_clear-prism_v1.0_x1d.fits', 'z': 9.436, 'target': 'generic'},
            'Galaxy2': {'file': 'hlsp_jades_jwst_nirspec_goods-s-mediumjwst-00003204_clear-prism_v1.0_x1d.fits', 'z': 2.820, 'target': 'generic'},
            'SF_galaxy': {'file': '001882_prism_clear_v5.0_1D.fits', 'z': 5.4431, 'target': 'generic'},
            'GSz14': {'file': '183348_prism_clear_v5.0_1D.fits', 'z': 14.18, 'target': 'GSz14'},
            'COS30': {'file': '007437_prism_clear_v3.1_1D.fits', 'z': 6.856, 'target': 'generic'},
            'SF2': {'file': '001927_prism_clear_v5.0_1D.fits', 'z': 3.6591, 'target': 'generic'},
            'PSB': {'file': '023286_prism_clear_v5.1_1D.fits', 'z': 1.781, 'target': 'generic'},
            'zhig': {'file': '066585_prism_clear_v5.1_1D.fits', 'z': 7.1404, 'target': 'low_snr'},
        }
        
        # Load initial data and setup
        self.load_data('Galaxy1')
        self.generate_model()
        
        # Setup app layout and callbacks
        self.setup_layout()
        self.setup_callbacks()
    
    def load_data(self, dataset_key):
        """Load data from FITS files or create mock data"""
        config = self.data_files[dataset_key]
        self.z = config['z']
        self.target = config['target']
        
        try:
            pth = sys.path[0] if sys.path[0] else '.'
            filepath = os.path.join(pth, 'Data/phot',config['file'])
            with pyfits.open(filepath) as hdu:
                self.image = hdu['F444W'].data
                self.image_header = hdu['F444W'].header
                self.image_error = stats.sigma_clipped_stats(self.image, sigma=3.0, maxiters=10)[2] * \
                     np.ones_like(self.image)  # Use stddev as error estimate
                self.shape = self.image.shape

                # Get the zoom region for better contrast calculation
                self.x_min = max(0, int(self.x - self.image_lim))
                self.x_max = min(self.shape[1], int(self.x + self.image_lim))
                self.y_min = max(0, int(self.y - self.image_lim))
                self.y_max = min(self.shape[0], int(self.y + self.image_lim))
                
            
                
        except Exception as e:
            print(f"Error loading {config['file']}: {e}")
            
    
    def generate_model(self):
        """Generate model spectrum with current parameters"""
        
        x, y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))

        self.model = Sersic2D(amplitude=self.amplitude, r_eff=self.radius, n=self.index,\
                               x_0=self.x, y_0=self.y, ellip=self.ellipticity, theta=np.deg2rad(self.theta))
        self.model_image = self.model(x, y)

        # Calculate residual
        self.residual = (self.image - self.model_image)/self.image_error
        
    
    def calculate_score(self):
        """Calculate chi-squared score"""
        
        chi_squared = np.nansum(((self.image[self.y_min:self.y_max, self.x_min:self.x_max]\
                                   - self.model_image[self.y_min:self.y_max, self.x_min:self.x_max]) / \
                                   self.image_error[self.y_min:self.y_max, self.x_min:self.x_max]) ** 2)
        dof = np.sum(~np.isnan(self.image[self.y_min:self.y_max, self.x_min:self.x_max])) - 7  # Number of data points minus number of parameters
        self.score = chi_squared / dof if dof > 0 else np.nan
        return self.score

    
    def setup_layout(self):
        """Setup the Dash app layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("JWST Photometry Lab", className="text-center mb-4"),
                ], width=12)
            ]),
            
            # Control buttons
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("Galaxy1", id="btn-Galaxy1", color="info", size="sm"),
                        dbc.Button("Galaxy2", id="btn-Galaxy2", color="info", size="sm"),
                        dbc.Button("SF_galaxy", id="btn-SF_galaxy", color="info", size="sm"),
                        dbc.Button("GSz14", id="btn-GSz14", color="info", size="sm"),
                        dbc.Button("COS30", id="btn-COS30", color="info", size="sm"),
                        dbc.Button("SF2", id="btn-SF2", color="info", size="sm"),
                        dbc.Button("PSB", id="btn-PSB", color="info", size="sm"),
                        dbc.Button("zhig", id="btn-zhig", color="info", size="sm"),
                    ], className="mb-3")
                ], width=12)
            ]),
            
            # Main plot only
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="main-plot", style={'height': '600px'}),
                ], width=12)
            ], style={'margin-bottom': '60px'}),

            # Parameter sliders - organized in two columns
            dbc.Row([
                # Left column
                dbc.Col([
                    html.Div([
                        html.Label("Peak size", className="fw-bold mb-2"),
                        dcc.Slider(id="amp-slider", min=0.1, max=2, step=0.01, value=0.5,
                                  marks={i: str(i) for i in range(11)},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label("Size (pix)", className="fw-bold mb-2"),
                        dcc.Slider(id="size-slider", min=0, max=10, step=0.1, value=5,
                                  marks={i: str(i) for i in range(11)},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label("Sersic index", className="fw-bold mb-2"),
                        dcc.Slider(id="Sersic-slider", min=0.0, max=5, step=0.1, value=2,
                                  marks={0: '0', 1: '1', 2: '2', 3: '3', 4: '4'},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),

                    html.Div([
                        html.Label("Angle", className="fw-bold mb-2"),
                        dcc.Slider(id="angle-slider", min=0.0, max=180, step=1, value=90,
                                  marks={0: 'horizontal', 45: '45', 90: 'vertical', 135: '135', 180: 'horizontal'},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4")
                ], width=6),
                
                # Right column
                dbc.Col([
                    html.Div([
                        html.Label("X", className="fw-bold mb-2"),
                        dcc.Slider(id="x-slider", min=70, max=98, step=0.5, value=84,
                                  marks={i: str(i) for i in range(70, 98)},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label("Y", className="fw-bold mb-2"),
                        dcc.Slider(id="y-slider", min=70, max=98, step=0.5, value=84,
                                  marks={i: str(i) for i in range(70, 98)},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label("Ellipticity", className="fw-bold mb-2"),
                        dcc.Slider(id="ellipticity-slider", min=0, max=1, step=0.01, value=0.5,
                                  marks={0: 'Circle', 0.5: '0.5', 1: 'Line'},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4")
                ], width=6)
            ], className="mt-3")
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup Dash callbacks"""
        
        # Combined callback for all interactions
        @self.app.callback(
            Output("main-plot", "figure"),
            [Input("amp-slider", "value"),
             Input("size-slider", "value"),
             Input("Sersic-slider", "value"),
             Input("angle-slider", "value"),
             Input("x-slider", "value"),
             Input("y-slider", "value"),
             Input("ellipticity-slider", "value")],
            [Input(f"btn-{dataset_key}", "n_clicks") for dataset_key in self.data_files.keys()],
            prevent_initial_call=False
        )
        def update_app(*args):
            ctx = callback_context
            
            # Update parameters from sliders
            self.amplitude = args[0]
            self.radius = args[1]
            self.index = args[2]
            self.theta = args[3]
            self.x = args[4]
            self.y = args[5]
            self.ellipticity = args[6]
            
            # Check if a dataset button was clicked
            if ctx.triggered:
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
                for i, dataset_key in enumerate(self.data_files.keys()):
                    if trigger_id == f"btn-{dataset_key}":
                        n_clicks = args[7 + i]
                        if n_clicks:
                            self.load_data(dataset_key)
                        break
            
            # Generate model with current parameters
            self.generate_model()
            
            # Create plot
            main_fig = self.create_main_plot()
            return main_fig
        
    def create_main_plot(self):
        """Create the main spectral plot with three panels"""
        from plotly.subplots import make_subplots
        import plotly.express as px

        # Better scaling for astronomical images using asinh (arcsinh) stretch
        # This is commonly used in astronomy as it handles both faint and bright features well
        image_zoom = self.image[self.y_min:self.y_max, self.x_min:self.x_max]
        model_zoom = self.model_image[self.y_min:self.y_max, self.x_min:self.x_max]

        # Calculate robust statistics for better scaling
        # Use percentiles to avoid outliers
        vmin = np.percentile(image_zoom, 1)
        vmax = np.percentile(image_zoom, 99.5)
        
        # Asinh scaling - excellent for astronomical images
        # a parameter controls the transition between linear and log
        a = (vmax - vmin) / 10  # Adjust this for more/less contrast
        
        def asinh_stretch(data, vmin, vmax, a):
            """Apply asinh stretch - better than log for astro images"""
            data_normalized = (data - vmin) / (vmax - vmin)
            return np.arcsinh(data_normalized / a) / np.arcsinh(1 / a)
        
        image_scaled = asinh_stretch(self.image, vmin, vmax, 0.1)
        model_scaled = asinh_stretch(self.model_image, vmin, vmax, 0.1)
        
        # For residual, we can use symmetric log or keep linear
        # Using linear for residual to show positive and negative differences
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Observed Image (log)", "Model (log)", "Residual (Observed - Model)"),
            horizontal_spacing=0.1
        )
        
        # Add observed image (log scale)
        fig.add_trace(
            go.Heatmap(
                z=image_scaled,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(x=0.3, len=0.8, title="log10(Flux)")
            ),
            row=1, col=1
        )
        
        # Add model image (log scale)
        fig.add_trace(
            go.Heatmap(
                z=model_scaled,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(x=0.63, len=0.8, title="log10(Flux)")
            ),
            row=1, col=2
        )
        
        # Add residual (linear, with diverging colorscale)
        fig.add_trace(
            go.Heatmap(
                z=self.residual,
                colorscale='RdBu_r',  # Red-Blue diverging
                showscale=True,
                zmid=0,  # Center the colorscale at 0
                zmin=-5, zmax=5,
                colorbar=dict(x=1.0, len=0.8, title="Residual")
            ),
            row=1, col=3
        )
        
        # Update layout
        fig.update_layout(
            title=f"JWST Photometry Fit, score= {self.calculate_score():.2f}",
            template="plotly_white",
            height=600,
            showlegend=False
        )
        
        # Set zoom range for all subplots
        x_range = [self.x - self.image_lim, self.x + self.image_lim]
        y_range = [self.y - self.image_lim, self.y + self.image_lim]
        
        fig.update_xaxes(range=x_range, title_text="X (pixels)", row=1, col=1)
        fig.update_xaxes(range=x_range, title_text="X (pixels)", row=1, col=2)
        fig.update_xaxes(range=x_range, title_text="X (pixels)", row=1, col=3)
        
        fig.update_yaxes(range=y_range, title_text="Y (pixels)", row=1, col=1)
        fig.update_yaxes(range=y_range, title_text="Y (pixels)", row=1, col=2)
        fig.update_yaxes(range=y_range, title_text="Y (pixels)", row=1, col=3)
        
        return fig

    
    
    def run_server(self, debug=True, port=8051):
        """Run the Dash server"""
        self.app.run(debug=debug, port=port)


