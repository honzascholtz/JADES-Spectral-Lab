#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JWST Stellar Population Lab - Flask Integration
Dash app integrated with Flask

@author: jansen (converted to Flask)
"""

from flask import Flask, render_template
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
from astropy.convolution import Gaussian2DKernel, convolve_fft

import astropy.io.fits as pyfits
import astropy.stats as stats

nan = float('nan')
pi = np.pi
e = np.e
c = 3.*10**8

# Create Flask server
server = Flask(__name__)

class JADES_photo_lab:
    def __init__(self, server):
        """Initialize the Dash application with Flask server"""
        # Create Dash app with Flask server
        self.app = dash.Dash(
            __name__, 
            server=server,
            url_base_pathname='/dashboard/',  # Dash app will be at /dashboard/
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
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
        self.x = 0
        self.y = 0
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
        
        if 1==1:
            pth = sys.path[0] if sys.path[0] else '.'
            filepath = os.path.join(pth, 'Data/phot',config['file'])
            with pyfits.open(filepath) as hdu:
                self.image = hdu['F444W'].data
                self.image = self.image[84-self.image_lim:84+self.image_lim+1, 84-self.image_lim:84+self.image_lim+1]
                
                self.image_header = hdu['F444W'].header
                self.image_error = stats.sigma_clipped_stats(self.image, sigma=3.0, maxiters=10)[2] * \
                     np.ones_like(self.image)
                self.shape = self.image.shape

                self.psf_pixel = 0.145/(self.image_header['CDELT1']*3600) / 2.355
                self.PSF_kernel = Gaussian2DKernel(self.psf_pixel)
    
    def generate_model(self):
        """Generate model spectrum with current parameters"""
        x, y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))

        self.model = Sersic2D(
            amplitude=self.amplitude, 
            r_eff=self.radius, 
            n=self.index,
            x_0=self.x+self.image_lim, 
            y_0=self.y+self.image_lim, 
            ellip=self.ellipticity, 
            theta=np.deg2rad(self.theta)
        )
        self.model_image = self.model(x, y)
        self.model_image = convolve_fft(self.model_image, self.PSF_kernel)
        self.residual = (self.image - self.model_image)/self.image_error
        
    def calculate_score(self):
        """Calculate chi-squared score"""
        chi_squared = np.nansum(((self.image - self.model_image) / self.image_error) ** 2)
        dof = np.sum(~np.isnan(self.image)) - 7
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
            
            # Main plot
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="main-plot", style={'height': '600px'}),
                ], width=12)
            ], style={'margin-bottom': '60px'}),

            # Parameter sliders
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Label("Peak size", className="fw-bold mb-2"),
                        dcc.Slider(id="amp-slider", min=0.1, max=10, step=0.01, value=0.5,
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
                
                dbc.Col([
                    html.Div([
                        html.Label("X", className="fw-bold mb-2"),
                        dcc.Slider(id="x-slider", min=-5, max=5, step=0.1, value=0,
                                  marks={i: str(i) for i in range(-5, 6)},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label("Y", className="fw-bold mb-2"),
                        dcc.Slider(id="y-slider", min=-5, max=5, step=0.1, value=0,
                                  marks={i: str(i) for i in range(-5, 6)},
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
            
            self.amplitude = args[0]
            self.radius = args[1]
            self.index = args[2]
            self.theta = args[3]
            self.x = args[4]
            self.y = args[5]
            self.ellipticity = args[6]
            
            if ctx.triggered:
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
                for i, dataset_key in enumerate(self.data_files.keys()):
                    if trigger_id == f"btn-{dataset_key}":
                        n_clicks = args[7 + i]
                        if n_clicks:
                            self.load_data(dataset_key)
                        break
            
            self.generate_model()
            main_fig = self.create_main_plot()
            return main_fig
        
    def create_main_plot(self):
        """Create the main spectral plot with three panels"""
        vmin = np.percentile(self.image, 1)
        vmax = np.percentile(self.image, 99.5)
        
        def asinh_stretch(data, vmin, vmax, a):
            data_normalized = (data - vmin) / (vmax - vmin)
            return np.arcsinh(data_normalized / a) / np.arcsinh(1 / a)
        
        image_scaled = asinh_stretch(self.image, vmin, vmax, 0.1)
        model_scaled = asinh_stretch(self.model_image, vmin, vmax, 0.1)
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Observed Image", "Your Model", "Difference (Observed - Model)"),
            horizontal_spacing=0.1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=image_scaled,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(x=0.3, len=0.8, title="Brightness")
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Heatmap(
                z=model_scaled,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(x=0.63, len=0.8, title="Brightness")
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Heatmap(
                z=self.residual,
                colorscale='RdBu_r',
                showscale=True,
                zmid=0,
                zmin=-5, zmax=5,
                colorbar=dict(x=1, len=0.8, title="Residual")
            ),
            row=1, col=3
        )

        fig.update_layout(
            title=f"JWST Photometry Fit, score= {self.calculate_score():.2f}, lower is better",
            template="plotly_white",
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1, constrain="domain")
        fig.update_yaxes(constrain="domain", row=1, col=1)
        fig.update_xaxes(scaleanchor="y2", scaleratio=1, row=1, col=2, constrain="domain")
        fig.update_yaxes(constrain="domain", row=1, col=2)
        fig.update_xaxes(scaleanchor="y3", scaleratio=1, row=1, col=3, constrain="domain")
        fig.update_yaxes(constrain="domain", row=1, col=3)
        
        return fig


# Flask routes
@server.route('/')
def index():
    """Home page with link to dashboard"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>JWST Lab Home</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
            }
            h1 { color: #333; }
            a {
                display: inline-block;
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin-top: 20px;
            }
            a:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <h1>Welcome to JWST Photometry Lab</h1>
        <p>This application allows you to fit galaxy models to JWST Imageobservations.</p>
        <a href="/dashboard/">Launch Dashboard</a>
    </body>
    </html>
    '''

@server.route('/api/health')
def health():
    """API health check endpoint"""
    return {'status': 'healthy', 'message': 'Flask + Dash integration working'}


# Initialize the app
dash_app = JADES_photo_lab(server)

if __name__ == '__main__':
    # Run the Flask server (which includes the Dash app)
    server.run(debug=True, port=8051)