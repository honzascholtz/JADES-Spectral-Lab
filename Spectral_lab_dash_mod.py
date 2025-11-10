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

# Try to import required packages, provide fallbacks if not available
try:
    from astropy.io import fits as pyfits
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    print("Warning: astropy not available. Using mock data for demo.")

try:
    import bagpipes as pipes
    pipes.config.max_redshift = 17
    BAGPIPES_AVAILABLE = True
except ImportError:
    BAGPIPES_AVAILABLE = False
    print("Warning: bagpipes not available. Using mock model generation.")


nan = float('nan')
pi = np.pi
e = np.e
c = 3.*10**8

class JADES_spectral_lab:
    def __init__(self):
        """Initialize the Dash application"""
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "JWST Stellar Population Lab"
        
        # Initialize data variables
        self.data_wave = None
        self.data_flux = None
        self.data_error = None
        self.model = None
        self.model_spectrum = None
        
        # Model parameters
        self.z = 9.431
        self.Mass = 9.0
        self.age = 0.3
        self.tau = 0.3
        self.Z = 1.0
        self.U = -3
        self.Av = 0.5
        self.target = 'generic'
        self.score = 0
        
        # Data file configurations
        self.data_files = {
            'SF943': {'file': '10058975_prism_clear_v5.0_1D.fits', 'z': 9.436, 'target': 'generic'},
            'QC_galaxy': {'file': '199773_prism_clear_v5.0_1D.fits', 'z': 2.820, 'target': 'generic'},
            'SF_galaxy': {'file': '001882_prism_clear_v5.0_1D.fits', 'z': 5.4431, 'target': 'generic'},
            'GSz14': {'file': '183348_prism_clear_v5.0_1D.fits', 'z': 14.18, 'target': 'GSz14'},
            'COS30': {'file': '007437_prism_clear_v3.1_1D.fits', 'z': 6.856, 'target': 'generic'},
            'SF2': {'file': '001927_prism_clear_v5.0_1D.fits', 'z': 3.6591, 'target': 'generic'},
            'PSB': {'file': '023286_prism_clear_v5.1_1D.fits', 'z': 1.781, 'target': 'generic'},
            'zhig': {'file': '066585_prism_clear_v5.1_1D.fits', 'z': 7.1404, 'target': 'low_snr'},
        }
        
        # Load initial data and setup
        self.load_data('SF943')
        self.pregenerate_model()
        self.generate_model()
        
        # Setup app layout and callbacks
        self.setup_layout()
        self.setup_callbacks()
    
    def load_data(self, dataset_key):
        """Load data from FITS files or create mock data"""
        config = self.data_files[dataset_key]
        self.z = config['z']
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
        # Create a mock spectrum with some emission lines and continuum
        continuum = 0.005 * (self.data_wave / 2.0) ** (-1.5)
        emission_lines = (np.exp(-((self.data_wave - 1.2) / 0.05) ** 2) * 0.02 +
                         np.exp(-((self.data_wave - 2.1) / 0.03) ** 2) * 0.01 +
                         np.exp(-((self.data_wave - 3.7) / 0.04) ** 2) * 0.015)
        noise = np.random.normal(0, 0.001, len(self.data_wave))
        self.data_flux = continuum + emission_lines + noise
        self.data_error = np.ones_like(self.data_flux) * 0.001
    
    def create_mock_model_spectrum(self):
        """Create mock model spectrum when bagpipes is not available"""
        if self.data_wave is None:
            return
        
        # Simple mock model based on parameters
        age_factor = max(0.1, self.age / 2.0)
        mass_factor = 10 ** (self.Mass - 9.0)
        metal_factor = self.Z
        dust_factor = np.exp(-self.Av / 3.0)
        
        # Mock continuum shape
        continuum = age_factor * mass_factor * metal_factor * dust_factor * 0.005 * (self.data_wave / 2.0) ** (-1.5)
        
        # Mock emission lines if ionization parameter is high enough
        emission = np.zeros_like(self.data_wave)
        if self.U > -4:
            ionization_factor = 10 ** (self.U + 3)
            emission = ionization_factor * (
                np.exp(-((self.data_wave - 1.2) / 0.05) ** 2) * 0.01 +
                np.exp(-((self.data_wave - 2.1) / 0.03) ** 2) * 0.005 +
                np.exp(-((self.data_wave - 3.7) / 0.04) ** 2) * 0.008
            )
        
        model_flux = continuum + emission
        self.model_spectrum = np.column_stack([self.data_wave * 1e4, model_flux])
    
    def pregenerate_model(self):
        """Pre-generate the model components"""
        global BAGPIPES_AVAILABLE
        if BAGPIPES_AVAILABLE:
            try:
                exponential = {
                    "age": self.age,
                    "tau": self.tau,
                    "massformed": self.Mass,
                    "metallicity": self.Z
                }
                
                dust = {
                    "type": "Calzetti",
                    "Av": self.Av
                }
                
                model_components = {
                    "redshift": self.z,
                    "exponential": exponential,
                    "dust": dust
                }
                
                if self.U > -4:
                    nebular = {"logU": self.U}
                    model_components["nebular"] = nebular
                
                # Try to load resolution curve
                try:
                    pth = sys.path[0] if sys.path[0] else '.'
                    with pyfits.open(os.path.join(pth, "Data", "jwst_nirspec_prism_disp.fits")) as hdul:
                        model_components["R_curve"] = np.c_[
                            1e4 * hdul[1].data["WAVELENGTH"], 
                            hdul[1].data["R"]
                        ]
                except:
                    print("Warning: Could not load resolution curve")
                
                self.model = pipes.model_galaxy(
                    model_components, 
                    spec_wavs=self.data_wave * 1e4 if self.data_wave is not None else np.linspace(5000, 53000, 1000)
                )
            except Exception as e:
                print(f"Error creating bagpipes model: {e}")
                BAGPIPES_AVAILABLE = False
                self.create_mock_model_spectrum()
        else:
            self.create_mock_model_spectrum()
    
    def generate_model(self):
        """Generate model spectrum with current parameters"""
        global BAGPIPES_AVAILABLE
        if BAGPIPES_AVAILABLE and self.model is not None:
            try:
                exponential = {
                    "age": self.age,
                    "tau": self.tau,
                    "massformed": self.Mass,
                    "metallicity": self.Z
                }
                
                dust = {
                    "type": "Calzetti",
                    "Av": self.Av
                }
                
                model_components = {
                    "redshift": self.z,
                    "exponential": exponential,
                    "dust": dust
                }
                
                if self.U > -5:
                    nebular = {"logU": self.U}
                    model_components["nebular"] = nebular
                
                # Try to load resolution curve
                try:
                    pth = sys.path[0] if sys.path[0] else '.'
                    with pyfits.open(os.path.join(pth, "Data", "jwst_nirspec_prism_disp.fits")) as hdul:
                        model_components["R_curve"] = np.c_[
                            1e4 * hdul[1].data["WAVELENGTH"], 
                            hdul[1].data["R"]
                        ]
                except:
                    pass
                
                self.model.update(model_components)
                # Get the spectrum from the model
                self.model_spectrum = self.model.spectrum
            except Exception as e:
                print(f"Error updating bagpipes model: {e}")
                self.create_mock_model_spectrum()
        else:
            self.create_mock_model_spectrum()
    
    def calculate_score(self):
        """Calculate chi-squared score"""
        if self.model_spectrum is not None and self.data_flux is not None:
            try:
                model_flux = self.model_spectrum[:, 1]
                if len(model_flux) == len(self.data_flux) and np.nansum(model_flux) > 0:
                    return np.nansum((self.data_flux - model_flux)**2 / self.data_error**2) / (len(self.data_flux) - 6)
            except:
                pass
        return 0.0
    
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
            r'[O⁺⁺]': (4960.0, 'red'),
            r'[O⁰]': (6302.0, 'green'),
            'Na': (5891.583, 'yellow'),
            'Hα': (6564.522, 'red'),
            r'[S⁺]': (6725, 'blue'),
            r'[S⁺⁺]': (9070.0, 'purple'),
            'HeI': (10832.1, 'orange'),
            'Paγ': (10940.978, 'pink'),
            r'[Fe⁺]': (12570.200, 'brown'),
            'Paβ': (12821.432, 'green'),
        }
        
        visible_lines = {}
        for line_name, (rest_wave, color) in emlines.items():
            obs_wave = rest_wave * (1 + self.z) / 1.e4
            if 0.5 * 1.001 < obs_wave < 5.3 * 0.999:
                visible_lines[line_name] = (obs_wave, color)
        
        return visible_lines
    
    def get_universe_age(self, z):
        """Calculate age of universe at given redshift (simplified Planck 2018)"""
        # Simplified cosmology calculation for age of universe
        # Using Planck 2018 parameters: H0=67.4, Omega_m=0.315, Omega_Lambda=0.685
        from scipy.integrate import quad
        
        H0 = 67.4  # km/s/Mpc
        Om = 0.315
        OL = 0.685
        
        def E(zp):
            return np.sqrt(Om * (1 + zp)**3 + OL)
        
        try:
            t_hubble = 9.78 / (H0 / 100)  # Hubble time in Gyr
            integral, _ = quad(lambda zp: 1 / ((1 + zp) * E(zp)), z, np.inf)
            age = t_hubble * integral
            return age
        except:
            # Fallback approximation
            return 13.8 / (1 + z)**1.5
    
    def calculate_sfh(self):
        """Calculate delayed exponential star formation history"""
        # Time array in Gyr
        t = np.linspace(0, self.age * 1.2, 1000)
        
        # Delayed exponential: SFR(t) = t * exp(-t/tau)
        sfr = t * np.exp(-t / self.tau)
        
        # Normalize to total mass formed
        sfr = sfr / np.trapz(sfr, t) * 10**self.Mass
        
        # Convert times to age of universe
        age_universe_now = self.get_universe_age(self.z)
        age_universe = age_universe_now - (self.age - t)
        
        return age_universe, sfr
    
    def setup_layout(self):
        """Setup the Dash app layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("JWST Stellar Population Lab", className="text-center mb-4"),
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
                    ], className="mb-3")
                ], width=12)
            ]),
            
            # Main plots - spectrum (80%) and SFH (20%)
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="main-plot", style={'height': '600px'}),
                ], width=9),
                dbc.Col([
                    dcc.Graph(id="sfh-plot", style={'height': '600px'}),
                ], width=3)
            ], style={'margin-bottom': '60px'}),
            
            # Parameter sliders - organized in two columns
            dbc.Row([
                # Left column
                dbc.Col([
                    html.Div([
                        html.Label("Mass [log(M☉)]", className="fw-bold mb-2"),
                        dcc.Slider(id="mass-slider", min=7, max=12, step=0.1, value=8,
                                  marks={i: str(i) for i in range(7, 13)},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label("Radiation Strength [log U]", className="fw-bold mb-2"),
                        dcc.Slider(id="logU-slider", min=-4.01, max=-1, step=0.1, value=-2.5,
                                  marks={-4: '-4', -3: '-3', -2: '-2', -1: '-1'},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label("Heavy Elements [Z/Z☉]", className="fw-bold mb-2"),
                        dcc.Slider(id="metal-slider", min=0.01, max=1.4, step=0.05, value=0.5,
                                  marks={0: '0', 0.5: '0.5', 1: '1', 1.4: '1.4'},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4")
                ], width=6),
                
                # Right column
                dbc.Col([
                    html.Div([
                        html.Label("Age of Stars [log Gyr]", className="fw-bold mb-2"),
                        dcc.Slider(id="age-slider", min=-2, max=1, step=0.1, value=-1,
                                  marks={-2: '0.01', -1: '0.1', 0: '1', 1: '10'},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label("Decline of Stars [log Gyr]", className="fw-bold mb-2"),
                        dcc.Slider(id="tau-slider", min=-2, max=1, step=0.1, value=-1.4,
                                  marks={-2: '0.01', -1: '0.1', 0: '1', 1: '10'},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label("Dust [Av mag]", className="fw-bold mb-2"),
                        dcc.Slider(id="dust-slider", min=0, max=3, step=0.1, value=0.5,
                                  marks={0: '0', 1: '1', 2: '2', 3: '3'},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4")
                ], width=6)
            ], className="mt-3")
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup Dash callbacks"""
        
        # Combined callback for all interactions
        @self.app.callback(
            [Output("main-plot", "figure"),
             Output("sfh-plot", "figure")],
            [Input("mass-slider", "value"),
             Input("logU-slider", "value"),
             Input("metal-slider", "value"),
             Input("age-slider", "value"),
             Input("tau-slider", "value"),
             Input("dust-slider", "value")] +
            [Input(f"btn-{dataset_key}", "n_clicks") for dataset_key in self.data_files.keys()],
            prevent_initial_call=False
        )
        def update_app(*args):
            ctx = callback_context
            
            # Update parameters from sliders
            self.Mass = args[0]
            self.U = args[1]
            self.Z = args[2]
            self.age = 10 ** args[3]
            self.tau = 10 ** args[4]
            self.Av = args[5]
            
            # Check if a dataset button was clicked
            if ctx.triggered:
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
                for i, dataset_key in enumerate(self.data_files.keys()):
                    if trigger_id == f"btn-{dataset_key}":
                        n_clicks = args[6 + i]
                        if n_clicks:
                            self.load_data(dataset_key)
                            self.pregenerate_model()
                        break
            
            # Generate model with current parameters
            self.generate_model()
            
            # Create plots
            main_fig = self.create_main_plot()
            sfh_fig = self.create_sfh_plot()
            return main_fig, sfh_fig
    
    def create_main_plot(self):
        """Create the main spectral plot"""
        global BAGPIPES_AVAILABLE
        if self.data_wave is None:
            return go.Figure()
        
        fig = go.Figure()
        
        # Check if model is valid (removed universe age check as it was buggy)
        model_valid = (self.model_spectrum is not None and 
                      len(self.model_spectrum) > 0 and
                      np.nansum(self.model_spectrum[:, 1]) > 0)
        
        if not model_valid:
            # Show warning for invalid model
            fig.add_annotation(
                text="Your stars are older than the Universe!<br>Reduce Age parameter!",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=16, color="red"),
                bgcolor="yellow", bordercolor="red"
            )
        else:
            # Add model spectrum first (so it appears behind observed data)
            if self.model_spectrum is not None:
                fig.add_trace(go.Scatter(
                    x=self.model_spectrum[:, 0] / 1e4,
                    y=self.model_spectrum[:, 1] / 1e-18,
                    mode='lines',
                    line=dict(color='firebrick', shape='hv'),
                    name='Model',
                    showlegend=True
                ))
        
        # Add observed spectrum (no error shading)
        fig.add_trace(go.Scatter(
            x=self.data_wave,
            y=self.data_flux / 1e-18,
            mode='lines',
            line=dict(color='black', shape='hv'),
            name='Observations',
            showlegend=True
        ))
        
        # Add emission lines if model is valid
        if model_valid:
            emission_lines = self.get_emission_lines()
            for line_name, (obs_wave, color) in emission_lines.items():
                fig.add_vline(
                    x=obs_wave,
                    line=dict(color=color, dash='dash', width=1.5),
                    opacity=0.5
                )
                
                fig.add_annotation(
                    x=obs_wave,
                    y=0.95,
                    text=line_name,
                    textangle=90,
                    showarrow=False,
                    yref='paper',
                    bgcolor='white',
                    bordercolor=color,
                    font=dict(size=16, color=color),
                    borderwidth=1
                )
        
        # Set layout
        fig.update_layout(
            title=f"JWST Spectrum - z={self.z:.3f} | χ² Score: {self.calculate_score():.2f} (smaller is better)",
            xaxis_title="Wavelength [μm]",
            yaxis_title="Flux [10⁻¹⁸ erg/s/cm²/Å]",
            xaxis=dict(range=[0.5, 5.5]),
            template="plotly_white",
            height=600,
            legend=dict(x=0.02, y=0.98),
            title_font_size=16
        )
        
        # Set y-axis limits based on target type
        if self.target == 'GSz14':
            fig.update_yaxes(range=[-0.00025, 0.01])
        elif self.target == 'gnz11':
            fig.update_yaxes(range=[-0.01, 0.04])
        elif self.target == 'low_snr':
            fig.update_yaxes(range=[-0.01, 0.025])
        
        return fig
    
    def create_sfh_plot(self):
        """Create the star formation history plot"""
        fig = go.Figure()
        
        try:
            age_universe, sfr = self.calculate_sfh()
            
            # Plot the SFH
            fig.add_trace(go.Scatter(
                x=age_universe,
                y=sfr,
                mode='lines',
                line=dict(color='steelblue', width=2),
                fill='tozeroy',
                fillcolor='rgba(70, 130, 180, 0.3)',
                name='SFR',
                showlegend=False
            ))
            
            # Mark current age of universe
            age_now = self.get_universe_age(self.z)
            fig.add_vline(
                x=age_now,
                line=dict(color='red', dash='dash', width=2),
                annotation_text="Now",
                annotation_position="top"
            )
            
            # Set layout
            fig.update_layout(
                title="Star Formation History",
                xaxis_title="Age of Universe [Gyr]",
                yaxis_title="SFR [M☉/yr]",
                template="plotly_white",
                height=600,
                title_font_size=14,
                margin=dict(l=60, r=20, t=60, b=60)
            )
            
            fig.update_yaxes(rangemode='tozero')
            
        except Exception as e:
            print(f"Error creating SFH plot: {e}")
            fig.add_annotation(
                text="Error calculating SFH",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14, color="red")
            )
        
        return fig
    
    def run_server(self, debug=True, port=8051):
        """Run the Dash server"""
        self.app.run(debug=debug, port=port)