#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JWST Labs - Flask Integration with Multiple Dash Apps
Three separate Dash apps integrated with Flask

@author: jansen (converted to Flask)
"""

from flask import Flask,  render_template
import dash_bootstrap_components as dbc

import Phot_flask as phot
import redshift_flask as redshift
import stellar_flask as stellar


# Create Flask server
server = Flask(__name__)


# ============================================================================
# FLASK ROUTES
# ============================================================================
@server.route('/')
def index():
    """Home page with links to all three labs"""
    return render_template('index.html')

@server.route('/api/health')
def health():
    """API health check endpoint"""
    return {'status': 'healthy', 'message': 'Flask + 3 Dash apps running'}


# ============================================================================
# INITIALIZE ALL THREE APPS
# ============================================================================
photometry_app = phot.JADES_photo_lab(server, url_base_pathname='/photometry/')
redshift_app = redshift.Redshift_lab(server, url_base_pathname='/redshift/')
stellar_app = stellar.Stellar_pop_lab(server, url_base_pathname='/stellar-pop/')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("JWST Laboratory Suite - Multi-App Flask Server")
    print("="*60)
    print("\nStarting server on http://localhost:8051")
    print("\nAvailable endpoints:")
    print("  → Home:              http://localhost:8051/")
    print("  → Photometry Lab:    http://localhost:8051/photometry/")
    print("  → Redshift Lab:      http://localhost:8051/redshift/")
    print("  → Stellar Pop Lab:   http://localhost:8051/stellar-pop/")
    print("  → Health Check:      http://localhost:8051/api/health")
    print("\n" + "="*60 + "\n")
    
    server.run(debug=True, port=8051)