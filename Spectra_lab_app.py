
import Spectral_lab_dash as sld

# Create and run the app
if __name__ == '__main__':
    app = sld.JADES_spectral_lab()
    app.run_server(debug=True)