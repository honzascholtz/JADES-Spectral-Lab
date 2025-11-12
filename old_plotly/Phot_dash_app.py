import Phot_dash as phd
# Create and run the app
if __name__ == '__main__':
    app = phd.JADES_photo_lab()
    app.run_server(debug=True)