
import Redshift_dash as rld

# Create and run the app
if __name__ == '__main__':
    app = rld.Redshift_dash()
    app.run_server(debug=True)