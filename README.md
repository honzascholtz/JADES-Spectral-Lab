# JADES-Spectral-Lab

JADES-Spectral-Lab is a suite of of three outreach activities, designed to show public how we analyze JWST NIRCam and NIRSpec data. The three activities are:

1) Photometric lab - Fitting images to find their shapes

![alt text](https://github.com/honzascholtz/SED_Fitter/blob/main/Images/Photometry.png "Example of the UI")


2) The Redshift Lab - Determining the redshift of the galaxies from NIRSpec spectrocopy. 

![alt text](https://github.com/honzascholtz/SED_Fitter/blob/main/Images/Redshift.png "Example of the UI")


3) The Stellar Population Lab - Modelling the NIRspec spectra to find the masses, star-formation histories and interstellar gas properties in the galaxies. 

![alt text](https://github.com/honzascholtz/SED_Fitter/blob/main/Images/Stellar_pop.png "Example of the UI")

The Stellar Population Lab is a wrapper around Bagpipes to create a best fitting model to various JWST prism spectra using sliders controlling various parameters. 

![alt text](https://github.com/honzascholtz/SED_Fitter/blob/main/Images/UI.png "Example of the UI")

The score is actually just a reduced $chi2^{2}$. The aim of this game is to try to get as low score as possible - reduce the difference between your best fit model and the data. The top panel displays the data in black and the current model in a red line. I have also put labels for the various emission lines. 


# How to run 

In order to run the code, we need to install Bagpipes from Adam Carnall. You can find more info about Bagpipes <a href="https://bagpipes.readthedocs.io/en/latest/" target=_blank>here</a>.

requirements:
```python
dash==2.14.1

flask

dash-bootstrap-components==1.5.0

plotly==5.17.0

numpy==1.24.3

astropy==5.3.4

pandas==2.0.3

bagpipes
```

To run the entire lab suite: 
```python

python JADES_lab.py

```
This will run the flask server and show you the following landing page:

![alt text](https://github.com/honzascholtz/SED_Fitter/blob/main/Images/Landing_page.png "Example of the UI")

You can then run click on the different buttons to launch the different "Labs".

# Grid models updates

If you are a scientist, I would highly recommend to create a separate Conda/other enviroment to install this. This is mostly due to the need to ensure that the maximum redshift to estimate IGM is setup above z>14 to ensure we can fit GS-z14. if the redshift_max is less than the value, it will copy over the new config file. Once it is done, you will have to rerun the code. 
