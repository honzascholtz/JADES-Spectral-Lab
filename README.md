# JWST-Spectral-Lab

JWST-Spectral-Lab is an outreach activity designed to show students and the general public how SED fitting works and what we can learn from a spectrum. The entire thing is a wrapper around Bagpipes to create a best fitting model to various JWST prism spectra using sliders controlling various paramters. 

![alt text](https://github.com/honzascholtz/SED_Fitter/blob/main/Images/UI.png "Example of the UI")

The score is actually just a reduced $chi2^{2}$. The aim of this game is to try to get as low score as possible - reduce the difference between your best fit model and the data. The top panel displays the data in black and the current model in a red line. I have also put labels for the various emission lines. 

The bottom panel are the residuals between your current model and the data. This should help visualise where you can find the best place for an improvement. The red dashed lines show the zero level on the plot. 

The model we are trying to fit is quite simple, using exponential star formation history and nebular emission lines. The parameters of the model are as follows: 


1) stellar mass of the source

2) logU - Strength of the radiation field

3) Metallicity - amount of elements heavier than Hydrogen and Helium in the galaxy

4) 5)  Age, Tau - describe a pattern where the rate at which stars form in a galaxy decreases exponentially over time. This means that the star formation rate initially is high, but then declines, with the rate of decline becoming slower over time. The age sets when the star-formation episode started and Tau is the rate at which the star formation is declining 


6) Dust - amount of dust in the galaxy (for scientists: Calzetti 2000)


# How to run 

In order to run the code, we need to install Bagpipes from Adam Carnall. You can find more info about Bagpipes <a href="https://bagpipes.readthedocs.io/en/latest/" target=_blank>here</a>.

requirements:
```python
dash==2.14.1

dash-bootstrap-components==1.5.0

plotly==5.17.0

numpy==1.24.3

astropy==5.3.4

pandas==2.0.3

bagpipes-phy==1.0.0
```

To run the redshifting code: 
```python

python Redshift_app.py

```

To run the the full SED fitting visualizer:
```python

python Spectral_lab_app.py

```

# Grid models updates

If you are a scientist, I would highly recommend to create a separate Conda/other enviroment to install this. This is mostly due to the need to ensure that the maximum redshift to estimate IGM is setup above z>14 to ensure we can fit GS-z14. if the redshift_max is less than the value, it will copy over the new config file. Once it is done, you will have to rerun the code. 
