#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun April 27 21:23:34 2025

@author: jansen
"""

#importing modules            
import sys
from astropy.io import fits as pyfits
import Graph_setup as gst
import numpy as np

nan= float('nan')

pi= np.pi
e= np.e
c= 3.*10**8

fsz = gst.graph_format_official()

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, RangeSlider, TextBox


import matplotlib.pyplot as plt
import matplotlib
import numpy as np
    
pth = sys.path[0]



class JWST_Spectral_lab_redshift:
    '''Class to visualize SED'''
    def __init__(self, ):
        """
        Load the fit and the data
        Returns
        -------
        """

        #import Check_setup as grid_check
        #grid_check.check_setup()

        self.fig = plt.figure(figsize=(15.6, 8))
        self.fig.canvas.manager.set_window_title('Redshift')
        #gs = self.fig.add_gridspec(
        #    3, 3, height_ratios=(2.2,0.7,1.3), width_ratios=(1,1,1))

        self.ax0 = self.fig.add_subplot([0.1,0.2,0.85, 0.7])

        plt.subplots_adjust(hspace=0)

        with pyfits.open(pth+'/Data/10058975_prism_clear_v5.0_1D.fits') as hdu:
            self.data_wave = hdu['WAVELENGTH'].data*1e6
            self.data_flux = hdu['DATA'].data*1e-7
            self.data_error = hdu['ERR'].data*1e-7
            self.ztrue = 9.436
            
        self.ax0.plot(self.data_wave, self.data_flux, color='black', drawstyle='steps-mid')
        self.z = 1

        
        self.target = 'generic'
        

        self.ax0.set_ylabel("Brightness", fontsize=14)
        self.ax0.set_xlabel(" blue <---- ----> red ", fontsize=14)

        y0 = 0.1
        dy = 0.04
        x0 = 0.1

        # Redshift slider
        self.slider_z_ax = plt.axes([x0, y0, 0.8, 0.03])
        self.z_slider = Slider(self.slider_z_ax, 'redshift', valmin=1,
                              valmax=15,
                              valinit=self.z)
        self.z_slider.on_changed(self.slide_update)


        # Button to load SF943 data
        axbutton1 = plt.axes([0.1,0.9, 0.05,0.05])
        self.SF943 = Button(axbutton1, 'SF943', color='lightblue', hovercolor='lightgreen')
        self.SF943.on_clicked(self.SF943_load)

        # Buttons to load other QC galaxies
        axbutton2 = plt.axes([0.2,0.9, 0.05,0.05])
        self.QC_galaxy = Button(axbutton2, 'QC_galaxy', color='lightblue', hovercolor='lightgreen')
        self.QC_galaxy.on_clicked(self.QC_galaxy_load)

        # Buttons to load other SF galaxies
        axbutton3 = plt.axes([0.3,0.9, 0.05,0.05])
        self.SF_galaxy = Button(axbutton3, 'SF_galaxy', color='lightblue', hovercolor='lightgreen')
        self.SF_galaxy.on_clicked(self.SF_galaxy_load)

        # Buttons to load other GSz14
        axbutton4 = plt.axes([0.4,0.9, 0.05,0.05])
        self.GSz14 = Button(axbutton4, 'GSz14', color='lightblue', hovercolor='lightgreen')
        self.GSz14.on_clicked(self.GSz14_load)

        # Buttons to load other COS3018
        axbutton5 = plt.axes([0.5,0.9, 0.05,0.05])
        self.COS30 = Button(axbutton5, 'COS30', color='lightblue', hovercolor='lightgreen')
        self.COS30.on_clicked(self.COS30_load)

        # Buttons to load other COS3018
        axbutton6 = plt.axes([0.6,0.9, 0.05,0.05])
        self.SF2 = Button(axbutton6, 'SF2', color='lightblue', hovercolor='lightgreen')
        self.SF2.on_clicked(self.SF2_galaxy_load)

        # Buttons to load other PSB
        axbutton7 = plt.axes([0.7,0.9, 0.05,0.05])
        self.PSB = Button(axbutton7, 'PSB', color='lightblue', hovercolor='lightgreen')
        self.PSB.on_clicked(self.PSB_load)

        # Buttons to load other zhig
        axbutton8 = plt.axes([0.8,0.9, 0.05,0.05])
        self.zhig = Button(axbutton8, 'zhig', color='lightblue', hovercolor='lightgreen')
        self.zhig.on_clicked(self.zhig_load)

        # Buttons to load other zhig2
        axbutton9 = plt.axes([0.9,0.9, 0.05,0.05])
        self.zhig2 = Button(axbutton9, 'zhig2', color='lightblue', hovercolor='lightgreen')
        self.zhig2.on_clicked(self.zhig2_load)


        # Button to load show res data
        axbutton_res = plt.axes([0.1,0.15, 0.1,0.05])
        self.Show_button = Button(axbutton_res, 'Show Score', color='lightblue', hovercolor='lightgreen')
        self.Show_button.on_clicked(self.show_score)

        self.plot_general()
        plt.show()

    def submit_redshift(self, text): 
        ydata = float(text)
        self.z_slider.set_val(ydata)
        self.z = ydata

    def show_score(self, text): 
        self.score = (self.z-self.ztrue)/(1+self.ztrue)*3e5

        self.ax0.text(0.05, 0.75,'(Get to 0)\nscore: {:.4f}'.format(self.score),\
                      transform=self.ax0.transAxes)
        
        self.fig.canvas.draw()
        
    def SF943_load(self, event):
        with pyfits.open(pth+'/Data/10058975_prism_clear_v5.0_1D.fits') as hdu:
            self.data_wave = hdu['WAVELENGTH'].data*1e6
            self.data_flux = hdu['DATA'].data*1e-7
            self.data_error = hdu['ERR'].data*1e-7

            self.target = 'generic'
            self.ztrue = 9.436

            self.plot_general()
        
    def QC_galaxy_load(self, event):
        with pyfits.open(pth+'/Data/199773_prism_clear_v5.0_1D.fits') as hdu:
            self.data_wave = hdu['WAVELENGTH'].data*1e6
            self.data_flux = hdu['DATA'].data*1e-7
            self.data_error = hdu['ERR'].data*1e-7

            self.ztrue = 2.820

            self.target = 'generic'

            self.plot_general()

    def SF_galaxy_load(self, event):
        with pyfits.open(pth+'/Data/001882_prism_clear_v5.0_1D.fits') as hdu: #000110
            self.data_wave = hdu['WAVELENGTH'].data*1e6
            self.data_flux = hdu['DATA'].data*1e-7
            self.data_error = hdu['ERR'].data*1e-7

            self.ztrue = 5.4431

            self.target = 'generic'

            self.plot_general()
    
    def SF2_galaxy_load(self, event):
        with pyfits.open(pth+'/Data/001927_prism_clear_v5.0_1D.fits') as hdu: #000110
            self.data_wave = hdu['WAVELENGTH'].data*1e6
            self.data_flux = hdu['DATA'].data*1e-7
            self.data_error = hdu['ERR'].data*1e-7

            self.ztrue = 3.6591

            self.target = 'generic'

            self.plot_general()

    def GSz14_load(self, event):
        with pyfits.open(pth+'/Data/183348_prism_clear_v5.0_1D.fits') as hdu:
            self.data_wave = hdu['WAVELENGTH'].data*1e6
            self.data_flux = hdu['DATA'].data*1e-7
            self.data_error = hdu['ERR'].data*1e-7

            self.ztrue = 14.18

            self.target = 'GSz14'

            self.plot_general()

    def COS30_load(self, event):
        with pyfits.open(pth+'/Data/007437_prism_clear_v3.1_1D.fits') as hdu:
            self.data_wave = hdu['WAVELENGTH'].data*1e6
            self.data_flux = hdu['DATA'].data*1e-7
            self.data_error = hdu['ERR'].data*1e-7

            self.ztrue = 6.856

            self.data_wave = np.append(self.data_wave, np.linspace(5.32,5.5, 32))
            self.data_flux = np.append(self.data_flux, np.zeros(32))
            self.data_error = np.append(self.data_error, np.ones(32)*0.001e-18)
                                       

            self.plot_general()
            self.target = 'generic'

    def PSB_load(self, event):
        with pyfits.open(pth+'/Data/023286_prism_clear_v5.1_1D.fits') as hdu:
            self.data_wave = hdu['WAVELENGTH'].data*1e6
            self.data_flux = hdu['DATA'].data*1e-7
            self.data_error = hdu['ERR'].data*1e-7

            self.ztrue = 1.781

            self.plot_general()
            self.target = 'generic'

    def zhig_load(self, event):
        with pyfits.open(pth+'/Data/066585_prism_clear_v5.1_1D.fits') as hdu:
            self.data_wave = hdu['WAVELENGTH'].data*1e6
            self.data_flux = hdu['DATA'].data*1e-7
            self.data_error = hdu['ERR'].data*1e-7

            self.ztrue = 7.1404
            self.target = 'low_snr'
            self.plot_general()
            

    def zhig2_load(self, event):
        with pyfits.open(pth+'/Data/003991_prism_clear_v5.1_1D.fits') as hdu:
            self.data_wave = hdu['WAVELENGTH'].data*1e6
            self.data_flux = hdu['DATA'].data*1e-7
            self.data_error = hdu['ERR'].data*1e-7

            self.ztrue = 10.603
            self.target = 'gnz11'

            self.plot_general()
            

    def slide_update(self,val):
        self.z = self.z_slider.val
        
        self.plot_general()
    

       
    def plot_general(self, event=None):
        self.ax0.clear()       
        
        self.ax0.plot(self.data_wave, self.data_flux/1e-18, color='black', drawstyle='steps-mid')
        self.ax0.fill_between(self.data_wave,  (self.data_flux-self.data_error)/1e-18,  (self.data_flux+self.data_error)/1e-18, \
                                  color='lightgrey', alpha=0.5, step='mid')

        self.ax0.plot(self.data_wave, self.data_flux/1e-18, color='black', drawstyle='steps-mid')

        self.ax0.set_ylabel("Brightness", fontsize=14)
        self.ax0.set_xlabel(" blue <---- ----> red ", fontsize=14)
            
        self.ax0.set_xlim(0.5, 5.3)

        #self.score = np.nansum((self.data_flux- self.model.spectrum[:, 1])**2/self.data_error**2)/(len(self.data_flux)-6)

        #self.ax0.text(0.05, 0.75,'(Smaller is better)\nscore: {:.2f}'.format(self.score),\
        #                transform = self.ax0.transAxes, bbox=dict(boxstyle='Round,pad=0.01', facecolor='white',
        #                    alpha=1.0, edgecolor='none'))

        self.labels_eml()
        if self.target == 'GSz14':
            self.ax0.set_ylim(-0.00025, 0.01)

        if self.target == 'gnz11':
            self.ax0.set_ylim(-0.01, 0.04)

        if self.target == 'low_snr':
            self.ax0.set_ylim(-0.01, 0.025)
        
        self.fig.canvas.draw()
  

    def labels_eml(self,):
        emlines = {
            r'C$^{++}$'   : ( 1907.,  0.00,  0.95),
            r'Mg$^{+}$'    : ( 2797.,  0.00,  0.95),
            r'[O$^{+}$]'   : ( 3728.,  -0.02,  0.95),
            r'[Ne$^{++}$]' : ( 3869.860, -0.00, 0.95),
            'H$\\delta$'         : ( 4102.860, -0.00, 0.95),
            'H$\\gamma$'          : ( 4341.647,  0.025, 0.95),
            'H$\\beta$'           : ( 4862.647, -0.03, 0.9),
            r'[O$^{++}$]'  : ( np.array((4960.0,5008.0)),  0.03,  0.95),
            r'[O$^{0}$]'    : ( np.array((6302.0,6365.0)),  -0.03,  0.95),
            'Na'       : ( 5891.583,  0.00,  0.95),
            'H$\\alpha$'          : (  6564.522, -0.00, 0.95),
            r'[S$^{+}$]'   : ( 6725,  0.03,  0.9),
            r'[S$^{++}$]'  : ( np.array((9070.0,9535.0)),  0.0,  0.95),
            'HeI'     : ( 10832.1, -0.03, 0.95),
            'Pa$\\gamma$'         : ( 10940.978,  0.03, 0.95),
            r'[Fe$^{+}$]'  : ( 12570.200, -0.015, 0.50),
            'Pa$\\beta$'          : ( 12821.432,  0.015, 0.95),
            'Pa$\\alpha$'          : ( 18755.80357,  0.015, 0.95),
        }
        n_lines = len(emlines)+1
        cmap = matplotlib.colormaps['nipy_spectral']
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=n_lines)
        transMix = matplotlib.transforms.blended_transform_factory(
            self.ax0.transData, self.ax0.transAxes)

        for i,(line,lineprop) in enumerate(emlines.items()):
            waves, offset, y_position = lineprop
            waves = np.atleast_1d(waves)
            waves= np.array(waves, dtype=float)
            waves *= (1+self.z)/1.e4
            wave = np.mean(waves)
            #if not 1.001*self.data_wave<wave<self.data_wave[-1]*0.999: continue
            if not self.ax0.get_xlim()[0]*1.001<wave<self.ax0.get_xlim()[1]*0.999: continue
        
            color = cmap(norm(float(i)))
            where_line = np.argmin(np.abs(self.data_wave-wave))
            where_line = slice(where_line-5, where_line+6, 1)
            #data_at_line = (
            #    np.min(spec1d[where_line]) if pre_dash
            #    else np.max(spec1d[where_line])
            #)
            va = 'center'
            if y_position<0.05: va='bottom'
            if y_position>0.90: va='top'
            
            line = line
            for w in waves:
                    self.ax0.axvline(w, color=color, lw=1.5, alpha=0.5, ls='--', zorder=0)
                            
            self.ax0.text(
                wave+offset, y_position, line,
                color=color, transform=transMix, va=va, ha='center',
                fontsize=12,
                rotation='vertical',
                bbox=dict(boxstyle='Round,pad=0.01', facecolor='white',
                            alpha=1.0, edgecolor='none'),
                zorder=99,
                )        