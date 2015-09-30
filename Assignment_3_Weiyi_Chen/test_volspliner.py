"""
Author: Weiyi Chen
Copyright: Copyright (C) 2015 Baruch College, Modeling and Market Making in FX - All Rights Reserved
Description: Test for VolSpliner
"""

from volspliner import VolSpliner
from scipy import *
import scipy.stats as stats
import matplotlib.pyplot as plot

def test():
    '''Test with VolSpliner'''
    
    for extrap_fact in [0.01,10]:
        
        # generate the spline and change the extrapolation factor
        
        sp = VolSpliner()
        sp.Extrap_fact = extrap_fact
        
        nstrikes   = 100
        dstrike    = (sp.StrikeMax-sp.StrikeMin)/(nstrikes-1)
        
        plot_strikes, plot_vols = [], []
        for i in range(nstrikes):
            strike = sp.StrikeMin + i * dstrike
            plot_strikes.append(strike)
            plot_vols.append(sp.volatility(strike))
        plot.plot(plot_strikes,plot_vols)
    plot.show()

if __name__=="__main__":
    test()
