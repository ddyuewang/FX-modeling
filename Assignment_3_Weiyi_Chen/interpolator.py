"""
Author: Weiyi Chen
Copyright: Copyright (C) 2015 Baruch College, Modeling and Market Making in FX - All Rights Reserved
Description: A cubic spline interpolation for implied volatility vs strike which has non-standard boundary conditions 
             to give more intuitive volatility extrapolation
Test: assig3_test.py
"""

import bisect
import math
from scipy import *
import scipy.stats as stats
import matplotlib.pyplot as plot

def _fitted_spline(strikes,vols,texp,extrap_fact):
    '''Function that takes the input strikes, vols, time to expiration, and
    extrapolation factor and constructs the spline parameters'''
    
    strike_min = strikes[0] *math.exp(-extrap_fact*vols[0] *math.sqrt(texp))
    strike_max = strikes[-1]*math.exp( extrap_fact*vols[-1]*math.sqrt(texp))
    
    all_strikes = [strike_min]+strikes+[strike_max]
    
    # there are six intervals for the spline where the function takes a cubic
    # form, like
    #
    #    y_i(x) = A[i] + B[i] x + C[i] x^2 + D[i] x^3
    #
    # for i=0 through 5. x[0] == strike_min, and x[6] == strike_max.
    #
    # The goal is to solve for those 24 parameters. The constraints: first,
    # that the volatilities match at the input points, for i=1->5:
    #
    #    y[i] = A[i] + B[i] x[i] + C[i] x[i]^2 + D[i] x[i]^3
    #
    # That gives us 5 equations of the 24 we need. Next, we know that the
    # value, slope, and second derivative must match at the end of each interval
    # for i=0 to 4, like
    #
    #    A[i] + B[i] x[i+1] + C[i] x[i+1]^2 + D[i] x[i+1]^3 = A[i+1] + B[i+1] x[i+1] + C[i+1] x[i+1]^2 + D[i+1] x[i+1]^3 
    #    B[i] + 2 C[i] x[i+1] + 3 D[i] x[i+1]^2 = B[i+1] + 2 C[i+1] x[i+1] + 3 D[i+1] x[i+1]^2
    #    2 C[i] + 6 D[i] x[i+1] = 2 C[i+1] + 6 D[i+1] x[i+1]
    #
    # That gives us an addition 3*5 = 15 equations, taking us to a total of 20
    # equations so far for the 24 parameters. 
    #
    # The final set of equations is where our extrapolation comes in. We require
    # that the slope and second derivative go to zero at x[0] and x[6], which are the
    # points we added beyond the marked points, based on the extrapolation factor parameter:
    #
    #    B[0] + 2 C[0] x[0] + 3 D[0] x[0]^2 = 0
    #    2 C[0] + 6 D[0] x[0] = 0
    #    B[5] + 2 C[5] x[6] + 3 D[5] x[6]^2 = 0
    #    2 C[5] + 6 D[5] x[6] = 0
    #
    # And that gives us an additional four equations, bring us to 24 equations for our
    # 24 parameters. So we can construct a linear system representing those equations
    # and invert it to solve for the parameter values.
    
    a = matrix(zeros((24,24)))
    b = matrix(zeros((24,1)))
    
    xs  = all_strikes
    x2s = [x*x for x in xs]
    x3s = [x*x*x for x in xs]
    
    # first five rows correspond to the five equations relating function values to the input vols
    
    for i in range(5):
        a[i,4*(i+1)] = 1
        a[i,4*(i+1)+1] = xs[i+1]
        a[i,4*(i+1)+2] = x2s[i+1]
        a[i,4*(i+1)+3] = x3s[i+1]
        
        b[i] = vols[i]
    
    # next require the value to match at the end of each interval for interval=0->4
    
    for i in range(5):
        a[i+5,4*i]       = 1
        a[i+5,4*i+1]     = xs[i+1]
        a[i+5,4*i+2]     = x2s[i+1]
        a[i+5,4*i+3]     = x3s[i+1]
        a[i+5,4*(i+1)]   = -1
        a[i+5,4*(i+1)+1] = -xs[i+1]
        a[i+5,4*(i+1)+2] = -x2s[i+1]
        a[i+5,4*(i+1)+3] = -x3s[i+1]
        
        b[i+5] = 0
    
    # next require the slopes to match
    
    for i in range(5):
        a[i+10,4*i+1] = 1
        a[i+10,4*i+2] = 2*xs[i+1]
        a[i+10,4*i+3] = 3*x2s[i+1]
        a[i+10,4*(i+1)+1] = -1
        a[i+10,4*(i+1)+2] = -2*xs[i+1]
        a[i+10,4*(i+1)+3] = -3*x2s[i+1]
        
        b[i+10] = 0
    
    # next the 2nd derivs
    
    for i in range(5):
        a[i+15,4*i+2] = 2
        a[i+15,4*i+3] = 6*xs[i+1]
        a[i+15,4*(i+1)+2] = -2
        a[i+15,4*(i+1)+3] = -6*xs[i+1]
        
        b[i+15] = 0
    
    # then the final four equations forcing 1st and 2nd derivs to go
    # to zero and the edge points we added in
    
    a[20,1] = 1
    a[20,2] = 2*xs[0]
    a[20,3] = 3*x2s[0]
    b[20]   = 0
    
    a[21,2] = 2
    a[21,3] = 6*xs[0]
    b[21]   = 0
    
    a[22,21] = 1
    a[22,22] = 2*xs[6]
    a[22,23] = 3*x2s[6]
    b[22]    = 0
    
    a[23,22] = 2
    a[23,23] = 6*xs[6]
    b[23]    = 0
    
    # then solve the equation
    
    sol = a.I*b
    
    cs_params = [sol[i,0] for i in range(24)]
    
    return all_strikes, cs_params

class vol_spline:
    '''Represents a cubic spline fit to five implied volatilities/strikes, with 
    boundary conditions set such that vols flatten out a certain number of standard
    deviations away from the outside strikes on either side'''
    
    def __init__(self,strikes,vols,texp,extrap_fact):
        '''Initializes the spline and calculates the spline parameters internally.
        
        strikes:     list of five strikes (must be monotonically increasing)
        vols:        implied volatilities for the strikes
        texp:        time to expiration
        extrap_fact: cubic spline extrapolation factor, defining number of standard
                     deviations after the outside strikes that vols turn flat
        '''
        
        # validate that there are exactly five strikes and five vols, and that strikes
        # are increasing
        
        if len(strikes)!=5: raise ValueError('There should be 5 strike values')
        if len(vols)!=5:    raise ValueError('There should be 5 volatility values')
            
        if strikes!=sorted(set(strikes)): raise ValueError('Strikes should be monotonically increasing')
        
        # remember the inputs
        
        self.strikes     = strikes
        self.vols        = vols
        self.texp        = texp
        self.extrap_fact = extrap_fact
        
        # calculate the spline parameters
        
        self.all_strikes, self.cs_params = _fitted_spline(strikes,vols,texp,extrap_fact)
    
    def volatility(self,strike):
        '''Interpolates a volatility for the given strike'''
        
        # if it's asking for a vol for a strike outside the region
        # where vols are flat, use the edges
        
        if strike<self.all_strikes[0]:
            strike = self.all_strikes[0]
        
        if strike>self.all_strikes[-1]:
            strike = self.all_strikes[-1]
        
        # interpolate a vol from the spline
        
        ind = bisect.bisect_left(self.strikes,strike)
        
        a = self.cs_params[4*ind]
        b = self.cs_params[4*ind+1]
        c = self.cs_params[4*ind+2]
        d = self.cs_params[4*ind+3]
        
        return a+b*strike+c*strike*strike+d*strike*strike*strike
    
def test():
    '''Test it out with some strikes & vols as per the assignment question'''
    
    # note the market inputs from the question
    
    spot = 1
    atm  = 0.08
    rr25 = 0.01
    rr10 = 0.018
    bf25 = 0.0025
    bf10 = 0.0080
    texp = 0.5
    
    # turn the RR and BF values into vols for the OTM strikes
    
    vol25c = atm+rr25/2.+bf25
    vol25p = atm-rr25/2.+bf25
    vol10c = atm+rr10/2.+bf10
    vol10p = atm-rr10/2.+bf10
    
    # figure out the strikes (note that forward==spot since rates are zero)
    
    atm_strike = spot*math.exp(atm*atm*texp/2.)
    strike25c  = spot*math.exp(vol25c*vol25c*texp/2.-vol25c*math.sqrt(texp)*stats.norm.ppf(0.25))
    strike25p  = spot*math.exp(vol25p*vol25p*texp/2.+vol25p*math.sqrt(texp)*stats.norm.ppf(0.25))
    strike10c  = spot*math.exp(vol10c*vol10c*texp/2.-vol10c*math.sqrt(texp)*stats.norm.ppf(0.10))
    strike10p  = spot*math.exp(vol10p*vol10p*texp/2.+vol10p*math.sqrt(texp)*stats.norm.ppf(0.10))
    
    strikes = [strike10p,strike25p,atm_strike,strike25c,strike10c]
    vols    = [vol10p,vol25p,atm,vol25c,vol10c]
    
    # for a bunch of extrapolation factor values, generate the spline and get vol vs strike values
    
    all_strikes, all_vols = [], []
    
    for extrap_fact in [0.01,10]:
        # generate the spline
        
        sp = vol_spline(strikes,vols,texp,extrap_fact)
        
        # figure out the range of strikes for the plot; we'll use 1-delta on either side. We'll
        # approximate the 1d vols with the 10d vols for the purpose of calculating the strikes
        
        strike_min = spot*math.exp(vol10p*vol10p*texp/2.+vol10p*math.sqrt(texp)*stats.norm.ppf(0.01))
        strike_max = spot*math.exp(vol10c*vol10c*texp/2.-vol10c*math.sqrt(texp)*stats.norm.ppf(0.01))
        
        nstrikes   = 100
        dstrike    = (strike_max-strike_min)/(nstrikes-1)
        
        plot_strikes, plot_vols = [], []
        for i in range(nstrikes):
            strike = strike_min+i*dstrike
            plot_strikes.append(strike)
            plot_vols.append(sp.volatility(strike)*100) # convert to % for display
        
        all_strikes.append(plot_strikes)
        all_vols.append(plot_vols)
    
    plot.plot(all_strikes[0],all_vols[0])
    plot.plot(all_strikes[1],all_vols[1])
    plot.show()

if __name__=="__main__":
    test()

