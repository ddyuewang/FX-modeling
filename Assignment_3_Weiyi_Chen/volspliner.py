"""
Author: Weiyi Chen
Copyright: Copyright (C) 2015 Baruch College, Modeling and Market Making in FX - All Rights Reserved
Description: A cubic spline interpolation for implied volatility vs strike which has non-standard boundary conditions 
             to give more intuitive volatility extrapolation
Test: interpolator_test.py
"""

# python imports
import bisect
import math

# 3rd party imports
from scipy import *
import scipy.stats as stats
from lazy import lazy

class VolSpliner:
    '''Represents a cubic spline fit to five implied volatilities/strikes, with 
    boundary conditions set such that vols flatten out a certain number of standard
    deviations away from the outside strikes on either side'''
    
    def __init__(self):
        super(VolSpliner, self).__init__()

    # Market Inputs

    @lazy
    def Spot(self):
        return 1

    @lazy
    def ATM(self):
        return .08

    @lazy
    def Rr25(self):
        return .01

    @lazy
    def Rr10(self):
        return .018

    @lazy
    def Bf25(self):
        return .0025

    @lazy
    def Bf10(self):
        return .0080

    @lazy
    def Texp(self):
        ''' time to expiration '''
        return .5

    @lazy
    def Extrap_fact(self):
        ''' 
        cubic spline extrapolation factor, defining number of standard
        deviations after the outside strikes that vols turn flat
        '''
        return None

    @lazy
    def ATMStrike(self):
        return self.Spot * exp(self.ATM**2 * self.Texp / 2.)

    @lazy
    def Strike25c(self):
        return self.Spot * exp(self.Vol25c**2 * self.Texp / 2. - self.Vol25c * sqrt(self.Texp) * stats.norm.ppf(0.25))

    @lazy
    def Strike25p(self):
        return self.Spot * exp(self.Vol25p**2 * self.Texp / 2. + self.Vol25p * sqrt(self.Texp) * stats.norm.ppf(0.25))

    @lazy
    def Strike10c(self):
        return self.Spot * exp(self.Vol10c**2 * self.Texp / 2. - self.Vol10c * sqrt(self.Texp) * stats.norm.ppf(0.10))

    @lazy
    def Strike10p(self):
        return self.Spot * exp(self.Vol10p**2 * self.Texp / 2. + self.Vol10p * sqrt(self.Texp) * stats.norm.ppf(0.10))

    @lazy
    def Strikes(self):
        ''' list of five strikes (must be monotonically increasing) '''
        return [self.Strike10p, self.Strike25p, self.ATMStrike, self.Strike25c, self.Strike10c]

    @lazy
    def Vol10p(self):
        return self.ATM - self.Rr10 / 2. + self.Bf10

    @lazy
    def Vol25p(self):
        return self.ATM - self.Rr25 / 2. + self.Bf25

    @lazy
    def Vol25c(self):
        return self.ATM + self.Rr25 / 2. + self.Bf25

    @lazy
    def Vol10c(self):
        return self.ATM + self.Rr10 / 2. + self.Bf10

    @lazy
    def Vols(self):
        ''' implied volatilities for the strikes '''
        return [self.Vol10p, self.Vol25p, self.ATM, self.Vol25c, self.Vol10c]

    @lazy
    def StrikeMin(self):
        return self.Strikes[0] * exp(-self.Extrap_fact * self.Vols[0] * sqrt(self.Texp))

    @lazy
    def StrikeMax(self):
        return self.Strikes[-1] * exp(self.Extrap_fact * self.Vols[-1] * sqrt(self.Texp))

    @lazy
    def AllStrikes(self):
        return [self.StrikeMin] + self.Strikes + [self.StrikeMax]

    @lazy
    def CSParams(self):
        '''Function that takes the input strikes, vols, time to expiration, and
        extrapolation factor and constructs the spline parameters'''
        
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
        
        xs  = self.AllStrikes
        x2s = [x*x for x in xs]
        x3s = [x*x*x for x in xs]
        
        # first five rows correspond to the five equations relating function values to the input vols
        
        for i in range(5):
            a[i,4*(i+1)] = 1
            a[i,4*(i+1)+1] = xs[i+1]
            a[i,4*(i+1)+2] = x2s[i+1]
            a[i,4*(i+1)+3] = x3s[i+1]
            
            b[i] = self.Vols[i]
        
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
        
        return cs_params

    def volatility(self, strike):
        '''Interpolates a volatility for the given strike'''
        
        # if it's asking for a vol for a strike outside the region
        # where vols are flat, use the edges
        
        if strike < self.AllStrikes[0]:
            strike = self.AllStrikes[0]
        
        if strike > self.AllStrikes[-1]:
            strike = self.AllStrikes[-1]
        
        # interpolate a vol from the spline
        
        ind = bisect.bisect_left(self.Strikes,strike)
        
        a = self.CSParams[4*ind]
        b = self.CSParams[4*ind+1]
        c = self.CSParams[4*ind+2]
        d = self.CSParams[4*ind+3]
        
        return a + b*strike + c*strike**2 + d*strike**3