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
    ''' A cubic spliner fit to five implied volatilities/strikes, with boundary conditions set such that vols flatten 
    out a certain number of standard deviations away from the outside strikes on either side '''
    
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
        '''Construct the spline parameters'''
        
        a, b = matrix(zeros((24,24))), matrix(zeros((24,1)))
        
        xs  = self.AllStrikes
        x2s, x3s = [x**2 for x in xs], [x**3 for x in xs]
        
        for i in range(5):
            # given points
            a[i,4*(i+1)],   a[i,4*(i+1)+1],   a[i,4*(i+1)+2],   a[i,4*(i+1)+3]    =  1,  xs[i+1],  x2s[i+1],  x3s[i+1]
            b[i] = self.Vols[i]

            # original function edges
            a[i+5,4*i],     a[i+5,4*i+1],     a[i+5,4*i+2],     a[i+5,4*i+3]      =  1,  xs[i+1],  x2s[i+1],  x3s[i+1]   
            a[i+5,4*(i+1)], a[i+5,4*(i+1)+1], a[i+5,4*(i+1)+2], a[i+5,4*(i+1)+3]  = -1, -xs[i+1], -x2s[i+1], -x3s[i+1]
            b[i+5] = 0
            
            # 1st derivative edges
            a[i+10,4*i+1],     a[i+10,4*i+2],     a[i+10,4*i+3]     =  1,  2*xs[i+1],  3*x2s[i+1]
            a[i+10,4*(i+1)+1], a[i+10,4*(i+1)+2], a[i+10,4*(i+1)+3] = -1, -2*xs[i+1], -3*x2s[i+1]
            b[i+10] = 0
            
            # 2nd derivative edges
            a[i+15,4*i+2], a[i+15,4*i+3], a[i+15,4*(i+1)+2], a[i+15,4*(i+1)+3] = 2, 6*xs[i+1], -2, -6*xs[i+1]
            b[i+15] = 0

        # 1st and 2nd derivatives go zero and other edge points
        
        a[20,1], a[20,2], a[20,3], b[20] = 1, 2*xs[0], 3*x2s[0], 0        
        a[21,2], a[21,3], b[21] = 2, 6*xs[0], 0
        a[22,21], a[22,22], a[22,23], b[22] = 1, 2*xs[6], 3*x2s[6], 0        
        a[23,22], a[23,23], b[23] = 2, 6*xs[6], 0
 
        # Solve
        sol = a.I*b
        cs_params = [sol[i,0] for i in range(24)]
        return cs_params

    def volatility(self, strike):
        '''Interpolates a volatility for the given strike'''
        
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