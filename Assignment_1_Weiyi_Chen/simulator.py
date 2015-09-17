'''
Copyright:   Copyright (C) 2015 Baruch College, FX Modeling Course
Author:      Weiyi Chen
Description: Toy simulation algorithm to simulate electronic hedging - 

    1. At start of time step, generate a uniform random number in (0,1) and check if it's less than the trade probability. 
       If so, randomly generate the sign of the trade and update the position by +/- one unit. Get paid half the client bid/ask
    2. Check whether net position is outside the delta limit (+ or -). If so, pay half the dealer bid/ask on the whole position 
       to hedge to zero.
    3. Advance the spot by one step in the simulation. PNL = position at start of time step * change in spot over time step
    4. Go back to 1, keep looping until the end of the simulation happens.
    5. Redo the whole thing for a bunch of different simulation runs.
    6. Look at PNL distribution across the runs.

Test: Assign1_test.py
'''

# python import

from lazy import lazy
from multiprocessing import Pool

# 3rd party imports

from numpy import *


class Simulator(object):
    ''' Monte Carlo Simulator '''
    def __init__(self):
        super(Simulator, self).__init__()

    @lazy
    def Vol(self):
        ''' Volatility is 10%/year; when converted between seconds and years, assume 260 (trading) days per year.'''
        return 0.1 * sqrt(1 / 260.)

    @lazy
    def Lambda(self):
        ''' Poisson frequency for client trade arrival is 1 trade/second '''
        return 60 * 60 * 24

    @lazy
    def SpreadClient(self):
        ''' Bid/ask spread for client trades is 1bp '''
        return 1e-4

    @lazy
    def SpreadDealer(self):
        ''' Bid/ask spread for inter-dealer hedge trades is 2bp '''
        return 2e-4

    @lazy
    def FullHedge(self):
        ''' True means "hedge to zero position", otherwise means "hedge to delta limit" '''
        return True

    @lazy
    def DeltaLimit(self):
        ''' A delta limit of 3 units before the algorithm executes a hedge in the inter-dealer market.  '''
        return 3.

    @lazy
    def TimeStep(self):
        ''' a time step delta t equal to 0.1/lambda '''
        return 0.1 / self.Lambda

    @lazy
    def NSteps(self):
        ''' Use 500 time steps '''
        return 500

    @lazy
    def NRuns(self):
        ''' Number of monte carlo simulation runs to give sufficient convergence '''
        return 10000

    @lazy
    def TradingProb(self):
        ''' Trading probability '''
        return 1 - exp(-self.Lambda * self.TimeStep)

    @lazy
    def Seed(self):
        ''' Seed for random generators '''
        return 100

    @lazy
    def Parallel(self):
        return False

    def simulate(self):
        ''' Monte carlo simulation and change statistics on PNL '''

        if self.Parallel:
            random.seed(self.Seed)

        # Spot starts at 1
        spots = ones(self.NRuns)
        positions = zeros(self.NRuns)
        pnls = zeros(self.NRuns)
        
        for step in range(self.NSteps):
            # random numbers generators
            normals = random.normal(0, sqrt(self.TimeStep), self.NRuns)
            uniforms = random.uniform(0, 1, self.NRuns)
            binormails = random.binomial(1, 0.5, self.NRuns) * 2 - 1
            
            # check if there are client trades
            indicators = less(uniforms, self.TradingProb)
            positions += indicators * binormails
            pnls += ones(self.NRuns) * indicators * self.SpreadClient * spots / 2.
            
            # check if there are hedge trades
            if self.FullHedge == True:
                indicators = logical_or(less_equal(positions, -self.DeltaLimit), greater_equal(positions, self.DeltaLimit))
                pnls -= absolute(positions) * indicators * self.SpreadDealer * spots / 2.
                positions -= positions*indicators

            else:
                # Cases pos > delta_lim
                indicators  = greater(positions, self.DeltaLimit)
                pnls -= (positions - self.DeltaLimit) * indicators * self.SpreadDealer * spots/2.
                positions = positions * logical_not(indicators) + ones(self.NRuns) * indicators * self.DeltaLimit
                
                # Cases pos < -delta_lim
                indicators  = less(positions, -self.DeltaLimit)
                pnls -= (-self.DeltaLimit - positions) * indicators * self.SpreadDealer * spots/2.
                positions = positions * logical_not(indicators) + ones(self.NRuns) * indicators * (-self.DeltaLimit)
            
            dspots = self.Vol * spots * normals
            pnls += positions * dspots
            spots += dspots

            self.PNLMean = pnls.mean()
            self.PNLStdDev = pnls.std()
            self.SharpeRatio = self.PNLMean / self.PNLStdDev

    @lazy
    def SharpeRatio(self):
        ''' Sharpe Ratio of the PNL '''
        return 0.

    @lazy
    def PNLMean(self):
        ''' PNL Mean '''
        return 0.

    @lazy
    def PNLStdDev(self):
        ''' PNL standard deviation '''
        return 0.

    def __str__(self):
        ''' '''
        print('\tPNL Sharpe ratio:\t', self.SharpeRatio)
        print('\tPNL Mean:\t\t', self.PNLMean)
        print('\tPNL Std dev:\t\t', self.PNLStdDev)
        return ''
