'''
Copyright:   Copyright (C) 2015 Baruch College, FX Modeling Course
Author:      Weiyi Chen
Description: Test script for simulator.py
Run:         python3 test_simulator.py
'''

# 3rd party imports

from numpy import *
from multiprocessing import Pool

# local imports

from simulator import Simulator

def test_simulator():
    '''test function on Simulator'''

    # First approach: full hedge
    s = Simulator()
    s.simulate()
    print('Full Hedge approach - \n', s)

    # Second approach: partial hedge

    s.FullHedge = False
    res = s.simulate()
    print('Partial Hedge approach - \n', s)

def f(n):
    s = Simulator()
    s.Parallel = True
    s.Seed = n
    s.simulate()
    return s.NRuns, s.PNLMean, s.PNLStdDev

def test_simulators():
    ''' parallel version to call Simulator '''

    p = Pool(100)
    res = p.map(f, range(100))
    
    NRuns = sum([n for n, m, std in res])
    PNLMean = mean([n*m for n, m, std in res]) / NRuns
    PNLStdDev = mean([n*std for n, m, std in res]) / NRuns
    SharpeRatio = PNLMean / PNLStdDev

    print(SharpeRatio)

if __name__=="__main__":
    test_simulator()