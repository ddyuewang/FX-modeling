'''
Author:      Weiyi Chen
Copyright:   Copyright (C) 2015 Baruch College, Modeling and Market Making in Forex Exchange
Description: Test for hedger.py
'''

from hedger import Hedger

def hedger_test():
    for tenor in [0.1, 0.25, 0.5, 0.75, 1, 2]:
        print('Value of Tenor:', tenor)
        for i, hedingStrategy in enumerate(['Non-hedging', 'Triangle-hedging', 'Factor-hedging']):
            h = Hedger()
            h.Tenor = tenor
            h.HedgingStrategy = i
            print('\t', hedingStrategy,'strategy PNL std:', h.PNL_std * 1e4)

if __name__=="__main__":
    hedger_test()