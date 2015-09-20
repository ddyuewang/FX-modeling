from volspliner import VolSpliner
from scipy import *
import scipy.stats as stats
import matplotlib.pyplot as plot

def test():
    '''Test it out with some strikes & vols as per the assignment question'''
    
    # for a bunch of extrapolation factor values, generate the spline and get vol vs strike values

    all_strikes, all_vols = [], []
    
    for extrap_fact in [0.01,10]:
        # generate the spline
        
        sp = VolSpliner()
        sp.Extrap_fact = extrap_fact
        
        # figure out the range of strikes for the plot; we'll use 1-delta on either side. We'll
        # approximate the 1d vols with the 10d vols for the purpose of calculating the strikes
        
        strike_min = sp.StrikeMin#sp.Spot * exp(sp.Vol10p**2 * sp.Texp / 2. + sp.Vol10p * sqrt(sp.Texp) * stats.norm.ppf(0.01))
        strike_max = sp.StrikeMax#sp.Spot * exp(sp.Vol10c**2 * sp.Texp / 2. - sp.Vol10c * sqrt(sp.Texp) * stats.norm.ppf(0.01))
        
        nstrikes   = 100
        dstrike    = (strike_max-strike_min)/(nstrikes-1)
        
        plot_strikes, plot_vols = [], []
        for i in range(nstrikes):
            strike = strike_min + i * dstrike
            plot_strikes.append(strike)
            plot_vols.append(sp.volatility(strike)*100) # convert to % for display
        
        all_strikes.append(plot_strikes)
        all_vols.append(plot_vols)
    
    plot.plot(all_strikes[0],all_vols[0])
    plot.plot(all_strikes[1],all_vols[1])
    plot.show()

if __name__=="__main__":
    test()
