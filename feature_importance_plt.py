# first load them into a dictionary with x
import matplotlib as mpl
import numpy as np
mpl.rcdefaults()
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['text.usetex'] = True
mpl.rcParams['axes.linewidth'] = 2 # set the value globally
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.minor.size'] = 5
mpl.rcParams['ytick.minor.width'] = 1

y=[(x[i])*len(x[i]) for i in x]

plt.boxplot(y)
plt.xticks([1,2,3,4,5,6],[3,4,5,6,7,8])

plt.xlabel('$2j$ max value',fontsize=24)
plt.xlabel('Standardized Feature Importances',fontsize=24)

plt.savefig('featureimportance.eps',bbox_inches='tight')
