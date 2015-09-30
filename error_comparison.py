import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


from matplotlib.pyplot import cm 
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
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








#filenames=['cna300bcc','bidef300bcc','cna400bcc','bidef400bcc','cna500bcc','bidef500bcc','cna600bcc','bidef600bcc']
filenames=['cna300fcc','bidef300fcc','cna400fcc','bidef400fcc','cna500fcc','bidef500fcc','cna600fcc','bidef600fcc']


temps={filenames[1]:300,filenames[3]:400,filenames[5]:500,filenames[7]:600}
frames=200
#total_atoms=16820
total_atoms=33620

def get_noise(name):
	df=pd.read_csv(name,delim_whitespace=True)

	noise=[]
	
	df=df[(df['E']<=2) & (df['E'] != 0)]

	g = df.groupby('Frame').groups	
	
	for x in range(frames):
		try:	
			noise.append(len(df.ix[g[x]]))
		except:
			noise.append(0)
	return noise,df

fig, ax = plt.subplots()
N={}
for f in filenames:
	N[f],df=get_noise(f)
	#V=pd.rolling_mean(pd.DataFrame(N[f]),5).fillna(0)[0].values
	if f.find('bidef')!=-1:
		tot=np.array(store)-np.array(N[f])
		tot=tot/float(total_atoms)*100
		plt.plot(range(frames),pd.rolling_mean(pd.DataFrame(tot),10).fillna(0)[0].values,label=str(temps[f])+' K',linewidth=3)
		#plt.plot(range(frames),tot,alpha=0.3)
	else:
		store=N[f]

ax.minorticks_on()
plt.xlabel('Simulation Progression in 100s of steps (fs/100)',fontsize=24)
plt.ylabel('\% Error Difference $(Err_{CNA}-Err_{BiDef})*100$', fontsize=24)
#plt.axis([xlo,xhi,ylo,yhi], fontsize=24)	
#plt.legend(bbox_to_anchor=(0.9,-0.15),prop={'size':12},ncol=4)
plt.legend(bbox_to_anchor=(0.25,1.0),prop={'size':16})
plt.savefig('fcc_error.eps', edgecolor='none',bbox_inches='tight')
