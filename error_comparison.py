'''
Plot error comparisions using both false negative and false positive classifications. Compares CNA, CSP and  BiDef
as detailed in the  paper/thesis.

'''
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import collections as c
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
#filenames=['cna300fcc','bidef300fcc','cna400fcc','bidef400fcc','cna500fcc','bidef500fcc','cna600fcc','bidef600fcc']
#filenames=['vacancies_ovitos.out']
filenames=['1022vacerr_ovitos.out']


error=c.namedtuple('error', ['fn','fp','e'])

#temps={filenames[1]:300,filenames[3]:400,filenames[5]:500,filenames[7]:600}
frames=1000
#total_atoms=16820
total_atoms=33515

def get_noise(name,error):
	df=pd.read_csv(name,delim_whitespace=True)

	noise=[]
	#cull down to only the relevant pieces, so this doesnt take forever. get both false positive error and false negative error
	#for bidef
	bidef=error(fn=df[(df['VID']==10)&(df['E']<=11)&(df['VE']>0)],fp=df[(df['VID']==15)&(df['E']==12)&(df['VE']==0)],e=[])
	#for CNA -- remember to eliminate surface sites (coordination should be higher than 8!)
	cna=error(fn=df[(df['CNA']==1)&(df['E']<=11)&(df['E']>8)&(df['CNAVE']>0)],fp=df[(df['CNA']==0)&(df['E']==12)&(df['E']>8)&(df['CNAVE']==0)],e=[])
	# for CSP
	csp=error(fn=df[(df['CSP']<5)&(df['E']<=11)&(df['E']>8)&(df['CSPVE']>0)],fp=df[(df['CNA']>=5)&(df['E']==12)&(df['E']>8)&(df['CSPVE']==0)],e=[])
	
	comparisons=[bidef,cna,csp]

	for x in range(frames):
		for comp in comparisons:		
			try:
				comp.e.append(comp.fn[comp.fn['Frame']==x].count()[0]+comp.fp[comp.fp['Frame']==x].count()[0])
			except:
				comp.e.append(0)
	return comparisons

fig, ax = plt.subplots()
N={}
for f in filenames:
	comps=get_noise(f,error)
	pn1=np.array(comps[0].e)/float(total_atoms)*100
	pn2=np.array(comps[1].e)/float(total_atoms)*100
	pn3=np.array(comps[2].e)/float(total_atoms)*100
	plt.plot(range(frames),pn1,label='BiDef')	
	plt.plot(range(frames),pn2,label='CNA')	
	plt.plot(range(frames),pn3,label='CSP')	
	#V=pd.rolling_mean(pd.DataFrame(N[f]),5).fillna(0)[0].values
	#if f.find('bidef')!=-1:
	#	tot=np.array(store)-np.array(N[f])
	#	tot=tot/float(total_atoms)*100
	#	plt.plot(range(frames),pd.rolling_mean(pd.DataFrame(tot),10).fillna(0)[0].values,label=str(temps[f])+' K',linewidth=3)
		#plt.plot(range(frames),tot,alpha=0.3)
	#else:
	#	store=N[f]

ax.minorticks_on()
plt.xlabel('Simulation Progression in 100s of steps (fs/100)',fontsize=24)
plt.ylabel('\% Misidentified Error', fontsize=24)
#plt.axis([xlo,xhi,ylo,yhi], fontsize=24)	
#plt.legend(bbox_to_anchor=(0.9,-0.15),prop={'size':12},ncol=4)
plt.legend(bbox_to_anchor=(1.0,0.5),prop={'size':16})
plt.savefig('vacancy_error.eps', edgecolor='none',bbox_inches='tight')
