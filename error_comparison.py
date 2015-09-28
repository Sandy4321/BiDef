import pandas as pd
from matplotlib import pyplot as plt


filenames=['cna300bcc','bidef300bcc']
frames=200

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

N={}
for f in filenames:
	N[f],df=get_noise(f)
	plt.plot(range(frames),N[f],label=f)

plt.legend()
plt.show()
