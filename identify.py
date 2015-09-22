#this script identifies the entries coming from a lammps dump -- any lammps dump using the specified trained ML classifier
import pickle
import pandas as pd
from lammps import lammps
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn import svm
import numpy as np

def norm (s, m) : return lambda x : (x - m) / s


def remove_surface_atoms (df):
	ind=df[df['x'].max()-df['x']>5].index.values
	ind=np.intersect1d(ind,df[df['x']-df['x'].min()>5].index.values)
	ind=np.intersect1d(ind,df[df['y'].max()-df['y']>5].index.values)
	ind=np.intersect1d(ind,df[df['y']-df['y'].min()>5].index.values)
	#ind=np.intersect1d(ind,df[df['z'].max()-df['z']>5].index.values)
	#ind=np.intersect1d(ind,df[df['z']-df['z'].min()>5].index.values)
	return ind


def remove_surface_atoms_wZ (df):
	ind=df[df['x'].max()-df['x']>5].index.values
	ind=np.intersect1d(ind,df[df['x']-df['x'].min()>5].index.values)
	ind=np.intersect1d(ind,df[df['y'].max()-df['y']>5].index.values)
	ind=np.intersect1d(ind,df[df['y']-df['y'].min()>5].index.values)
	ind=np.intersect1d(ind,df[df['z'].max()-df['z']>3].index.values)
	ind=np.intersect1d(ind,df[df['z']-df['z'].min()>3].index.values)
	return ind


def initialize_classification_env():
	print "\n loading in classifiers..."
	classdict=pickle.load(open("classdict.p","rb"))
	clfdict={}
	for k in classdict.keys():
		clfdict[k]=joblib.load('./pickled/'+classdict[k]+'.pkl')
	return clfdict


def initialize_full_classifier():
	print "\n loading in classifier..."
	return joblib.load('./pickled/clftot.pkl')

def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None

def nab_and_format_bispec(fn,clfdict,expected_struct,clffull,get_cats=False,need_PCA=False,full_clf=True):
	df=pd.read_csv(fn,skiprows=8,delim_whitespace=True,low_memory=False)
	# get rid of the bogus first columns
	cols=df.columns[2::]
	df.drop(df.columns[-2::],1,inplace=True)
	df.columns=cols
	
	tdata=df

	#Need the stats for each bispec component
	print "\nre-normalizing data..."
	for x in tdata.columns[5::]:
		m=pickle.load(open( "./data_stats/"+str(x)+"_m.p", "rb"))
		s=pickle.load(open( "./data_stats/"+str(x)+"_s.p", "rb"))
		tdata[x]=tdata[x].map(norm(s,m))
	if need_PCA==True:
		# now we transform it using the previously trained PCA
		print "\nloading in PCA..."
		pca=pickle.load(open("pca.p","rb"))
		trans_values=pca.transform(tdata[tdata.columns[5::]].values)
	else:
		trans_values=tdata[tdata.columns[5::]].values

	if get_cats==True:
		KM=KMeans(n_clusters=2)
		print "\n Separating into " + str(KM.get_params()['n_clusters']) +" parts, and removing surface atoms."
		nonsurf=remove_surface_atoms(df)
		atom_cats=np.zeros(len(trans_values))
		atom_cats[nonsurf]=KM.fit_predict(trans_values[nonsurf])
		return df,atom_cats
	#make_output(fn,df,atom_cats)
	
	classdict=pickle.load(open("classdict.p","rb"))
	
	print "\n making prediction..."
	if full_clf==False:	
		predictions={}
		values={}
		for k in classdict.keys():
			#values[k]=clfdict[k].predict(trans_values)
			predictions[k]=clfdict[k].predict_proba(trans_values)
	
		out=np.zeros(len(trans_values))
		#need the maximum likelyhood defect (could be bulk if all are small)
		for x in range(len(trans_values)):
			poss_def=np.zeros(len(classdict))
			for k in classdict.keys():
				if k > 9:
					p=1
				else:
					p=0
				if predictions[k][x][p] >= 0.60 and classdict[k].find(expected_struct)!=-1:
					poss_def[k]=predictions[k][x][p]
			if sum(poss_def) > 0:
				out[x]=np.array(poss_def).argmax()
			else:
				out[x]=-1
	else: #default full clf structure
		probs=clffull.predict_proba(trans_values)
		out=[]
 		for p in probs:
			# if the difference is less than 5%, and one option is a defect, take the defect!
			mp = max(p)
			'''			
			if max(p) - second_largest(p) <= 0.05 and (np.array(p).argmax() == 9 or np.array(p).argmax() == 10): 
				mp = (np.array(p)==second_largest(p)).argmax()
			else:
				mp = np.array(p).argmax()
			'''
			out.append(mp)

		predictions=probs
	return df,out,trans_values,tdata,predictions,classdict

def get_former_dump(lat,output,datname,step,bounds,boundary):

	return 'units metal\n'+'boundary '+str(boundary)+'\nregion 		sim block '+str(bounds[0])+' '+str(bounds[1])+' '+str(bounds[2])+' '+str(bounds[3])+' '+str(bounds[4])+' '+str(bounds[5])+'\n'+'create_box 1 sim\n'+'read_dump '+str(datname)+' '+str(step)+' x y z add yes box yes'+'\n'+'mass 1 1.0\n'+'pair_style lj/cut '+str(2*lat)+'\n'+\
		'pair_coeff * * 1 1\nneighbor        0.5 bin\nneigh_modify    every 50 delay 0 check yes\ntimestep        0.001\nlog equib.out append\ncompute vb all sna/atom 1.0 0.99 8 '+str(lat)+' 1.0 diagonal 3\n'+'dump myDump all custom 1 '+output+' id type x y z c_vb[1] c_vb[2] c_vb[3] c_vb[4] c_vb[5] c_vb[6] c_vb[7] c_vb[8] c_vb[9] c_vb[10] c_vb[11] c_vb[12] c_vb[13] c_vb[14] c_vb[15] c_vb[16] c_vb[17] c_vb[18] c_vb[19] c_vb[20] c_vb[21] c_vb[22] c_vb[23] c_vb[24] c_vb[25] c_vb[26] c_vb[27] c_vb[28] c_vb[29] c_vb[30] c_vb[31] c_vb[32] c_vb[33] c_vb[34] c_vb[35] c_vb[36] c_vb[37] c_vb[38] c_vb[39] c_vb[40] c_vb[41] c_vb[42] c_vb[43] c_vb[44] c_vb[45] c_vb[46] c_vb[47] c_vb[48] c_vb[49] c_vb[50] c_vb[51] c_vb[52] c_vb[53] c_vb[54] c_vb[55]\n'

def get_boxesf(scale,f):
	b=['x scale '+str(scale*f), ' y scale '+str(scale*f), ' z scale '+str(scale*f)]
	return str(b[0])+' '+str(b[1])+' '+str(b[2])

def make_into_bispec(fn,latt,se,clfdict,clf,expected_struct='fcc',frame=0,bounds='p p p'):
#turns a lammps dump into bispec components
# we convert everything to Cu lattice parameter scaling for now
	b=[8.0960598,71.3280029,0.0,22.0,-51.1679993,57.1920013]
	template=get_former_dump(se.lattice,'temporaryfile.out',fn,frame,b,bounds) 
	with open('temp.in', 'w') as f:
		f.write(template)
		f.write('\nchange_box all '+get_boxesf(1.0,se.lattice/latt)+' boundary '+bounds+' remap units box\n')
		f.write('\nrun 0 post no\n')
	
	lmp=lammps()						
	lmp.file('temp.in')
	lmp.close()
	dat,output,trans,tdata,preds,cd=nab_and_format_bispec('temporaryfile.out',clfdict,expected_struct,clf)
	make_output("temporaryfile.out",dat,output)
	return dat,output,trans,tdata,preds,cd

def make_output(fn,df,out):

	with open(fn) as myfile:
	    head = [next(myfile) for x in xrange(8)]

	with open('output', 'w') as f:
		for h in head:    
			f.write(h) 
		f.write('ITEM: ATOMS id type x y z ') 
		f.write('vid')
		f.write('\n')
		for x in range(len(df)):
			for v in df[x:x+1].values[0][:5]:
				f.write(str(v)+' ')
			f.write(str(out[x]))
			f.write('\n')
	pass

if __name__ == "__main__":

	class simulation_env:
	     def __init__(self):
		 self.sampling_rate = 0.025#how often we select for keeps
		 self.num_comp = 5#number of PCA components that the bi components are reduced to
		 self.sample_size=3000#size of samples from each classification when training
		 self.thermal=10 #number of thermal perturbations for each config.
		 self.lattice=3.597 #lattice size
		 self.indicies=[[[1,1,1],[-1,1,0],[-1,-1,2]], [[0,1,0],[0,0,1],[1,0,0]]]#crystal orientations
		 self.interstitial={'fcc':[[0.25,0.25,0.25],[0.5,0.0,0.0]],'bcc':[[0.5,0.25,0.0],[0.5,0.5,0.0]]}# interstitial sites for fcc, bcc (tet then oct)
		 self.structs=['fcc','bcc'] #structures considered
		 self.bounds=['p p s', 'p s p', 's p p'] #boundaries
		 self.scales=[1+x for x in np.arange(0.01,0.11,0.05)] #scales for straining the simulation cell
		 self.defect_names=['fccfull.lmp','fccpartial.lmp','bcc_disloc.lmp']
		 self.flocs={self.defect_names[0]:'fccfull_temp.structures',self.defect_names[2]:'BCC_temp.structures',self.defect_names[1]:'temp.structures'} #dislocation prototype file structural analysis files
	
	#f has the each of the files to consider, tdict relates the file name to the defect name
		#condit contains the conditions for each simulation, surf or no surf, CNA or CSP [0] spot
		# is the description and the [1] spot is the value needed

	se=simulation_env()

	#clfdict=initialize_classification_env()
	clfdict={}	
	clftot=initialize_full_classifier()
	#dat,output,trans,tdata,preds,cd=nab_and_format_bispec("keep_octahedralinterstitial_bcc4012",clfdict)
	#dat,output=nab_and_format_bispec("dislocationbcc_disloc.lmp6538",get_cats=True)	
	#make_output("keep_octahedralinterstitial_bcc4012",dat,output)
	dat,output,trans,tdata,preds,cd=make_into_bispec('dump.Cu.Cu_225_10000',se.lattice,se,clfdict,clftot,frame=600,bounds='s p p')

