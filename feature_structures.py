'''
mega-script used for generating defective features from templates. Deals  with straining the system and applying
thermal pertubations to the atoms. Incorporates each of the defects listed in the paper/thesis. Will need template
files provided in the same folder and a working lammps-python library for bi-spectrum coefficient generation.
'''
import random
import numpy as np
from lammps import lammps
import pandas as pd
import itertools
import subprocess
from template_generation import template_gen
from ovito_master import *
import time

def get_boxes(scale):
	b=['x scale '+str(scale), ' y scale '+str(scale), ' z scale '+str(scale)]
	return [b[0],b[1],b[2], b[0]+b[1],b[0]+b[2],b[1]+b[2],b[0]+b[1]+b[2]]

def get_boxesf(scale,f):
	b=['x scale '+str(scale*f), ' y scale '+str(scale*f), ' z scale '+str(scale*f)]
	return [b[0]+b[1]+b[2]]


# the lammps file templates need to be able to dynamically assign the correct number of vb spots, so the the B can be changed from 8 to W/E.
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)

# we need to nab the bispec components for every defect type, then merge them all into one dataframe, then standardize the coefficients, then perform the PCA

def nab_bispec_train(infile):
	df=pd.read_csv(infile,skiprows=8,delim_whitespace=True,low_memory=False)
	# get rid of the bogus first columns
	cols=df.columns[2::]
	df.drop(df.columns[-2::],1,inplace=True)
	df.columns=cols
	return df

#need one of every surface type and of each structure -- then need to apply the 
#prototype to EACH one of the systems

def make_surface_prototypes(se,alld,dic,lt):
	DOUT={}
	tdict={(se.structs[0],tuple(se.indicies[0][0])):'fcc_111',\
			(se.structs[0],tuple(se.indicies[0][1])):'fcc_110',\
			(se.structs[0],tuple(se.indicies[0][2])):'fcc_112',\
			(se.structs[0],tuple(se.indicies[1][2])):'fcc_100',\
			(se.structs[1],tuple(se.indicies[0][0])):'bcc_111',\
			(se.structs[1],tuple(se.indicies[0][1])):'bcc_110',\
			(se.structs[1],tuple(se.indicies[0][2])):'bcc_112',\
			(se.structs[1],tuple(se.indicies[1][2])):'bcc_100'}	
	counter=0
	for stru in se.structs:
			for i in se.indicies:
				for bound in se.bounds:
						if i[dic[bound]]==[0,1,0] or i[dic[bound]]==[0,0,1]:
							break

						lx=se.lattice*np.sqrt(i[0][0]**2+i[0][1]**2+i[0][2]**2)
						ly=se.lattice*np.sqrt(i[1][0]**2+i[1][1]**2+i[1][2]**2)
						lz=se.lattice*np.sqrt(i[2][0]**2+i[2][1]**2+i[2][2]**2)							
						
						fiz='tempstruct_'+str(counter)
						counter+=1
						template=lt.adjust_temp_athermal(se.lattice,lx,ly,lz,i[0],i[1],i[2],bound,stru,fiz)

						with open('temp.in', 'w') as f:
								f.write(template)
								f.write('\nrun 0 post no\n')
						lmp=lammps()
						try:							
							lmp.file('temp.in')
							temp=nab_bispec_train(fiz)
							alld=alld.append(temp)
						except:
							print "\nFAILURE TO RUN LAMMPS."																					
						lmp.close()
						#get the CNA parameters from ovito
						check=run_ovitos('CNA',fiz,fiz+'CNAOUT')
						lil_d={fiz+'CNAOUT':tdict[(stru,tuple(i[dic[bound]]))]}
						if stru=='fcc':
							lil_cond={fiz+'CNAOUT':['CNA',1]}
						elif stru=='bcc':
							lil_cond={fiz+'CNAOUT':['CNA',3]}
						DOUT[(stru,tuple(i[dic[bound]]))]=initialize_dislocation_descriptors(fiz+'CNAOUT',lil_d,lil_cond)
						#reset the index to the atomic IDs for the merge with the bispec data
						DOUT[(stru,tuple(i[dic[bound]]))].index=DOUT[(stru,tuple(i[dic[bound]]))]['PID']
							#need_proto=False
	print "\n\n\n			FINISHED SURFACE PROTOTYPES		\n\n\n"
	return DOUT						

def surfaces(se,alld,descriptors,dic,DOUT,lt):
	
	need_proto=False
	#DOUT={}

	for stru in se.structs:
			for i in se.indicies:
				for bound in se.bounds:
					for s in se.scales:
						box=get_boxes(s)
						for b in box:
							# we only need to check one [100] face!!
							if i[dic[bound]]==[0,1,0] or i[dic[bound]]==[0,0,1]:
								break		
							for t in range(se.thermal):

								lx=se.lattice*np.sqrt(i[0][0]**2+i[0][1]**2+i[0][2]**2)
								ly=se.lattice*np.sqrt(i[1][0]**2+i[1][1]**2+i[1][2]**2)
								lz=se.lattice*np.sqrt(i[2][0]**2+i[2][1]**2+i[2][2]**2)

								if random.random() < se.sampling_rate:
									fiz='example'+'_'+stru+str(random.randint(0,10000))
								else:
									fiz='temporaryfile'

								template=lt.adjust_temp(se.lattice,lx,ly,lz,i[0],i[1],i[2],bound,stru,fiz)
		
								with open('temp.in', 'w') as f:
									f.write(template)
									f.write('\nchange_box all '+b+' boundary '+bound+' remap units box\n')
									f.write('\nrun 0 post no\n')
				
								lmp=lammps()
								try:							
									lmp.file('temp.in')
									temp=nab_bispec_train(fiz)
									#alld=alld.append(temp)
								except:
									print "\nFAILURE TO RUN LAMMPS."																					
								lmp.close()
								# we need the number of atoms, the bounds (to tell which is periodic), the orientations, and the structure
								#descriptors.append([i,s,b,t,bound,len(temp)])
								if i == se.indicies[1]:
									strang='1 0 0'
								else:
									strang=str(i[dic[bound]])
								print '\n'+strang+' '+stru
								temp.index=temp['id']
								temp['desc']=DOUT[(stru,tuple(i[dic[bound]]))]['desc']

								alld=alld.append(temp)
	return alld


def make_vacancy_prototypes(se,alld,dic,lt):
	#dictionary of prototype features	
	DOUT={}
	
	dcheck={se.structs[0]:False,se.structs[1]:False}
	tdict={se.structs[0]:'fcc_vacancy',se.structs[1]:'bcc_vacancy'}
	counter=0
	i = se.indicies[1] #standard cell
	for stru in se.structs:

				lx=se.lattice*np.sqrt(i[0][0]**2+i[0][1]**2+i[0][2]**2)
				ly=se.lattice*np.sqrt(i[1][0]**2+i[1][1]**2+i[1][2]**2)
				lz=se.lattice*np.sqrt(i[2][0]**2+i[2][1]**2+i[2][2]**2)							
				
				fiz='tempstruct_'+str(counter)
				counter+=1
				template=lt.adjust_temp_athermal(se.lattice,lx,ly,lz,i[0],i[1],i[2],'p p p',stru,fiz)

				with open('temp.in', 'w') as f:
						f.write(template)
						f.write('\n group g1 id 1')
						f.write('\n delete_atoms group g1')
						f.write('\nrun 0 post no\n')
				lmp=lammps()
				try:							
					lmp.file('temp.in')
					temp=nab_bispec_train(fiz)
					#alld=alld.append(temp)
				except:
					print "\nFAILURE TO RUN LAMMPS."																					
				lmp.close()
				#make a prototype if needed
				#if need_proto==True:
				check=run_ovitos('CNA',fiz,fiz+'CNAOUT')
				lil_d={fiz+'CNAOUT':tdict[stru]}
				if stru=='fcc':
					lil_cond={fiz+'CNAOUT':['CNA',1]}
				elif stru=='bcc':
					lil_cond={fiz+'CNAOUT':['CNA',3]}
				DOUT[stru]=initialize_dislocation_descriptors(fiz+'CNAOUT',lil_d,lil_cond)
				#reset the index to the atomic IDs for the merge with the bispec data
				DOUT[stru].index=DOUT[stru]['PID']
					#need_proto=False
	print "\n\n\n			FINISHED VACANCY PROTOTYPES		\n\n\n"
	return DOUT	

def vacancies(se, alld, descriptors,DOUT,lt):

#loops to create all of the vacancies
	for stru in se.structs:
		for s in se.scales:
			box=get_boxes(s)
			for b in box:
				for t in range(se.thermal):

					lx=se.lattice
					ly=se.lattice
					lz=se.lattice

					if random.random() < se.sampling_rate:
						fiz='vacancy_'+stru+str(random.randint(0,10000))
					else:
						fiz='temporaryfile'

					template=lt.adjust_temp(se.lattice,lx,ly,lz,[1,0,0],[0,1,0],[0,0,1],'p p p',stru,fiz)

					with open('temp.in', 'w') as f:
						f.write(template)
						f.write('\n group g1 id 1')
						f.write('\n delete_atoms group g1')
						f.write('\nchange_box all '+b+' boundary '+'p p p'+' remap units box\n')
						f.write('\nrun 0 post no\n')
	
					lmp=lammps()
					try:					
						lmp.file('temp.in')
						temp=nab_bispec_train(fiz)
						#alld=alld.append(temp)
					except:
						print "\nFAILURE TO RUN LAMMPS."						
					lmp.close()
			
					# we need the number of atoms, the structure, flag that it's a vacancy
					print '\n'+'vacancy '+stru
					temp.index=temp['id']
					temp['desc']=DOUT[stru]['desc']

					alld=alld.append(temp)

	return alld


def make_interstitial_prototypes(se,alld,dic,lt):
	#dictionary of prototype features	
	DOUT={}
	ot={0:'tetrahedral',1:'octahedral'}
	tdict={(se.structs[0],ot[0]):'fcc_tet_interstitial',\
		   (se.structs[0],ot[1]):'fcc_oct_interstitial',\
		   (se.structs[1],ot[0]):'bcc_tet_interstitial',\
		   (se.structs[1],ot[1]):'bcc_oct_interstitial'}
	counter=0
	i = se.indicies[1] #standard cell


	for stru in se.structs:
		c=0
		for inter in se.interstitial[stru]:
			lx=se.lattice*np.sqrt(i[0][0]**2+i[0][1]**2+i[0][2]**2)
			ly=se.lattice*np.sqrt(i[1][0]**2+i[1][1]**2+i[1][2]**2)
			lz=se.lattice*np.sqrt(i[2][0]**2+i[2][1]**2+i[2][2]**2)							
			
			fiz='tempstruct_'+str(counter)
			counter+=1
			template=lt.adjust_temp_athermal(se.lattice,lx,ly,lz,i[0],i[1],i[2],'p p p',stru,fiz)

			with open('temp.in', 'w') as f:
				f.write(template)
				f.write('\ncreate_atoms 1 single '+str(inter[0])+' '+str(inter[1])+' '+str(inter[2]))
				f.write('\nrun 0 post no\n')
			lmp=lammps()
			try:							
				lmp.file('temp.in')
				temp=nab_bispec_train(fiz)
			except:
				print "\nFAILURE TO RUN LAMMPS."																					
			lmp.close()
			#make a prototype if needed
			check=run_ovitos('CNA',fiz,fiz+'CNAOUT')
			lil_d={fiz+'CNAOUT':tdict[(stru,ot[c])]}
			if stru=='fcc':
				lil_cond={fiz+'CNAOUT':['CNA',1]}
			elif stru=='bcc':
				lil_cond={fiz+'CNAOUT':['CNA',3]}
			DOUT[(stru,ot[c])]=initialize_dislocation_descriptors(fiz+'CNAOUT',lil_d,lil_cond)
			#reset the index to the atomic IDs for the merge with the bispec data
			DOUT[(stru,ot[c])].index=DOUT[(stru,ot[c])]['PID']
			c+=1
					#need_proto=False
	print "\n\n\n			FINISHED INTERSTITIAL PROTOTYPES		\n\n\n"
	return DOUT


def interstitials(se,alld,descriptors,DOUT,lt):

	ot={0:'tetrahedral',1:'octahedral'}
	for stru in se.structs:
		c=0
		for inter in se.interstitial[stru]:
			for s in se.scales:
				box=get_boxes(s)
				for b in box:
					for t in range(se.thermal):
						
						lx=se.lattice
						ly=se.lattice
						lz=se.lattice
						
						if random.random() < se.sampling_rate:
							fiz='keep_'+ot[c]+'interstitial_'+stru+str(random.randint(0,10000))
						else:
							fiz='temporaryfile'

						template=lt.adjust_temp(se.lattice,lx,ly,lz,[1,0,0],[0,1,0],[0,0,1],'p p p',stru,fiz)

						with open('temp.in', 'w') as f:
							f.write(template)
							f.write('\ncreate_atoms 1 single '+str(inter[0])+' '+str(inter[1])+' '+str(inter[2]))
							f.write('\nchange_box all '+b+' boundary '+'p p p'+' remap units box\n')
							f.write('\nrun 0 post no\n')

						lmp=lammps()
						try:						
							lmp.file('temp.in')
							temp=nab_bispec_train(fiz)
							#alld=alld.append(temp)
						except:
							print "\nFAILURE TO RUN LAMMPS."						
						lmp.close()
						# we need the number of atoms, the structure, flag for the type of intersitial
						print '\n'+ot[c]+'interstitial '+stru
						temp.index=temp['id']
						temp['desc']=DOUT[(stru,ot[c])]['desc']

						alld=alld.append(temp)
			c+=1

	return alld

def make_dislocations_prototypes(se):
	#OK what types of defects, FCC full, partial, BCC full... strain and thermalize
	DOUT={}
	tdict={se.defect_names[0]:'fcc_full_disloc',\
		   se.defect_names[1]:'fcc_partial_disloc',\
		   se.defect_names[2]:'bcc_full_disloc'}


	for d in se.defect_names:
		lil_d={se.flocs[d]:tdict[d]}
		if tdict[d].find('fcc')!=-1:
			if tdict[d].find('full')!=-1:
				lil_cond={se.flocs[d]:['CSP_surf',0.25]}
			else:
				lil_cond={se.flocs[d]:['CNA',1]}
		elif tdict[d].find('bcc')!=-1:
			lil_cond={se.flocs[d]:['CNA_surf',3]}
		
		DOUT[d]=initialize_dislocation_descriptors(se.flocs[d],lil_d,lil_cond)	
		DOUT[d].index=DOUT[d]['PID']
	print "\n\n\n			FINISHED DISLOCATION PROTOTYPES		\n\n\n"
	return DOUT



# make system calls to atomsk -- easy way to generate the inital dislocation stuctures, then strain and thermalize them such that they are still identified.
def dislocations(se,alld,descriptors,DOUT,lt):
	#OK what types of defects, FCC full, partial, BCC full... strain and thermalize
	for d in se.defect_names:
		for s in se.scales:
			box=get_boxes(s)
			for b in box:
				for t in range(se.thermal):
						
						lx=se.lattice
						ly=se.lattice
						lz=se.lattice
						
						if random.random() < se.sampling_rate:
							fiz='dislocation'+d+str(random.randint(0,10000))
						else:
							fiz='temporaryfile'

						template=lt.adjust_temp_read(se.lattice,fiz,d)

						with open('temp.in', 'w') as f:
							f.write(template)
							f.write('\nchange_box all '+b+' boundary '+'p p p'+' remap units box\n')
							f.write('\nrun 0 post no\n')

						lmp=lammps()
						try:						
							lmp.file('temp.in')
							temp=nab_bispec_train(fiz)
							#alld=alld.append(temp)
						except:
							print "\nFAILURE TO RUN LAMMPS."						
						lmp.close()
						# we need the number of atoms, the structure, flag for the type of intersitial
						print '\n'+d+'dislocation '
						temp.index=temp['id']
						temp['desc']=DOUT[d]['desc']
						alld=alld.append(temp)
	return alld


def tensile_prototype(se,lt):
	DOUT={}
	# we have 3 tensile tests right now
	for x in range(1,4):
		template=lt.adjust_temp_read(se.lattice,'tensile_'+str(x)+'.struct','tensile_'+str(x)+'.out',athermal=True)
		with open('temp.in', 'w') as f:
			f.write(template)
			f.write('\nrun 0 post no\n')
		lmp=lammps()					
		lmp.file('temp.in')
		lmp.close()
		check=run_ovitos('CNA','tensile_'+str(x)+'.struct','tensile_'+str(x)+'CNAOUT')
		lil_d={'tensile_'+str(x)+'CNAOUT':'fcc_partial_disloc'}
		lil_cond={'tensile_'+str(x)+'CNAOUT':['CNAonly',2]}

		DOUT[x]=initialize_dislocation_descriptors('tensile_'+str(x)+'CNAOUT',lil_d,lil_cond)	
		DOUT[x].index=DOUT[x]['PID']
	print "\n\n\n			FINISHED TENSILE PROTOTYPES		\n\n\n"

	return DOUT

def add_tensile_dislocations(se,alld,DOUT,lt):
	
	for x in range(1,4):
		#for t in range(se.thermal):
		template=lt.adjust_temp_read(se.lattice,'temporary.out','tensile_'+str(x)+'.out',athermal=True)
		with open('temp.in', 'w') as f:
			f.write(template)
			f.write('\nrun 0 post no\n')
		lmp=lammps()
		lmp.file('temp.in')
		temp=nab_bispec_train('temporary.out')					
		lmp.close()
		# we need the number of atoms, the structure, flag for the type of intersitial
		print '\n tensile dislocation run '+str(x)
		temp.index=temp['id']
		temp['desc']=DOUT[x]['desc']
		alld=alld.append(temp)

	return alld


def make_grain_prototypes(se,lt):
	#OK what types of defects, FCC full, partial, BCC full... strain and thermalize
	DOUT={}
	tdict={se.grain_names[0]:'fcc_GB',\
		   se.grain_names[1]:'bcc_GB'}
	
	for gn in se.grain_names:						
			
		template=lt.adjust_temp_read(se.lattice,gn+'.struct',gn,athermal=True)
		with open('temp.in', 'w') as f:
			f.write(template)
			f.write('\nrun 0 post no\n')
		lmp=lammps()					
		lmp.file('temp.in')
		lmp.close()
		check=run_ovitos('CNA',gn+'.struct',gn+'CNAOUT')
		
		lil_d={gn+'CNAOUT':tdict[gn]}
		if tdict[gn].find('fcc')!=-1:
			lil_cond={gn+'CNAOUT':['CNA',1]}
		elif tdict[gn].find('bcc')!=-1:
			lil_cond={gn+'CNAOUT':['CNA',3]}

		DOUT[gn]=initialize_dislocation_descriptors(gn+'CNAOUT',lil_d,lil_cond)	
		DOUT[gn].index=DOUT[gn]['PID']
	print "\n\n\n			FINISHED GRAIN BOUNDARY PROTOTYPES		\n\n\n"
	return DOUT



def grain_boundaries(se,alld,DOUT,lt):

	for gn in se.grain_names:
		template=lt.adjust_temp_read(se.lattice,'temporary.out',gn)
		with open('temp.in', 'w') as f:
			f.write(template)
			f.write('\nrun 0 post no\n')
		lmp=lammps()
		lmp.file('temp.in')
		temp=nab_bispec_train('temporary.out')					
		lmp.close()
		# we need the number of atoms, the structure, flag for the type of intersitial
		print '\n'+gn
		temp.index=temp['id']
		temp['desc']=DOUT[gn]['desc']
		alld=alld.append(temp)
	return alld

#assign the descriptors for each of the dislocation structures (no Kmeans necessary)
def initialize_dislocation_descriptors(f,tdict,condit):
	#f has the each of the files to consider, tdict relates the file name to the defect name
	#condit contains the conditions for each simulation, surf or no surf, CNA or CSP [0] spot
	# is the description and the [1] spot is the value needed
	tdf=pd.read_csv(f,delim_whitespace=True)
	tdf['desc']='Please Drop' #all of the spares go to please drop!
	if condit[f][0].find('surf')!=-1:#if it needs to trim the surfaces
		#filter out the surface atoms
		t1=tdf[tdf['X']<tdf['X'].max()-3.]
		t1=t1[t1['X']>t1['X'].min()+3.]
		t1=t1[t1['Y']<t1['Y'].max()-3.]
		t1=t1[t1['Y']>t1['Y'].min()+3.]
		if condit[f][0].find('CSP')!=-1:
			t1.ix[t1[t1['TYPE']>condit[f][1]].index,'desc']=tdict[f]
			if tdict[f].find('fcc')!=-1:
				t1.ix[t1[t1['TYPE']<=condit[f][1]].index,'desc']='bulk fcc'
			else:
				t1.ix[t1[t1['TYPE']<=condit[f][1]].index,'desc']='bulk bcc'
		elif condit[f][0].find('CNA')!=-1:
			t1.ix[t1[t1['TYPE']!=condit[f][1]].index,'desc']=tdict[f]
			if tdict[f].find('fcc')!=-1:
				t1.ix[t1[t1['TYPE']==condit[f][1]].index,'desc']='bulk fcc'
			else:
				t1.ix[t1[t1['TYPE']==condit[f][1]].index,'desc']='bulk bcc'
		
		tdf.ix[t1.index,'desc']=t1['desc']
	else:
		if condit[f][0].find('CNA')!=-1:
			if condit[f][0].find('only')!=-1: # say we only want the atoms with a particular CNA
				tdf.ix[tdf[tdf['TYPE']==condit[f][1]].index,'desc']=tdict[f]
			else: # takes both bulk and non bulkies here
				tdf.ix[tdf[tdf['TYPE']!=condit[f][1]].index,'desc']=tdict[f]
				if tdict[f].find('fcc')!=-1:
					tdf.ix[tdf[tdf['TYPE']==condit[f][1]].index,'desc']='bulk fcc'
				else:
					tdf.ix[tdf[tdf['TYPE']==condit[f][1]].index,'desc']='bulk bcc'
		elif condit[f][0].find('CSP')!=-1:
			tdf.ix[tdf[tdf['TYPE']>condit[f][1]].index,'desc']=tdict[f]
			if tdict[f].find('fcc')!=-1:
				tdf.ix[tdf[tdf['TYPE']<=condit[f][1]].index,'desc']='bulk fcc'
			else:
				tdf.ix[tdf[tdf['TYPE']<=condit[f][1]].index,'desc']='bulk bcc'
	
	return tdf


def make_all_structures(se,thermal_diff,num_bispec):
	
	lmp_templates=template_gen(thermal_diff,num_bispec)
	alld=pd.DataFrame()
	dic={se.bounds[0]:2,se.bounds[1]:1,se.bounds[2]:0} #convert the different boundaries to an index
	descriptors=[]
	#Bi spectrum gathering pipeline
	DOUT=make_surface_prototypes(se,alld,dic,lmp_templates)
	alld=surfaces(se,alld,descriptors,dic,DOUT,lmp_templates)
	DOUT=make_vacancy_prototypes(se,alld,dic,lmp_templates)
	alld=vacancies(se,alld,descriptors,DOUT,lmp_templates)
	DOUT=make_interstitial_prototypes(se,alld,dic,lmp_templates)
	alld=interstitials(se,alld,descriptors,DOUT,lmp_templates)
	DOUT=make_dislocations_prototypes(se)
	alld=dislocations(se,alld,descriptors,DOUT,lmp_templates)
	DOUT=make_grain_prototypes(se,lmp_templates)
	alld=grain_boundaries(se,alld,DOUT,lmp_templates)
	DOUT=tensile_prototype(se,lmp_templates)
	alld=add_tensile_dislocations(se,alld,DOUT,lmp_templates)

	alld.index = range(len(alld))	
	#alld=alld.dropna()
	alld=alld[alld['desc']!='Please Drop']
	
	get_timed_performance(se,thermal_diff,num_bispec)

	return alld,descriptors,DOUT


def get_timed_performance(se,thermal_diff,num_bispec):
	
	lmp_templates=template_gen(thermal_diff,num_bispec)
	with Timer('Grain Generations:'):
		DOUT=make_grain_prototypes(se,lmp_templates)
pass


if __name__=="__main__":

	class simulation_env:
	     def __init__(self):
		 self.sampling_rate = 0.000001#how often we select for keeps
		 self.num_comp = 5#number of PCA components that the bi components are reduced to
		 self.sample_size=100000#size of samples when machine learning
		 self.thermal=10 #number of thermal perturbations for each config.
		 self.lattice=3.597 #lattice size
		 self.indicies=[[[1,1,1],[-1,1,0],[-1,-1,2]], [[0,1,0],[0,0,1],[1,0,0]]]#crystal orientations
		 self.interstitial={'fcc':[[0.25,0.25,0.25],[0.5,0.0,0.0]],'bcc':[[0.5,0.25,0.0],[0.5,0.5,0.0]]}# interstitial sites for fcc, bcc (tet then oct)
		 self.structs=['fcc','bcc'] #structures considered
		 self.bounds=['p p s', 'p s p', 's p p'] #boundaries
		 self.scales=[1+x for x in np.arange(0.01,0.11,0.05)] #scales for straining the simulation cell
		 self.defect_names=['fccfull.lmp','fccpartial.lmp','bcc_disloc.lmp']
		 self.flocs={self.defect_names[0]:'fccfull_temp.structures',self.defect_names[2]:'BCC_temp.structures',self.defect_names[1]:'temp.structures'} #dislocation prototype file structural analysis files
		 self.grain_names=['fcc_polycrystal.lmp','bcc_polycrystal.lmp']
	
	#f has the each of the files to consider, tdict relates the file name to the defect name
		#condit contains the conditions for each simulation, surf or no surf, CNA or CSP [0] spot
		# is the description and the [1] spot is the value needed

	se=simulation_env()

	alld,descriptors,DOUT=make_all_structures(se,0.01,2)
	

