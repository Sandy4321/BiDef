# get the num NN for each defective particle, printed with ML ID

import sys
from ovito.io import *
from ovito.modifiers import *
from ovito.anim import *
from ovito.data import CutoffNeighborFinder


node = None
cutoff = 3.0
#stylename='Structure Type'
#num_style=3
stylename='vid'
num_style=10


for file in sys.argv[1:]:

    if not node:
        # Import the first file using import_file().
        # This creates the ObjectNode and sets up the modification pipeline.
        node = import_file(file,multiple_frames=True)
        print(node.source.num_frames)
        cna = CommonNeighborAnalysisModifier(adaptive_mode=True)
        csp = CentroSymmetryModifier()
        node.modifiers.append(cna)
        node.modifiers.append(csp)
    else:
        continue	
        # To load subsequent files, call the load() function of the FileSource.
        #node.source.load(file)
	
# Evaluate pipeline and wait until the analysis results are available.
c=[]

fout=open(sys.argv[2:][0],'w')
fout.write('PID X Y Z VID E Frame VE CSP CSPVE CNA CNAVE\n')
for frame in range(0, ovito.dataset.anim.last_frame + 1):
	errors=[]
	NNe=[]
	NNcsp=[]
	NNcna=[]
	ovito.dataset.anim.current_frame = frame    # Jump to the animation frame.
	o=node.compute()
	data = node.source
	num_particles = data.position.size
	finder = CutoffNeighborFinder(cutoff, data)
	for index in range(num_particles):
		errors.append(0)
		NNe.append(0)
		NNcsp.append(0)
		NNcna.append(0)
		print("Neighbors of particle %i:" % index)
		# Iterate over the neighbors of the current particle:
		#if o[stylename].array[index] != num_style:
		for neigh in finder.find(index):
			errors[-1]+=1
			if o['vid'].array[neigh[0]]==15:
				NNe[-1]+=1
			if o['Centrosymmetry'].array[neigh[0]]>=5:
				NNcsp[-1]+=1
			if o['Structure Type'].array[neigh[0]]==0:
				NNcna[-1]+=1
		#else:
		#	continue

	for iden,pos,vid,ee,nn,cspa,nncsp,cnaa,nncna in zip(o['Particle Identifier'].array,o['Position'].array,o['vid'].array,errors,NNe,o['Centrosymmetry'].array,NNcsp,o['Structure Type'].array,NNcna):
		fout.write(str(iden)+' '+str(pos[0])+' '+str(pos[1])+' '+str(pos[2])+' '+str(vid)+' '+str(ee)+' '+str(frame)+' '+str(nn)+' '+str(cspa)+' '+str(nncsp)+' '+str(cnaa)+' '+str(nncna)+'\n')

fout.close()
