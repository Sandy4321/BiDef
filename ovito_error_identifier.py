import sys
from ovito.io import *
from ovito.modifiers import *
from ovito.anim import *
from ovito.data import CutoffNeighborFinder


node = None
cutoff = 6.0
stylename='Structure Type'
num_style=1
#stylename='vid'
#num_style=9


for file in sys.argv[1:]:

    if not node:
        # Import the first file using import_file().
        # This creates the ObjectNode and sets up the modification pipeline.
        node = import_file(file,multiple_frames=True)
        print(node.source.num_frames)
        # Insert a modifier into the pipeline.
        cna = CommonNeighborAnalysisModifier(adaptive_mode=True)
        node.modifiers.append(cna)
    else:
        continue	
        # To load subsequent files, call the load() function of the FileSource.
        #node.source.load(file)
	
# Evaluate pipeline and wait until the analysis results are available.
c=[]

fout=open(sys.argv[2:][0],'w')
fout.write('PID TYPE X Y Z E Frame\n')
for frame in range(0, ovito.dataset.anim.last_frame + 1):
	errors=[]
	ovito.dataset.anim.current_frame = frame    # Jump to the animation frame.
	o=node.compute()
	data = node.source
	num_particles = data.position.size
	finder = CutoffNeighborFinder(cutoff, data)
	for index in range(num_particles):
		errors.append(0)
		print("Neighbors of particle %i:" % index)
		# Iterate over the neighbors of the current particle:
		if o[stylename].array[index] != num_style:
			for neigh in finder.find(index):
				if o[stylename].array[neigh[0]] != num_style: #1 for fcc here
					errors[-1]+=1
		else:
			continue

	for iden,st,pos,ee in zip(o['Particle Identifier'].array,o['Centrosymmetry'].array,o['Position'].array,errors):
		fout.write(str(iden)+' '+str(st)+' '+str(pos[0])+' '+str(pos[1])+' '+str(pos[2])+' '+str(ee)+' '+str(frame)+'\n')

fout.close()
