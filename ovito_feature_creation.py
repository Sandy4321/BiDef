import sys
from ovito.io import *
from ovito.modifiers import *
from ovito.anim import *

node = None
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
        # To load subsequent files, call the load() function of the FileSource.
        node.source.load(file)

    # Evaluate pipeline and wait until the analysis results are available.
c=[]
fout=open('temp.structures','w')
fout.write('PID TYPE\n')
for frame in range(0, ovito.dataset.anim.last_frame + 1):
    ovito.dataset.anim.current_frame = frame    # Jump to the animation frame.
    o=node.compute()
    #print(node.compute())
    #print(o['Structure Type'].array)
    for iden,st in zip(o['Particle Identifier'].array,o['Structure Type'].array):
    	fout.write(str(iden)+' '+str(st)+'\n')		

fout.close()
