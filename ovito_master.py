import subprocess

def run_ovitos(btype,name,outputname):
	bashCommands={"CSP":"ovitos ovito_feature_creation_CSP.py","CNA_nosurf":"ovitos ovito_feature_creation_nosurf.py","CNA":"ovitos ovito_feature_creation.py"}
	BC=bashCommands[btype]+' '+name+' '+outputname
	process = subprocess.Popen(BC.split(), stdout=subprocess.PIPE)
	output = process.communicate()[0]
	return output
