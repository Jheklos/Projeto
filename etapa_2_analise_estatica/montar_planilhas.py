import time
import os
import os.path

tip = []
tip.append('benign')
tip.append('malware')
antivirus = 'APT'
###########################################
def search(txt,chave):
	estado = 0
	count = 0
	arq = open(txt, "r")
	t = arq.readlines()
	for line in t:			
		line = line.strip()
		if (chave.lower() in line.lower()) and (estado==0):
			estado = 1
		elif(estado==1): 
			if not line:
				count -= 1 				
				break
			else:
				count += 1

			
	arq.close()
	return count
###########################################
## Captura DLLs
os.system('python PegaResource.py')
###########################################
## Captura DLLs
os.system('python PegaDlls.py')
###########################################
## Captura APIs
os.system('python PegaIAT.py')
###########################################
features = []
features.append('file')
features.append('class (0:benign, 1:malware)')

arqa = open('set8086.txt', 'r')
ta = arqa.readlines()
set_count = 1
for linea in ta:
	linea = linea.strip()
	features.append(linea + ' ')
	set_count += 1
###########################################
lista = ['listResource.txt', 'listDLLs.txt', 'listIATs.txt']
for ii in lista:
	###################################
	temp = []
	for tipo in tip:
		arqc = open(tipo + ii , "r")
		tc = arqc.readlines()
		for linec in tc:
			linec = linec.strip()
			features.append(linec)
		arqc.close()
		temp = list(set(temp))
	features.extend(temp)	
	
features.append('[SUSPICIOUS]')
features.append('TLS_callbacks_Amount')
features.append('Exports_Amount')
###########################################
for tipo in tip:
	arqd = open(tipo + '.csv', "w")
	i=0
	while (i<(len(features))):	
		arqd.write(features[i] + ';')	
		i += 1	
	arqd.write("\n")
	arqd.close()	
	###########################################
	direc = os.getcwd()
	os.chdir('../etapa_1_virustotal_api/' + antivirus +'/' + tipo + '/analysis')
	os.system('ls > '+ direc + '/nome.txt')
	os.chdir(direc)

	arqa = open('nome.txt','r') 
	ta = arqa.readlines()


	arqd = open(tipo + '.csv', 'a')

	for linea in ta:
		linea = linea.strip()
		linecsv = linea.strip('.txt')
		strb = '../etapa_1_virustotal_api/' + antivirus +'/' + tipo + '/analysis/' + linea
		arqb = open(strb, 'r')

		print(strb)
		tb = arqb.readlines()	
		arqd.write(linecsv + ';')
		if tipo == 'benign':
		    arqd.write('0;')
		else:
		    arqd.write('1;')
		print(linea)	
		#-------------------------------------------	
		estadoOffset = 0
		i=1	
		while (i<(len(features)-2)):  		
			count = 0	
			for lineb in tb:	
				lineb = lineb.strip()
				#------Offset Instructions----------
				if 'Offset | Instructions' in lineb:
					estadoOffset = 1
				if len(lineb)<2 and (estadoOffset==1): #linha em branco
					estadoOffset = 0
				#------Offset Instructions----------
				if (i<set_count): 
					if (estadoOffset==1):
						if features[i].lower() in lineb.lower():
							count += 1		
				#---library, API, and SUSPICIOUS-----
				elif features[i].lower() in lineb.lower():
					count += 1
			#-----------------------------------	
			buff =  str(count) + ";"
			arqd.write(buff)
			i += 1	
		arqb.close()
		#-------------------------------------------	
		#	TLS	
		#-----------------------------------------	
		count = search(strb,"TLS callbacks")
		buff =  str(count) + ";"
		arqd.write(buff)
		#-----------------------------------------	
		#	Exports
		#-----------------------------------------	
		count = search(strb,"Exports")
		buff =  str(count) + ";"
		arqd.write(buff)
		arqd.write("\n")
	arqd.close()

os.remove('nome.txt')	

os.remove('benignlistDLLs.txt')	
os.remove('malwarelistDLLs.txt')

os.remove('benignlistResource.txt')
os.remove('malwarelistResource.txt')

os.remove('benignlistIATs.txt')
os.remove('malwarelistIATs.txt')



