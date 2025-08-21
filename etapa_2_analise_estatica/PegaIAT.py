#!/usr/bin/env python
# -*- utf-8 -*-
import sys
import os
from time import sleep

tip = []
tip.append('benign')
tip.append('malware')
antivirus = 'APT'
###########################################
for tipo in tip:
	
	listIATs = open(tipo + 'listIATs.txt','w')
	listIATs.close

	IATs = []
	
	direc = os.getcwd()
	os.chdir('../etapa_1_virustotal_api/' + antivirus +'/' + tipo + '/analysis')
	os.system('ls > '+ direc + '/nome.txt')
	os.chdir(direc)
	arqNomes = open('nome.txt','r')

	nomesArquivos = arqNomes.readlines()

	for linha in nomesArquivos:
		NomesArq = linha.split('\n')
		print(NomesArq)
		arqResultPesc = open('../etapa_1_virustotal_api/' + antivirus +'/' + tipo + '/analysis/' + NomesArq[0],'r')
	
		listIATs = open(tipo + 'listIATs.txt','a')
		linhaDeTexto = arqResultPesc.readlines()

		estado = 0
		for linha in linhaDeTexto:
		
			if ('Suspicious IAT alerts' in linha) and (estado==0):
				estado = 1
			elif (estado==1):
				estado = 2
			elif (estado==2):
				if len(linha)>1:
					bag = linha.split()
					IATs.append(bag[-1]+'\n')
				else:
					break
			
		print('As IATs do {} foram capturadas!'.format(NomesArq[0]))
		arqResultPesc.close()
		listIATs.close

	print(IATs)
	IATs = list(set(IATs))


	listIATs.writelines(IATs)	

	
