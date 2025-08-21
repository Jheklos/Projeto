#!/usr/bin/env python
# -*- utf-8 -*-
import sys
import os
from time import sleep

tip = []
tip.append('benign')
tip.append('malware')
antivirus = 'APT'

#direc = os.getcwd()
#os.chdir('../etapa_1_virustotal_api/' + antivirus +'/' + tipo + '/analysis')
#os.system('ls > '+ direc + '/nome.txt')
#os.chdir(direc)
#arqa = open('nome.txt','r')
###########################################
for tipo in tip:
	
	listDlls = open(tipo + 'listDLLs.txt','w')
	listDlls.close

	Dlls = []
	
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
	
		listDlls = open(tipo + 'listDLLs.txt','a')
		linhaDeTexto = arqResultPesc.readlines()

		for linha in linhaDeTexto:
			if '.dll' in linha:
				if ' ' in linha: 
					bag = linha.split()
				temp = bag[-1].split()	
				temp = ''.join(map(str, temp)) 
				temp = temp.lower()
				Dlls.append(temp + '\n')
			elif '.DLL' in linha:
				if ' ' in linha:
					bag = linha.split()
				temp = bag[-1].split()
				temp = ''.join(map(str, temp)) 
				temp = temp.lower()
				Dlls.append(temp + '\n')		
	
		print('As Dlls do {} foram capturadas!'.format(NomesArq[0]))
		arqResultPesc.close()
		listDlls.close

	Dlls = list(set(Dlls))
	print(Dlls)

	listDlls.writelines(Dlls)	

