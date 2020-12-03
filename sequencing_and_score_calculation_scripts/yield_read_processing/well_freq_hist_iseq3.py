import multiprocessing
from Bio import SeqIO
import timeit
from functools import partial
from matplotlib import pyplot
import os.path
import numpy as np
pyplot.switch_backend('agg')
import math

def getcountinfo(cell,plate,iseq):
    let=['A','B','C','D','E','F','G','H']
    topcounts=[]
    toppercent=[]
    nextmax=[]
    idseq=0

    with open('./'+iseq+'_'+cell+'_seq.fasta','w') as outdna:
        for i in plate:
            for j in let:
                for k in range(12):
                    fname='./'+iseq+'/RPI'+i+'_'+j+str(k+1)+'_zotus_count.fasta'
                    if os.path.isfile(fname):
                        counts=[]
                        seqs=[]
                        with open(fname,'r') as indna:
                            parser= SeqIO.parse(indna,'fasta')
                            for m in parser:
                                seqs.append(str(m.seq).upper())
                                name=str(m.id).split(';')[1]
                                counts.append(int(name[5:]))
                            counts=np.array(counts)
                            topcounts.append(max(counts))
                            if max(counts)>=100:
                                tempcts=counts[counts!=max(counts)]
                                if tempcts.any():
                                    nextmax.append(max(tempcts))
                                    if max(tempcts)<=100:
                                        toppercent.append(max(counts)/sum(counts))
                                        if max(counts)/sum(counts)>0.4:
                                            outdna.write('>RPI'+i+'_'+j+str(k+1)+';'+'\n')
                                            outdna.write(seqs[int(np.where(counts==max(counts))[0])]+'\n')
                                            idseq=idseq+1
                                else:
                                    outdna.write('>RPI'+i+'_'+j+str(k+1)+';'+'\n')
                                    outdna.write(seqs[int(np.where(counts==max(counts))[0])]+'\n')
                                    idseq=idseq+1
    print(idseq)
    return (topcounts,nextmax,toppercent)

def main():

    print('read')
    topcounts=[]
    toppercent=[]
    nextmax=[]

    iseq='iseq3'
    for cell in ["IQ","SH"]:
        if cell=="SH":
            plate= ["34","35","36","37","38"]
        elif cell=="IQ":
            plate= ["24","25","26","27","28"]
        infoout=getcountinfo(cell,plate,iseq)
        topcounts.append(infoout[0])
        nextmax.append(infoout[1])
        toppercent.append(infoout[2])




    
if __name__ == '__main__':
    main()
        

