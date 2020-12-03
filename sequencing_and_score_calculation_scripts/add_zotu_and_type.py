import multiprocessing
from Bio import SeqIO
import numpy as np
import timeit
from functools import partial
import pandas as pd

def translate(dna):
    transdic={"TTT":"F","TTC":"F","TTA":"L","TTG":"L",
              "CTT":"L","CTC":"L","CTA":"L","CTG":"L",
              "ATT":"I","ATC":"I","ATA":"I","ATG":"M",
              "GTT":"V","GTC":"V","GTA":"V","GTG":"V",
              "TCT":"S","TCC":"S","TCA":"S","TCG":"S",
              "CCT":"P","CCC":"P","CCA":"P","CCG":"P",
              "ACT":"T","ACC":"T","ACA":"T","ACG":"T",
              "GCT":"A","GCC":"A","GCA":"A","GCG":"A",
              "TAT":"Y","TAC":"Y","TAA":"Z","TAG":"Z",
              "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
              "AAT":"N","AAC":"N","AAA":"K","AAG":"K",
              "GAT":"D","GAC":"D","GAA":"E","GAG":"E",
              "TGT":"C","TGC":"C","TGA":"Z","TGG":"W",
              "CGT":"R","CGC":"R","CGA":"R","CGG":"R",
              "AGT":"S","AGC":"S","AGA":"R","AGG":"R",
              "GGT":"G","GGC":"G","GGA":"G","GGG":"G"}
    AAseq=[]
    if len(dna)%3!=0:
        return "FRAMESHIFT"
    for i in range(0,len(dna),3):
        AAseq.append(transdic[str(dna[i:i+3])])
    AAseq=''.join(AAseq)
    if "KFWATV"==AAseq[0:6] and "VTRVRP"==AAseq[-6:] and "FEVPVYAETLDEALQLAEWQY" in AAseq:
        AAseq=AAseq[6:-6]
        mid=AAseq.find("FEVPVYAETLDEALQLAEWQY")
        l1=AAseq[:mid]
        l2=AAseq[mid+len("FEVPVYAETLDEALQLAEWQY"):]
        if (6<=len(l1)<=8) and (6<=len(l2)<=8):
            if len(l1)==6:
                l1=l1[:3]+'XX'+l1[3:]
            elif len(l1)==7:
                l1=l1[:4]+'X'+l1[4:]
            if len(l2)==6:
                l2=l2[:3]+'XX'+l2[3:]
            elif len(l2)==7:
                l2=l2[:4]+'X'+l2[4:]
            AAseq=l1+l2
            return AAseq
        else:
            return "FRAMESHIFT"
    else:
        return "FRAMESHIFT"
    


def main():
#     num_cores = multiprocessing.cpu_count()
#     pool=multiprocessing.Pool(processes=num_cores) 
#     DNAseq,name=[],[]
#     with open('./zotus_match.fasta') as indna:
#         parser= SeqIO.parse(indna,'fasta')
#         for i in parser:
#             DNAseq.append(i.seq)
#             name.append(i.id)
#     (AAseq)=pool.map(translate,DNAseq)
#     stop,CC,match=[],[],[]
#     for A in AAseq:
#         if "Z" in A:
#             stop.append(True)
#         else:
#             stop.append(False)
#         if A[0]==A[7] and A[0]=='C':
#             CC.append(True)
#         else:
#             CC.append(False)
#         if A=='FRAMESHIFT':
#             match.append(False)
#         else:
#             match.append(True)
#     name_type=pd.DataFrame([name,DNAseq,AAseq,stop,CC,match])
#     name_type=name_type.transpose()
#     name_type.columns=['Zotu_name','DNA','Paratope','Stop','CC','Match']
#     name_type=name_type[name_type['Match']==True]
#     name_type.to_pickle('./name_type.pkl')



# ### merge name and scores
    name_type=pd.read_pickle('./name_type.pkl')
    otu_table=pd.read_pickle('./match_scores.pkl')
    merged_otu=pd.merge(name_type,otu_table,how='left',left_on='Zotu_name',right_on='#OTU ID')
    merged_otu.to_pickle('./name_scores.pkl')

### clean up datatable
    final_table=pd.read_pickle('./name_scores.pkl')
    cols_test=[]
    for i in range(1,11):
        cols_test.append('Sort'+str(i)+'_mean_count')
        cols_test.append('Sort'+str(i)+'_mean_score')
        cols_test.append('Sort'+str(i)+'_std_score')

    cols_keep=["#OTU ID","DNA","Paratope","Stop","CC"]
    for i in range(1,11):
        for j in range(1,4):
            # cols_keep.append('Sort'+str(i)+'_'+str(j)+'_count')
            cols_keep.append('Sort'+str(i)+'_'+str(j)+'_score')
    for i in cols_test:
        cols_keep.append(i)

    clean=final_table[cols_keep]
    clean.to_pickle('./name_scores_cleaned.pkl')


# ### merge name+scores and dots
    otu_table=pd.read_pickle('./name_scores_cleaned.pkl')
    IQ_table=pd.read_csv('./IQ_final.csv',header=0)
    SH_table=pd.read_csv('./SH_final.csv',header=0)
    yields_table=pd.merge(IQ_table,SH_table,how='outer',on=['DNA','CC','Stop','AA'])
    yields_table['Paratope']=yields_table['AA']
    yields_table=yields_table[["DNA","Position_x","IQ_Average",'IQ_Trial1_Adj','IQ_Trial2_Adj','IQ_Trial3_Adj',"Position_y","SH_Average",'SH_Trial1_Adj','SH_Trial2_Adj','SH_Trial3_Adj',"CC",'Stop','Paratope']]

    final_table=pd.merge(otu_table,yields_table,how='outer',on=['DNA','CC','Stop','Paratope'])
    final_table.to_pickle('./name_scores_yield.pkl')




# ###save a subset to visualize
    final_table=pd.read_pickle('./name_scores_yield.pkl')
    final_table=final_table.iloc[-1000:]
    final_table.to_csv('./name_scores_yield_sample.csv')





if __name__ == '__main__':
    main()