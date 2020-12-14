import pandas as pd
import numpy as np 
from sklearn.metrics import pairwise_distances_chunked
import collections, functools, operator
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chisquare 


def reduce_func(D_chunk, start):
    print(start)
    dictionary=[]
    for row in D_chunk:
        distances=row[start+1:]*16 #find distances of untested pairs (upper right triangle)
        unique_elements, counts_elements = np.unique(distances, return_counts=True)
        dictionary.append(dict(zip(unique_elements,counts_elements)))
        start=start+1
    return dictionary

# def main():
    # df=pd.read_pickle('seq_to_assay_train_1,8,10_seq_and_assay_yield_forest_1_0.pkl')
    # df=pd.read_pickle('seq_and_assay_best_sequences.pkl')
    # print(len(df))
    # # df=df[0:1000]
    # x_a=df.loc[:,'Ordinal'].values.tolist()
    # for i in range(len(x_a)):
    #     x_a[i]=x_a[i].tolist()
    # x_a=np.array(x_a)
    # gen=pairwise_distances_chunked(x_a,metric='hamming',n_jobs=10,reduce_func=reduce_func,working_memory=4000)
    # result=[]
    # for part in gen:
    #     result.append(dict(functools.reduce(operator.add,map(collections.Counter,part))))
    # final_count=dict(functools.reduce(operator.add,map(collections.Counter,result)))

    # pickle.dump(final_count,open('./best_seq_and_assay_hamming_distances.pkl','wb'))

fig,ax = plt.subplots(1,1,figsize=[1.25,1.25],dpi=300)

best=pickle.load(open('./best_seq_and_assay_hamming_distances.pkl','rb'))
original=pickle.load(open('./unsorted_hamming_distances.pkl','rb'))


hamming_list=list(range(17))
best_hamming_freq=[]
for i in hamming_list:
    try:
        best_hamming_freq.append(best[i])
    except:
        best_hamming_freq.append(0)
best_hamming_freq=np.divide(best_hamming_freq,sum(best_hamming_freq))

best_average=0
for i in hamming_list:
    best_average=best_average+(i*best_hamming_freq[i])
print(best_average/np.sum(best_hamming_freq))

original_hamming_freq=[]
for i in hamming_list:
    try:
        original_hamming_freq.append(original[i])
    except:
        original_hamming_freq.append(0)
original_hamming_freq=np.divide(original_hamming_freq,sum(original_hamming_freq))
# original_hamming_freq=np.multiply(original_hamming_freq,sum(best_hamming_freq)) for chi square

original_average=0
for i in hamming_list:
    original_average=original_average+(i*original_hamming_freq[i])
print(original_average/np.sum(original_hamming_freq))

ax.plot(hamming_list,np.cumsum(original_hamming_freq),linewidth=0.5)
ax.plot(hamming_list,np.cumsum(best_hamming_freq),color='red',linewidth=0.5)
ax.set_ylabel('Cum. Sum\nof Seq. Pairs',fontsize=6)
# ax.set_ylim([np.float_power(10,-10),np.float_power(10,0)])
ax.tick_params(labelsize=6)
ax.set_xticks(hamming_list)
x_lbs=[]
for i in [0,2,4,6,8,10,12,14,16]:
    x_lbs.append(str(i))
    x_lbs.append('')
ax.set_xticklabels(x_lbs,rotation=90)
ax.set_xlabel('Hamming Distance',fontsize=6)
ax.set_yticks([0,0.5,1])
plt.tight_layout(pad=0.2)
fig.savefig('./best_seq_and_assay_hamming.png')

print(chisquare(best_hamming_freq[3:],original_hamming_freq[3:])) #compare only for d

