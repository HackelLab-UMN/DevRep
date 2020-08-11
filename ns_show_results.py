import matplotlib as mpl
from PIL import Image
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import ns_sampling_modules as sm
import ns_plot_modules as pm
import ns_data_modules as dm
from input_deck import inputs
import numpy as np
from input_deck import names,inputs
import glob
def violin_loop_plots(c,loops_2_show=None):
    if loops_2_show is None:
        loops_done=dm.read_pickle(c=c,file_description=names.loops_done_fn)
        #of the loops completed only show half of them.
        loops_2_show=loops_2_show[::2]

    pm.violin_saved_dataset(c=c,loops_2_show=loops_2_show)

def gif_make(c, file_prefix='heatmap_loop'):
    heat_maps=glob.glob(dm.make_file_name(c=c,file_description=file_prefix+'*',fileformat='png'))
    heat_maps=sorted(heat_maps,key=lambda x: int(x[len(dm.make_file_name(c=c,file_description=file_prefix,fileformat='')):-4]))
    im=[]
    for hm in heat_maps:
            im.append(Image.open(fp=hm))

    im[0].save(dm.make_file_name(c=c,file_description=file_prefix,fileformat='gif'),
               save_all=True,
               append_images=im[1:],
               optimize=False,
               duration=1000,
               loop=0)
    print('saving ... %s'%dm.make_file_name(c=c,file_description=file_prefix,fileformat='gif'))

def cys_stuff(c):
    seq_loop=os.listdir(path='./sampling_data/'+dm.make_directory(c=c))
    N=[]
    loops=[]
    for sl in seq_loop:
        if sl.startswith('sequences_loop') and sl.endswith('pkl'):
            df=dm.read_pickle(c=c,file_description=sl[0:-4])
            N.append(sm.convert2numpy(df=df,field='Ordinal'))
            loops.append(sl[15:-4])
    c_pos = 2
    c_max=-1
    for n in N:
        max=np.max(np.sum(n==c_pos,axis=1))
        if max > c_max:
            c_max=max

    for n,loop in zip(N,loops):
        # find the distribution for
        # make a new function for this
        b = n == c_pos
        c_nb=np.sum(b,axis=1)
        fig,ax=plt.subplots(1, 2, figsize=[10, 4], dpi=300)
        ax[0].hist(x=c_nb,bins=np.arange(0,c_max+1)-0.5,ec='k',rwidth=0.7,color='k')
        ax[0].set_xlabel('Number of Cys in Sequence')
        ax[0].set_ylabel('count')
        if c.mutation_type=='dynamic':
            title=c.mutation_type+' Loop %s'%loop
        else:
            title=c.mutation_type+':%i Loop %s'%(c.nb_mutations,loop)
        ax[0].set_title(title)



        TRU_LABELS=np.array(['7', '8', '9', '9b', '9c', '10', '11', '12', '34', '35', '36', '36b', '36c', '37', '38', '39'])
        COLORS=np.array(['w','r','hotpink','c','m','y','b','silver','lime','lightsalmon','pink','violet','aqua','peachpuff',
                'lightcyan','moccasin'])

        label = []
        percentage=[]
        nb_seq=[]
        p=[]
        # TODO: put average yield in here too
        for n_c in np.unique(b,axis=0):
            #calculate the percentage of each
            p.append(np.count_nonzero((b ==n_c).all(axis=1))/n.shape[0]*100)
            percentage.append('%0.2f'%p[-1])

            if n_c.any():
                s=','
                L=TRU_LABELS[n_c].tolist()
                label.append(s.join(L))
                nb_cys=np.count_nonzero(n_c)
                nb_seq.append(str(nb_cys))
            else:
                label.append('Non-Cys')
                nb_seq.append('0')



        # make the table
        table=[]
        for l,p,ns in zip(label,percentage,nb_seq):
            table.append([l,p,ns])


        table=sorted(table,key=lambda x: float(x[1]),reverse=True)
        tabel_row=['label', '% of total','# cys in sequence']

        colors=[]
        for t in table:
            ns=t[2]
            colors.append(['w','w',COLORS[int(ns)]])


        ax[1].set_title = 'LeaderBoard'
        ax[1].set_axis_off()


        table = ax[1].table(
            cellText=table,
            colLabels=tabel_row,
            cellColours=colors,
            colColours=["palegreen"] * 10,
            cellLoc='center',
            loc='upper left')

        fig.savefig(dm.make_file_name(c=c, file_description='cys_loop_%s' % loop, fileformat='png'))
        plt.close(fig)



C=[inputs(nb_loops=2,
         nb_steps=3,
         mutation_type='static',
         nb_mutations=1,
         nb_snapshots=2,
         Nb_sequences=1000)]
for c in C:
    violin_loop_plots(c=c,loops_2_show=np.array([1,2]))
    gif_make(c=c)
    dm.zip_data(c=c)
# for i in np.arange(1,3,1):
#     df=dm.read_pickle(c=c,file_description=dm.make_sequence_filename(loop_nb=i))
#     pm.make_heat_map(df=df,c=c,loop_nb=i-1)
# gif_heat_map(c=c)

