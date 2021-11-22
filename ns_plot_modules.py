import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np
import os
import ns_sampling_modules as sm
from ns_password import LOCAL_DIRECTORY
import ns_data_modules as dm
from   input_deck import names
fn=names()

def violin_saved_dataset(c,loops_2_show,y_lim=None):
    '''

    :param c: inputs() object
    :param loops_2_show: Nx1 ndarray array of loops to show in violin plot
    :param y_lim: 2x1 list . Max and min of y limits
    :return: makes violin plot of the loops specified in loops2show


    '''
    'nb strings is the number of strings in the violin plot'
    # first make the directory and get all the pkl files
    if y_lim is None:
        y_lim = [-1.5, 3]
    df_pp = pd.read_pickle(path=dm.make_file_name(c=c,file_description=fn.pp_fn))
    pp = sm.convert2numpy(df=df_pp, field=fn.pp_fn)
    df_min_yield =pd.read_pickle(path=dm.make_file_name(c=c,file_description=fn.min_yield_fn))
    min_yield=sm.convert2numpy(df=df_min_yield,field=fn.min_yield_fn)

    # make the figure
    fig,ax=plt.subplots(1, 1, figsize=[5, 3], dpi=300)

    labels=[]
    for k in np.arange(len(loops_2_show)):
        # read parralel file
        # s=strings[k]
        # n=numbers[k]

        n=loops_2_show[k]

        df=dm.read_pickle(c=c,file_description='sequences_loop_'+str(loops_2_show[k]))
        dev=sm.convert2numpy(df=df,field=c.yield2optimize)
        violin_parts =ax.violinplot([dev], positions=[k], showmedians=False,
                                                       showextrema=False, points=100,
                                                       widths=.9)
        # violin_parts['cmedians'].set_color('r')
        # violin_parts['cmedians']=min_yield[n-1]
        for pc in violin_parts['bodies']:
            pc.set_color('k')
        #TODO: figure out why min yield and pp are off by 2 idexes ,look to nested_sampling
        labels.append('Loop: %i'%n)

    nb_strings=loops_2_show.shape[0]
    ax.set_xticks(np.arange(nb_strings))
    ax.set_ylim(y_lim)
    ax.set_xticklabels(labels)
    # ax.set_xlabel('Loop',fontsize=6)
    ax.set_ylabel('Yield', fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_title('Nested Sampling Loops: %i, Random Walk Steps: %i ' % (np.max(loops_2_show),c.nb_steps))
    start=-0.5
    for k in loops_2_show:
        x=[start,start+1]
        y=[min_yield[k-1],min_yield[k-1]]
        ax.plot(x,y,'-r',linewidth=0.5,label='Threshold')
        ax.text(np.average(x),np.average(y)-0.02,'%0.2f'%min_yield[k-1],fontsize=6,color='grey',horizontalalignment='center',
                verticalalignment='top')
        start=start+1
    ax.legend(['Threshold'])
    fig.tight_layout()
    print('saving ' +dm.make_file_name(c=c,file_description='violin_plot_nb_strings_%i'%nb_strings,fileformat='png'))
    fig.savefig(dm.make_file_name(c=c,file_description='violin_plot_nb_strings_%i'%nb_strings,fileformat='png'))
    plt.close(fig)




def make_min_yield_plot(c, min_yield_lst):
    '''

    :param c: inputs() object
    :param min_yield_lst: min yield  list
    :return: makes a min yield plot saves to inputs() directory

    '''
    'this makes the min yield plot'
    plt.plot(np.arange(c.nb_loops + 1).tolist(), min_yield_lst)
    plt.title('min yield vs. nb of loops')
    plt.ylabel('min yield')
    plt.xlabel('nb of loops')
    plt.savefig(dm.make_file_name(c=c,
                                  file_description='min_yield',fileformat='png'))
    plt.close()

def make_percent_positive_plot(c,percent_pos):
    '''

    :param c: inputs() object
    :param percent_pos: percent positive list
    :return: makes a percent positive plot saves to inputs() directory

    '''
    'makes the percent positive plot '
    # pp = []
    # for i in np.arange(c.nb_loops):
    #     pp.append(sum(percent_pos[i])/ c.nb_steps) # Right now percent postive is just the average ...
    plt.plot(np.arange(c.nb_loops).tolist(), percent_pos)
    plt.title('percent accepted vs. for each loop')
    plt.ylabel('percent accepted')
    plt.xlabel('# of loops')
    plt.savefig(dm.make_file_name(c=c,file_description='percent_pos',fileformat='png'))
    plt.close()


def plot_hist(c, i, j, seq,bins=50):
    '''

    :param c: inputs() object
    :param i: step number
    :param j: loop number
    :param seq: dataframe containing developability column
    :param bins: # of bins to have in histogram
    :return: saves a histogram to folder specified by c
    '''
    # future version
    print('Plotting histogram Step:%i,Loop%i' % (i, j))
    dev=seq['Developability'].to_numpy()
    plt.hist(dev, bins=bins)
    plt.title('Step: %i Loop: %i threshold yield: %0.2f'
              % (i, j, np.min(dev)))
    plt.ylabel('frequency')
    plt.xlabel('yield')
    plt.savefig(dm.make_file_name(c=c,file_description='hist_loop_%i_step%i'%(j,i),fileformat='png'))
    plt.close()






def make_heat_map(df,c,loop_nb):
    '''

    :param df: dataframe with ordinal field
    :param c: inputs() object
    :param loop_nb: the loop number in the run
    :return: makes a heat map saves to inputs() directory

    '''
    ord=sm.convert2numpy(df=df,field='Ordinal')
    nb_AA=21
    nb_positions=16
    heat_map=np.zeros((nb_positions,nb_AA))
    for k in np.arange(nb_AA):
        # number of amino acids
        heat_map[:,k]=np.sum(ord==k,axis=0)

    frequency=(heat_map.T/np.sum(heat_map,axis=1)).T.copy()

    heat_map_plot(frequency=frequency,c=c,loop_nb=loop_nb)


def heat_map_plot(frequency,c,loop_nb):
    'function makes heat map : curtiousy of alex :)'
    frequency = pd.DataFrame(frequency)
    frequency.columns = list("ACDEFGHIKLMNPQRSTVWXY")
    frequency['Gap'] = frequency['X']
    frequency = frequency[
        ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'Gap']]
    frequency.index = ['7', '8', '9', '9b', '9c', '10', '11', '12', '34', '35', '36', '36b', '36c', '37', '38', '39']
    for pos in ['7', '8', '9', '10', '11', '12', '34', '35', '36', '37', '38', '39']:
        frequency['Gap'][pos] = np.nan
    frequency['Aromatic (F,W,Y)'] = frequency[['F', 'W', 'Y']].mean(axis=1)
    frequency['Small (A,C,G,S)'] = frequency[['A', 'G', 'C', 'S']].mean(axis=1)
    frequency['Non-Polar Aliphatic (A,G,I,L,M,P,V)'] = frequency[['P', 'M', 'I', 'L', 'V', 'A', 'G']].mean(axis=1)
    frequency['Polar Uncharged (C,N,Q,S,T)'] = frequency[['C', 'S', 'Q', 'S', 'T']].mean(axis=1)
    frequency['Negative Charged (D,E)'] = frequency[['D', 'E']].mean(axis=1)
    frequency['Positive Charged (H,K,R)'] = frequency[['H', 'K', 'R']].mean(axis=1)
    frequency['Hydrophobic (A,F,G,I,L,M,P,V,W,Y)'] = frequency[['A', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W', 'Y']].mean(
        axis=1)
    frequency['Hydrophilic (C,D,E,H,K,N,Q,R,S,T)'] = frequency[['C', 'D', 'E', 'H', 'K', 'N', 'Q', 'R', 'S', 'T']].mean(
        axis=1)
    frequency = frequency.transpose()
    frequency['Loop 1 (8-11)'] = frequency[['8', '9', '9b', '9c', '10', '11']].mean(axis=1)
    frequency['Loop 2 (34-39)'] = frequency[['34', '35', '36', '36b', '36c', '37', '38', '39']].mean(axis=1)
    frequency['Loop 1 & Loop 2'] = frequency[
        ['8', '9', '9b', '9c', '10', '11', '34', '35', '36', '36b', '36c', '37', '38', '39']].mean(axis=1)

    frequency = frequency[
        ['7', '8', '9', '9b', '9c', '10', '11', '12', '34', '35', '36', '36b', '36c', '37', '38', '39', 'Loop 1 (8-11)',
         'Loop 2 (34-39)', 'Loop 1 & Loop 2']]

    fig, ax = plt.subplots(1, 1, figsize=[5, 3], dpi=300)
    cmap = mpl.cm.Reds
    cmap.set_bad('black')

    heat_map = sns.heatmap(frequency.transpose(), square=True, vmin=0, vmax=1, cmap=cmap,
                           cbar_kws={"shrink": 0.6, "extend": 'min', "ticks": [0, 0.5, 1]})
    heat_map.figure.axes[-1].set_ylabel('Frequency', size=6)
    heat_map.figure.axes[-1].tick_params(labelsize=6)
    ax.set_yticks([x + 0.5 for x in list(range(19))])
    ax.set_yticklabels(
        ['7', '8', '9', '9b', '9c', '10', '11', '12', '34', '35', '36', '36b', '36c', '37', '38', '39', 'Loop 1 (8-11)',
         'Loop 2 (34-39)', 'Loop 1 & Loop 2'])
    ax.set_ylim([19.5, -0.5])

    ax.set_xticks([x + 0.5 for x in list(range(29))])
    pooled_AA = ['Aromatic', 'Small', 'Non-Polar Aliphatic', 'Polar Uncharged', 'Negative Charged', 'Positive Charged',
                 'Hydrophobic', 'Hydrophilic']
    ax.set_xticklabels(
        ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
         'Gap'] + pooled_AA)
    ax.set_xlim([-0.5, 29.5])
    ax.tick_params(labelsize=6)
    ax.set_title('Heat map , loop %i'%(loop_nb+1))

    plt.tight_layout()
    print('saving heatmap .. '+dm.make_file_name(c=c,file_description='heatmap_loop_%i'% loop_nb,fileformat='png'))
    fig.savefig(dm.make_file_name(c=c,file_description='heatmap_loop_%i'% (loop_nb+1),fileformat='png'))
    plt.close(fig)


def showFieldvsLoops(c, field2Show):
    '''

    :param c: inputs() object
    :param field2Show: what thing to show ?
    :return: shows a plot of specified field reading from parralel files in inputs() object

    '''

    df = pd.read_pickle(path=dm.make_file_name(c=c, file_description=field2Show))
    stat = sm.convert2numpy(df=df, field=field2Show)

    if c.mutation_type == 'dynamic':
        nm = ''
    else:
        nm = ' ,# mutations: %i' % c.nb_mutations
    plt.plot(np.arange(stat.shape[0]).tolist(), stat, label=c.mutation_type + ' ' + nm, )
    plt.title('%s vs nested sample loop' % field2Show)
    plt.ylabel(field2Show)
    plt.xlabel('loop number')
    plt.legend()
    plt.savefig(dm.make_file_name(c=c, file_description='%s_plot'%field2Show, fileformat='png'))
    plt.close()


def twinAxisvsLoops(c,fields2show):
    '''


    :param c: inputs() object
    :param fields2show: 2x1 List of two fields to show  [default: percent positive and nb mutations]
    :return: makes twin axis plot of the two fields that are specified

    '''

    df_pp = dm.read_pickle(c=c, file_description=fields2show[0])
    pp = sm.convert2numpy(df=df_pp, field=fields2show[0])

    df_nb = dm.read_pickle(c=c, file_description=fields2show[1])
    nb = sm.convert2numpy(df=df_nb, field=fields2show[1])



    # nb = nb[0:-2].copy()
    # if nb.shape[0] != pp.shape[0]:
    #     raise IndexError('number mutations and percent positive lengths are different?')

    loop_range = np.arange(pp.shape[0]).tolist()
    fig, ax1 = plt.subplots(1, 1, figsize=[5, 3], dpi=300)
    color = 'tab:red'
    ax1.set_xlabel('loop nb')
    ax1.set_ylabel(fields2show[1], color=color)
    ax1.plot(loop_range, nb[0:-1], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel(fields2show[0], color=color)  # we already handled the x-label with ax1
    ax2.plot(loop_range, pp, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(dm.make_file_name(c=c, file_description='%s_vs_%s_plot'%(fields2show[0],fields2show[1]), fileformat='png'))
    plt.close(fig)


def compare(C,field2Show):
    '''
    compare fields across runs
    make sure c.Nb_sequences and c.nb_loops is the same for all runs.
    :param C: list of runs
    :param field2Show: what thing to show ?
    :return: None
    '''


    for c in C:
        if C[0].Nb_sequences!=c.Nb_sequences or C[0].nb_loops !=c.nb_loops:
            raise SystemError('Your sequences and loops are not the same for comparing models.')
        df=pd.read_pickle(path=dm.make_file_name(c=c,file_description=field2Show))
        stat=sm.convert2numpy(df=df,field=field2Show)

        if c.mutation_type=='dynamic':
            nm=''
        else :
            nm=' ,# mutations: %i'%c.nb_mutations
        plt.plot(np.arange(stat.shape[0]).tolist(),stat,label=c.mutation_type+' '+nm,)
    plt.title('%s vs nested sample loop :%i'%(field2Show,C[0].Nb_sequences))
    plt.ylabel(field2Show)
    plt.xlabel('loop number')
    plt.legend()
    os.system('mkdir ./sampling_data/comparisons')
    print('saving ./sampling_data/comparisons/%s_nb_loops_%i_nb_sequences_%i'% (field2Show,C[0].nb_loops,C[0].Nb_sequences))
    plt.savefig('./sampling_data/comparisons/%s_nb_loops_%i_nb_sequences_%i'% (field2Show,C[0].nb_loops,C[0].Nb_sequences))
    plt.close()




# unit test for making the heat map... all seemed to go well.
# seed_parent = int.from_bytes(os.urandom(4), sys.byteorder)
# g_parent = tf.random.experimental.Generator.from_seed(seed_parent)
# original_seq=pd.DataFrame()
# original_seq['Ordinal']= sm.make_sampling_data(generator=g_parent,Nb_sequences=1000,Nb_positions=16)
# os.system('mkdir ./sampling_data/test_heat_map')
# make_heat_map(df=original_seq,dir_name='test_heat_map',loop_nb=0)


# def init_violin_plots(self,loops_2_show):
#     'initilize the violin plots'
#     nb_violinplots = len(loops_2_show)
#     for k in np.arange(nb_violinplots):
#         self.vp.append(plt.subplots(1, 1, figsize=[5, 3], dpi=300))
#     return self.vp
#
#
# def plot_violin(self, i, j, seq, steps_2_show, loops_2_show):
#     # i is the step number
#     # j is the Loop number
#     idx_loop = np.argmax(loops_2_show == j)
#     idx_step = np.argmax(steps_2_show == i)
#     dev = seq['Developability'].to_numpy()  # yield
#     violin_parts = self.vp[idx_loop][1].violinplot([dev], positions=[idx_step], showmedians=False,
#                                                    showextrema=False, points=100,
#                                                    widths=.9)
#     for pc in violin_parts['bodies']:
#         pc.set_color('k')
#
#
# def close_violin(self, j, steps_2_show, loops_2_show, Nb_steps, Nb_loops, y_lim=None):
#     'close a violin plot and save it as a png file'
#     if y_lim is None:
#         y_lim = [-1, 1.5]
#     v = self.vp[np.argmax(loops_2_show == j)]
#     str_steps_2_show = []
#     for i in steps_2_show:
#         if i == 0:
#             str_steps_2_show.append('Init')
#         else:
#             str_steps_2_show.append('step:%i,percent:%0.2f' % (i,sum(self.percent_pos[j])/Nb_steps))
#     fig = v[0]
#     ax = v[1]
#     ax.set_xticks(np.arange(len(steps_2_show)))
#     ax.set_ylim(y_lim)
#     ax.set_xticklabels(str_steps_2_show)
#     ax.set_ylabel('Yield', fontsize=6)
#     ax.tick_params(axis='both', which='major', labelsize=6)
#     ax.set_title('Loop %i of %i' % (j + 1, Nb_loops))
#     ax.axhline(self.min_yield[j]).set_color('r')
#     fig.tight_layout()
#     print('saving'+make_file_name(dir_name=self.dir_name,file_description='loop_%i'%(j+1),fileformat='png'))
#     fig.savefig(make_file_name(dir_name=self.dir_name,file_description='loop_%i'%(j+1),fileformat='png'))
#     plt.close(fig)

