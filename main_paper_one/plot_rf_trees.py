import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import submodels_module as mb
from sklearn import tree
import numpy as np
import pandas as pd
import seaborn as sns

#dive into the random forest to see if anything is interesting. 

def get_node_depths(tree):
    """
    Get the node depths of the decision tree

    >>> d = DecisionTreeClassifier()
    >>> d.fit([[1,2,3],[4,5,6],[7,8,9]], [1,2,3])
    >>> get_node_depths(d.tree_)
    array([0, 1, 1, 2, 2])
    """
    def get_node_depths_(current_node, current_depth, l, r, depths):
        depths += [current_depth]
        if l[current_node] != -1 and r[current_node] != -1:
            get_node_depths_(l[current_node], current_depth + 1, l, r, depths)
            get_node_depths_(r[current_node], current_depth + 1, l, r, depths)

    depths = []
    get_node_depths_(0, 0, tree.children_left, tree.children_right, depths) 
    return np.array(depths)





a=mb.seqandassay_to_yield_model([1,8,10],'forest',1)
a.load_model(0)

#sort1 = input 0
#sort8 = input 1
#sort10 = input 2
fig,ax=plt.subplots(1,3,figsize=[6,2],dpi=300,sharey=True)
sorts=["Prot K 37","GFP SHuffle",r'$\beta$'+"-lactamase SHuffle"]
for sort_no in [0,1,2]:

    blac_nodes_info=[]
    for j in range(a._model.model.n_estimators):
        a_tree=a._model.model.estimators_[j].tree_
        node_depth=get_node_depths(a_tree)

        for i in range(a_tree.node_count):
            if a_tree.feature[i]==sort_no:

                #tree left is always x<=treshold. and should be "lower" yield
                left_idx=a_tree.children_left[i]
                right_idx=a_tree.children_right[i]
                # if left value is lower, sign = True
                if a_tree.value[left_idx][0][0]<a_tree.value[right_idx][0][0]:
                    sign=1
                else:
                    sign=0

                if node_depth[i]<5:
                    blac_nodes_info.append([0,a_tree.threshold[i],sign])
                elif node_depth[i]<10:
                    blac_nodes_info.append([1,a_tree.threshold[i],sign])
                else:
                    blac_nodes_info.append([2,a_tree.threshold[i],sign])
                # blac_nodes_info.append([0,a_tree.threshold[i],sign])


    blac_nodes_info=np.array(blac_nodes_info)
    blac_nodes_info=pd.DataFrame(blac_nodes_info)
    blac_nodes_info.columns=['Node Depth','Threshold','Correlation']

    g=sns.violinplot(data=blac_nodes_info,x='Node Depth',y='Threshold',split=True,hue='Correlation',ax=ax[sort_no],inner=None,palette=['r','k'],saturation=1,scale='count')
    g.set_ylim([0,1])
    g.tick_params(axis='both', which='major', labelsize=6)
    g.xaxis.label.set_size(6)
    g.yaxis.label.set_size(6)
    g.set_title(sorts[sort_no],fontsize=6)
    shallow_count=sum(blac_nodes_info['Node Depth']==0)
    medium_count=sum(blac_nodes_info['Node Depth']==1)
    deep_count=sum(blac_nodes_info['Node Depth']==2)

    g.xaxis.set_ticklabels(['Shallow\n'+str(shallow_count),'Medium\n'+str(medium_count),'Deep\n'+str(deep_count)])
    if sort_no in [0,2]:
        g.get_legend().remove()
        g.xaxis.label.set_visible(False)
    else:
        g.legend(['Negative','Positive'],loc='center',bbox_to_anchor=[0.5,0.85],fontsize=6,title='Correlation',title_fontsize=6,ncol=2)
        g.set_ylabel("Node Depth\n# Nodes")
    if sort_no>0:
        g.yaxis.label.set_visible(False)
    # g.legend(['Negative','Postive'])




fig.tight_layout()
fig.savefig('./sort_forest_cutoffs.png')
plt.close()
