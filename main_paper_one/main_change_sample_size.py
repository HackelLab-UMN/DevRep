import sys
import submodels_module as modelbank
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import load_format_data
from sklearn.linear_model import LinearRegression 



def main():
    '''
    compare test performances when reducing training sample size. This version is for first paper, predicting yield from assays and one-hot encoded sequence. 
    '''


    a=int(sys.argv[1]) #for training via PBS array ID
    if a<4:
        b=0
    elif a<8:
        a=a-4
        b=1
    elif a<12:
        a=a-8
        b=2
    elif a==12:
        b=3
        a=a-12
    else:
        print('incorrect toggle number')



    arch_list=['ridge','svm','forest','fnn']

    size_list=[0.055,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    for size in size_list:
        if b==0:
            mdl=modelbank.seqandassay_to_yield_model([1,8,10],arch_list[a],size)
        elif b==1: #1,5,9,12
            mdl=modelbank.assay_to_yield_model([1,8,10],arch_list[a],size)
        elif b==2: 
            mdl=modelbank.seq_to_yield_model(arch_list[a],size)
        elif b==3:
            mdl=modelbank.control_to_yield_model(arch_list[a],size)

        for seed in range(9): #no seed is seed=42
            mdl.change_sample_seed(seed)
            mdl.cross_validate_model()
            mdl.limit_test_set([1,8,10])
            mdl.test_model()



# if __name__ == '__main__': #for training 
#     main()

#for evaluation
arch_list=['ridge','svm','forest']
best_arch_list=[]
size_list=[0.055,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
loss_per_model,std_per_model=[],[]
cv_loss_per_model,cv_std_per_model=[],[]
for b in range(4):
    loss_per_size,std_per_size=[],[]
    cv_loss_per_size,cv_std_per_size=[],[]
    for size in size_list:
        ####
        best_cv_per_seed_list,best_test_per_seed_list=[],[]
        best_cv_per_seed,best_test_per_seed=np.inf,np.inf
        for arch in arch_list:
            if b==0:
                mdl=modelbank.seqandassay_to_yield_model([1,8,10],arch,size)
            elif b==1: #1,5,9,12
                mdl=modelbank.assay_to_yield_model([1,8,10],arch,size)
            elif b==2: 
                mdl=modelbank.seq_to_yield_model(arch,size)
            elif b==3:
                mdl=modelbank.control_to_yield_model('ridge',size)

            if mdl.model_stats['cv_avg_loss']<best_cv_per_seed:
                best_cv_per_seed=mdl.model_stats['cv_avg_loss']
                mdl.limit_test_set([1,8,10])
                # mdl.test_model()
                best_test_per_seed=mdl.model_stats['test_avg_loss']

        best_cv_per_seed_list.append(best_cv_per_seed)
        best_test_per_seed_list.append(best_test_per_seed)

        for seed in range(9):
            best_cv_per_seed,best_test_per_seed=np.inf,np.inf
            for arch in arch_list:
                if b==0:
                    mdl=modelbank.seqandassay_to_yield_model([1,8,10],arch,size)
                elif b==1: #1,5,9,12
                    mdl=modelbank.assay_to_yield_model([1,8,10],arch,size)
                elif b==2: 
                    mdl=modelbank.seq_to_yield_model(arch,size)
                elif b==3:
                    mdl=modelbank.control_to_yield_model('ridge',size)

                for j in range(seed+1):
                    mdl.change_sample_seed(j)
                if mdl.model_stats['cv_avg_loss']<best_cv_per_seed:
                    best_cv_per_seed=mdl.model_stats['cv_avg_loss']
                    mdl.limit_test_set([1,8,10])
                    # mdl.test_model()
                    best_test_per_seed=mdl.model_stats['test_avg_loss']

            best_cv_per_seed_list.append(best_cv_per_seed)
            best_test_per_seed_list.append(best_test_per_seed)

        loss_per_size.append(best_test_per_seed_list)
        cv_loss_per_size.append(best_cv_per_seed_list)

    loss_per_model.append(loss_per_size)
    cv_loss_per_model.append(cv_loss_per_size)

size_list=np.multiply(size_list,len(mdl.training_df))
loss_per_model=np.array(loss_per_model)
cv_loss_per_model=np.array(cv_loss_per_model)

control_model=modelbank.control_to_yield_model('ridge',1)
control_model.limit_test_set([1,8,10])
exploded_df,_,_=load_format_data.explode_yield(control_model.testing_df)
exp_var=np.average(np.square(np.array(exploded_df['y_std'])))

exploded_df,_,_=load_format_data.explode_yield(control_model.training_df)
cv_exp_var=np.average(np.square(np.array(exploded_df['y_std'])))


import statsmodels.api as sm
from scipy.stats.stats import pearsonr
from uncertainties import ufloat

#fit prediction weighting by confidence in y (1/variance)
def poly_fit(x, y, yerr=None, order=1):
    w = 1. / np.array(yerr) if yerr is not None else np.ones(len(x))
    x = np.asarray(x)
    X = np.column_stack(tuple([x ** i for i in range(1, order + 1)]))
    X = sm.add_constant(X)
    wls_model = sm.WLS(y, X, weights=w)
    results = wls_model.fit()
    fit_params = results.params[::-1]
    fit_err = results.bse[::-1]
    # print results.summary()
    fit_fn = np.poly1d(fit_params)
    # rho = pearsonr(x, y)[0]
    return fit_fn, fit_params, fit_err

#plot fit and save predicted intercept with exp_var
fig,ax=plt.subplots(1,3,figsize=[6,2],dpi=300,sharey=True,sharex=True)
prd=[]
for i in range(3):
    x=np.log10(size_list[:]).reshape(-1,1)
    y=np.average(np.log10(loss_per_model[i,:,:]),axis=1)
    yerr=np.square(np.std(np.log10(loss_per_model[i,:,:]),axis=1))
    f,param,param_err=poly_fit(x,y,yerr=yerr)
    y_goal=np.log10(exp_var)
    m=ufloat(param[0],param_err[0])
    b=ufloat(param[1],param_err[1])
    print(10**((y_goal-b)/m))
    prd.append(10**((y_goal-b)/m))
    ax[i].plot(x,y,color='red',marker='o',markersize=4,linewidth=0)
    x_pred=np.array([1,2,3,4,5])
    ax[i].plot(x_pred,(x_pred*param[0])+param[1],color='black',linewidth=1,linestyle='--')
    ax[i].tick_params(axis='both', which='both', labelsize=6)
    ax[i].axhline(y_goal,0,5,color='purple',linestyle='--')
    ax[i].set_xlim([1,5])

#0 is seq and assay
#1 is assay
#2 is seq
print((prd[2]-prd[0])/prd[2])
print((prd[2]-prd[1])/prd[2])

#plot Fit
ax[0].set_title(r"$Seq.& P_{PK37},G_{SH},\beta_{SH}$",fontsize=6)
ax[0].set_ylabel('$Log_{10}$ Test Loss',fontsize=6)
ax[1].set_title(r"$P_{PK37},G_{SH},\beta_{SH}$",fontsize=6)
ax[1].set_xlabel('$Log_{10}$ Number of Training Sequences',fontsize=6)
ax[2].set_title('OH Sequence',fontsize=6)
fig.tight_layout()
fig.savefig('./changing_sample_size_fit.png')
plt.close()

#plot Fig 6
fig,ax=plt.subplots(1,2,figsize=[4,2],dpi=300,sharey=True)
ax[0].errorbar(np.divide(size_list,1),np.average(cv_loss_per_model[3,:,:],axis=1),yerr=np.std(cv_loss_per_model[3,:,:],axis=1),label='Strain Only',marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='red')
ax[0].errorbar(np.divide(size_list,1),np.average(cv_loss_per_model[2,:,:],axis=1),yerr=np.std(cv_loss_per_model[2,:,:],axis=1),label='OH Sequence',marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='blue')
ax[0].errorbar(np.divide(size_list,1),np.average(cv_loss_per_model[1,:,:],axis=1),yerr=np.std(cv_loss_per_model[1,:,:],axis=1),label=r"$P_{PK37},G_{SH},\beta_{SH}$",marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='black')
ax[0].errorbar(np.divide(size_list,1),np.average(cv_loss_per_model[0,:,:],axis=1),yerr=np.std(cv_loss_per_model[0,:,:],axis=1),label=r"$Seq.&\ P_{PK37},G_{SH},\beta_{SH}$",marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='orange')
ax[0].axhline(cv_exp_var,0,198,color='purple',linestyle='--',label='Experimental Variance')
# ax[0].legend(fontsize=6,framealpha=1)
ax[0].set_ylabel('CV Loss',fontsize=6)
ax[0].set_xlabel('Number of Training Sequences',fontsize=6)
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].tick_params(axis='both', which='major', labelsize=6)
ax[0].tick_params(axis='both', which='minor', labelsize=6)

# ax[0].set_ylim([0.3,1])
# ax[0].axis('scaled')

ax[1].errorbar(np.divide(size_list,1),np.average(loss_per_model[3,:,:],axis=1),yerr=np.std(loss_per_model[3,:,:],axis=1),label='Strain Only',marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='red')
ax[1].errorbar(np.divide(size_list,1),np.average(loss_per_model[2,:,:],axis=1),yerr=np.std(loss_per_model[2,:,:],axis=1),label='OH Sequence',marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='blue')
ax[1].errorbar(np.divide(size_list,1),np.average(loss_per_model[1,:,:],axis=1),yerr=np.std(loss_per_model[1,:,:],axis=1),label=r"$P_{PK37},G_{SH},\beta_{SH}$",marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='black')
ax[1].errorbar(np.divide(size_list,1),np.average(loss_per_model[0,:,:],axis=1),yerr=np.std(loss_per_model[0,:,:],axis=1),label=r"$Seq.&\ P_{PK37},G_{SH},\beta_{SH}$",marker='o',linestyle='--',fillstyle='none',markersize=3,linewidth=1,color='orange')
ax[1].axhline(exp_var,0,198,color='purple',linestyle='--',label='Experimental Variance')
ax[1].set_ylabel('Test Loss',fontsize=6)
ax[1].set_xlabel('Number of Training Sequences',fontsize=6)
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].tick_params(axis='both', which='major', labelsize=6)
ax[1].tick_params(axis='both', which='minor', labelsize=6)

ax[1].set_ylim([0.3,1])
ax[1].axis('scaled')

fig.tight_layout()
fig.savefig('./changing_sample_size.png')
plt.close()

for i in range(3):
    seq_needed_per_sample=[]
    for j in range(10):
        mdl=LinearRegression().fit(np.log10(size_list[:]).reshape(-1,1),np.log10(loss_per_model[i,:,j]))
        seq_needed=(np.log10(exp_var)-mdl.intercept_)/mdl.coef_
        seq_needed_per_sample.append(seq_needed)
    print(seq_needed_per_sample)
    print(np.average(seq_needed_per_sample))
    print(np.std(seq_needed_per_sample))