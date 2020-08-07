import pandas as pd 
from scipy.stats import ttest_ind_from_stats as ttest

def main():
    cv_scores=pd.read_csv('./aty_best_arch_cv.csv',index_col=0)
    cv_scores.columns=['model_name','cv_loss','cv_std']
    ## This access the aty_best_arch_cv file and labels the three columns 'model_name', 'cv_loss' and 'cv_std'
    ## Then assigns this to a variable cv_score
    best_assay=cv_scores[cv_scores['cv_loss']==cv_scores['cv_loss'].min()]
    ## best_assay is a data frame object which takes the model with the least cv_loss
    ## equal_combinations is an empty list variable.
    equal_combinations=[]
    for index,combin in cv_scores.iterrows():
        ## The above iterates through every row of the data frame and accesses each data row
        t,p=ttest(best_assay['cv_loss'],best_assay['cv_std'],10,combin['cv_loss'],combin['cv_std'],10)
        ## A t-test is performed against of each data row against the row with the smallest cv_loss value
        ## The t-test is performed assuming that both the data sets have equal variance i.e they passed the null hypothesis test (F-test)
        if p>=0.05:
            equal_combinations.append(combin)
        ## The data set is added to the empty list created if the data set is not significantly different from the best_assay as predicted by the ttest
    print(best_assay)
    print(pd.DataFrame(equal_combinations).sort_values(['cv_loss']))
    ## Finally the best_assay value is printed along with a sorted equal_combinations
		
	
if __name__=='__main__':
	main()