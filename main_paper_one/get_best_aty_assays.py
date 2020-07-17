import pandas as pd 
from scipy.stats import ttest_ind_from_stats as ttest

def main():
	cv_scores=pd.read_csv('./aty_best_arch_cv.csv',index_col=0)
	cv_scores.columns=['model_name','cv_loss','cv_std']


	best_assay=cv_scores[cv_scores['cv_loss']==cv_scores['cv_loss'].min()]

	equal_combinations=[]
	for index,combin in cv_scores.iterrows():
		# print(combin)
		t,p=ttest(best_assay['cv_loss'],best_assay['cv_std'],10,combin['cv_loss'],combin['cv_std'],10)
		if p>=0.05:
			equal_combinations.append(combin)
	print(best_assay)
	print(pd.DataFrame(equal_combinations).sort_values(['cv_loss']))




if __name__=='__main__':
	main()