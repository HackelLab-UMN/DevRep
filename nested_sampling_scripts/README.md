#**Nested sampling**
This is the set up for the nested sampling modules for DevRep. The goal of this nested sampling 
routine is to find the optimal sequences with the highest yield. As reference all files that lead with a `ns_*.py` 
pertain to nested sampling files. 

As of right now only doing single mutations. Also no parrelizations. 

#**Set up**

1. if not done already, make local conda environment as specified by `/DevRep/conda_package_list.txt`, 
instructions to create the environment are in the file. 
3. run script `main_DevRep_example.py`, this will save learned embedding pickle files to `/datasets/predicted`.
4. if not already present make a directory `/DevRep/sampling_data`. This is where results of a `ns_main_sampling.py` will 
be stored. 
3. go into `ns_main_sampling.py` to specify the Number of loops (`N_loops`) to use, and number of 
steps to use(`N_steps`), number of sequences (`nb_sequences`), and number of snapshots to 
 take (`nb_snapshots`)for a given run. Can also toggle output. 
4. To see results not in the form of pkl files, run `ns_show_results.py`
4. To change model parameters, go into `ns_nested_sampling.py` and change the model parameters 
`e2y` and `s2a` in initilization of `nested_sampling` class. 
5. If running on MSI, functions are provided to easily transfer small files 
and data between msi account and local drive in `ns_msi.py`. Just specify the full path to 
of DevRep in `ns_passwords.py` 

#**Python File Descriptions** 

- `ns_main_sampling.py` : go here to specify input parameters for a run 
- `ns_nested_sampling.py` : main script that contains classes to run nested sampling
if you would like to make your own monte carlo walk, it must inherit the `nested_sampling`
class and must have a single method named walk, specifying how to do the random walk. 
- `ns_show_results.py` : see results of `ns_main_sampling` with figures. 
- `ns_plot_modules.py` : contains helper functions to plot figures 
- `ns_sampling_modules.py`: helper functions to sample ordinals , change blanks
- `ns_msi.py` : some helper functions to transfer data to and from msi

#**The Algorithm** 



#Version Stats
Author: Bryce Johnson joh14192@umn.edu , Alex Golinski, Prof. Stefano Martiniani

University of Minnesota -Twin Cities - CEML - Martiniani Lab  

Version 0.3 - **not stable** - 7/27/2020 


