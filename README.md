# MPAS_O_Shallow_Water

The MPAS_O_Shallow_Water repository contains a bunch of Jupyter 
Notebooks and the corresponding Python scripts for solving the full non-linear 
barotropic shallow water equations on a MPAS-Ocean mesh, using the TRiSK scheme 
based on a mimetic finite volume spatial discretization, and a variety 
time-stepping algorithms. It also includes a suite of geophysically relevant 
verification exercises for testing spatial and temporal convergence. It can be 
used as a platform for developing and testing novel time-stepping algorithms to 
advance the barotropic equations of unstructured ocean models like MPAS-Ocean.

Please download the contents of the directories <br/>
(a) [Mesh+Initial_Condition+Registry_Files](https://www.dropbox.com/sh/txoitkq0mpk4wfx/AACBh6jrQx_1hRn-uWvwZXuja?dl=0) <br/>
(b) [MPAS_O_Shallow_Water_Mesh_Generation](https://www.dropbox.com/sh/gxo9jlvcce8ogdm/AAD2WdMfZ0wFPt-nc7KgXSfaa?dl=0) <br/>
and place these directories within your local MPAS_O_Shallow_Water repository.

If you would like to study the difference between any two commits on git or 
Github, please look at the Python scripts and not the Jupyter notebooks. As
explained [here](https://nextjournal.com/schmudde/how-to-version-control-jupyter),
Jupyter notebooks, containing metadata, source code, formatted text, and rich 
media, are considered to be poor candidates for conventional version control 
solutions, which works best with plain text.

You have the option to separately run either (a) the Jupyter Notebooks or 
(b) the Python scripts using the makefile which executes all the scripts in the 
proper sequence. Regarding the latter approach, typing make run over the 
terminal will do the trick and typing make clean will delete the generated 
\_\_pycache\_\_ directory. You can also run each individual Python script from 
the terminal by typing python <file_name.py>. But please remember that if you 
run any of the Jupyter Notebooks, it is going to modify the corresponding Python 
script by adding the line <br/> 
get_ipython().system('jupyter nbconvert --to script<file_name.ipynb>') <br/>
at the very end. This command basically converts the Jupyter Notebook into its 
corresponding Python script but the mere presence of this line will cause an 
error if you simply run the Python script (later on) by itself (in which case 
you need to delete the above-mentioned line).

If you specify the argument Show in a plotting routine as True, the Jupyter 
Notebook will display the figure right after execution of the cell containing 
the routine. I have, however, specified it to False at every instance. This is 
because if the user decides to run the Python script by itself, the display of 
the figure will pause the execution of the code, and the user has to close the 
figure window to resume it, which would be quite annoying. Besides, if the user 
decides to run just the Python script on a remote machine, specifying Show as 
True (i.e. attempting to display the figure during code execution) might result 
in an error and stop (not just pause) the execution of the code altogether.

Finally, I have tested every routine and verified the results. The option to 
test any routine is specified by the parameter do_<routine_name> (or 
test_<routine_name> or run_<routine_name> or something similar) following the 
routine. If it is turned on (i.e. specified as True), the routine will be 
executed. I have, however, turned off (i.e. specified to False) every such 
parameter in the code, after I am done with the testing and I am satisfied with 
the results. The reason is because if the testing of some routines is turned on 
and the script containing these routines is imported as a module by another 
routine, the execution of the second routine will be significantly slower than 
necessary. 

While you are checking out the Jupyter Notebooks after cloning the git 
repository onto your local computer, I would encourage you to revert the Show 
argument and the do_<routine_name> parameters to True everywhere and study the 
results. If you do so, you should be able to test the respective routines and 
generate all the figures (within a directory named MPAS_O_Shallow_Water_Output 
created within the local MPAS_O_Shallow_Water repository).
