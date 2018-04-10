# LANGUAGE UNDERSTANDING SYSTEMS

![COVER](./resources/landscape_cover.png)

### Introduction

This repository contains the code for the Spoken Language Understanding Module required as mid-term project
within the Language Understanding Systems course, held by Professor Giuseppe Riccardi at the University of
Trento in the academic year 2017/2018.

### Requirements


The code was developed using Jupyter as development environment. 

&#128279; link to Jupyter: http://jupyter.org/ \
&#128279; link to Plotly: https://plot.ly/

A <tt>.py</tt> format of each script is also provided, which can be executed via command line as follows

```
python3 script_name.py
```

&#9432; The only exception is the data analysis phase, which is dependent on Plotly and Jupyter for the visualization and generation of the graphs.

### Repository content

* &#128193; <tt>01_slu</tt> \
The <tt>config.json</tt> file is used to manage the code behavior. It must be modified before execution by choosing one of the following options:
	* <tt>output_dir</tt>
	* <tt>train_file</tt>
	* <tt>train_feats_file</tt>
	* <tt>test_file</tt>
	* <tt>test_feats_file</tt>
	* <tt>handle_unk</tt>
		* <tt>uniform</tt>
		* <tt>cut_off_#</tt>  (where # indicates the cut-off value)
	* <tt>ngram_size</tt>
	* <tt>smoothing</tt>
		* <tt>kneser_ney</tt>
		* <tt>witten_bell</tt>
		* <tt>katz</tt>
		* <tt>absolute</tt>
		* <tt>presmoothed</tt>
		* <tt>unsmoothed</tt>
	* <tt>additional_feature</tt>
		* <tt>lemma</tt>
		* <tt>lemmapos</tt>
		* <tt>tokenpos</tt>
	* <tt>improvement</tt>
		* <tt>wise</tt>
		* <tt>naive</tt>
	* <tt>backoff</tt>
	* <tt>bins</tt>
	* <tt>witten_bell_k</tt>
	* <tt>discount_D</tt>	

&#9432; Please refer to https://goo.gl/zCjxWW (NGramMake) for more details on some of the above options. Informations on <tt>handle_unk</tt>, <tt>additional_feature</tt> and <tt>improvement</tt> options can be found in the report. \

Two additional files are provided:
* <tt>parameters_search</tt> which will iterate over some parameters, generate a <tt>config.json</tt> and call the <tt>slu.py</tt> script
* <tt>k_fold_cross_validation</tt> which will do k-fold cross validation on the train file by creating for each fold a train, test and <tt>config.json</tt>. Then it will call <tt>slu.py</tt>.\
You can configure the cross validation run by editing the <tt>k_fold_config.json</tt>, which has an additional parameter for setting the <tt>k</tt> value

___

&#9888; <b>NOTE</b>: Every time a script is executed, it will always search for a <tt>config.json</tt> file.
___

* &#128193; <tt>02_baseline</tt> \
The <tt>config.json</tt> file is used to manage the code behavior. It must be modified before execution by choosing one of the following options:

	* <tt>random</tt>: the concept to be assigned to a token is randomly chosen among the set of train concepts
	* <tt>chance</tt>: the choice of the concept follows the probability distribution of the concepts within the train set
	* <tt>majority</tt>: the choice of the concept relies on the most common concept in the train

	You can run all the baselines by running the <tt>run_all_baselines.py</tt> script	
* &#128193; <tt>03_data_analysis</tt> \
This folder contains the code for the data analysis
* &#128193; <tt>04_report</tt> \
This folder contains both <tt>.pdf</tt> and LaTeX <tt>.zip</tt> versions of the report

