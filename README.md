# thermal-ambiguity-ToH

This repo contains the code and data for the human and robot studies in "Material Recognition via Heat Transfer Given Ambiguous Initial Conditions."

### Paper:
Bhattacharjee, Tapomayukh\*, Henry M. Clever\*, Joshua Wade, and Charles C. Kemp. "Material Recognition via Heat Transfer Given Ambiguous Initial Conditions." IEEE Transactions on Haptics, 2021.

### Human study:
We conducted the human study using data from 13 pilot participants and 32 study participants. The demographic information and data from these participants are contained in `pilot_data.txt` and `study_data.txt`, respectively. These text files are created from the `save_as_txt.py` script. This data is analyzed with the following steps and corresponding code scripts, which are in `./human_study/`:

* In a pilot study with 13 participants, we found that finger temperature can vary substantially between trials. Thus, we monitored finger temperature between trials and withdrew from statistical inference the trials where the finger temperature deviates too far from the ambiguous condition. We used the pilot data to find an optimum threshold for this deviation, which is described in Section V-C. To compute this optimum threshold, run `python compute_human_var_threshold.py`. It should output a threshold of 3.5 degrees Celsius.
* From this pilot data and threshold amount, we conducted a power analysis to determine how many subjects should be in the main study. We used the power analysis formulation from the following paper: *J.-M. Nam, "Power and sample size requirements for non-inferiority in studies comparing two matched proportions where the events are correlated,” Comput. Statist. I Data Anal., vol. 55, no. 10, pp. 2880–2887, 2011.* To compute the number of participants in the study, run `python power_analysis_samplesize_nam2011.py`. It should output a float between 28 and 29. We round this up to 30 for the study, and increased it to 32 during the study because two of the 32 subjects had no trials where the finger temperature was within the threshold.
* To plot out the data from paired trials among all 32 subjects on a histogram, run `python plot_human_data_hist.py`. This compares how materials were classified to the finger temperature deviation from ambiguous conditions.
* To run the similarity test and compute significance of similarity, run `python compute_human_data_similarity.py`. This uses the similarity testing for clustered pairs that is formulated in the following paper: *J. Nam and D. Kwon, "Non-inferiority tests for clustered matched-pair data," Statist. Med., vol. 28, no. 12, pp. 1668–1679, 2009.* It outputs a p-value that is well below the threshold for statistical significance.

### Robot study:
We conducted a robot study to match the human study that mimics a set of 30 people touching the same order of samples. This data is in `./robot_study/test/`. Code to analyze the data is in `./robot_study/run_classifiers.py`. The other python scripts in `./robot_study/` are for controlling the robot with an active heated and a passive temperature sensor.
