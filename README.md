# thermal-ambiguity-ToH

This repo contains the code and data for the human and robot studies in "Material Recognition via Heat Transfer Given Ambiguous Initial Conditions."

### Paper:
Bhattacharjee, Tapomayukh\*, Henry M. Clever\*, Joshua Wade, and Charles C. Kemp. "Material Recognition via Heat Transfer Given Ambiguous Initial Conditions." IEEE Transactions on Haptics, 2021.

### Human study:
We conducted the human study using data from 13 pilot participants and 32 study participants. The demographic information and data from these participants are contained in `pilot_data.txt` and `study_data.txt`, respectively. These text files are created from the `save_as_txt.py` script. This data is analyzed with the following steps and corresponding code:

* In a pilot study with 13 participants, we found that finger temperature can vary substantially between trials. Thus, we monitored finger temperature between trials and withdrew from statistical inference the trials where the finger temperature deviates too far from the ambiguous condition. We used the pilot data to find an optimum threshold for this deviation, which is described in Section V-C. To compute this optimum threshold, run `python compute_human_var_threshold.py`. It should output a threshold of 3.5 degrees Celsius.
* From this pilot data and threshold amount, we conducted a power analysis to determine how many subjects should be in the main study. We used the power analysis formulation from the following paper: J.-M. Nam, "Power and sample size requirements for non-inferiority in studies comparing two matched proportions where the events are correlated,” Comput. Statist. I Data Anal., vol. 55, no. 10, pp. 2880–2887, 2011. To compute the number of participants in the study, run `python power_analysis_samplesize_nam2011.py`. It should output a float between 28 and 29. We round this up to 30 for the study.
* 
