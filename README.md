# The ELAS simulator

The ELAS simulator is a discrete event simulator for liver allocation in Eurotransplant, which was written in Python 3.12. The conda environment requirements for the ELAS simulator are stored in the `requirements.txt` file. This requirements file may be used to set-up a conda environment with the following commands:

		`conda create --name <env_name> --file requirements.txt`,

where <env_name> should be replaced by the desired name for the environment.

The ELAS simulator was implemented in an object-oriented fashion. The main scripts used to simulate liver allocation are `simulate_elas.py` and `multiprocess_simulations.py`, the latter of which can execute simulations in parallel on multiple cores.

Both `simulate_elas.py` and `multiprocess_simulations.py` take YAML-files as inputs, which specify all settings specific to the simulation scenario. These include paths to input streams, the selected simulation window, with which point score to prioritize candidates (by default, UNOS-MELD), how to convert mortality equivalents to point scores, etc. The format of these YAML-files is discussed at the end of this document. Example YAML-files can be found in the `simulator/sim_yamls` subdirectory. 

The YAML-files used for validation of the ELAS simulator and the 2 case studies included in the manuscript are included in this repository.

# Workflow for the ELAS simulator

The ELAS simulator can be used for policy evaluation in liver allocation. Here, we give some examples of how the ELAS simulator can be used for policy evaluation, and which aspects of the ELAS simulator will have to be modified by end-users to simulate policy alternatives.

1. Changes to ET allocation prioritization rules

> Prioritization of liver transplantation candidates is based on the lexicographical ordering of the so-called ELAS match codes (see the ELAS simulator manuscript for explanation of these match codes). 
> 
> All rules regarding the ET match rules are collected in `simulator/magic_values/rules.py`. End-users of the ELAS simulator may adapt this file to simulate allocation under a different prioritization scheme. An example of a research question for which it would be necessary to adapt the `rules.py` file would be: "What if we introduce international sharing for candidates with high (>35) MELD scores?", which is inspired by the Share-35 rule in the United States.
> 
> To simulate such a scenario, it would be necesary to introduce a new match tier in `rules.py` for candidates with MELD scores exceeding 35. 

2. Changes to the (N)SE/PED-MELD systems

> In the first case study of the ELAS simulator's manuscript, we looked into limiting the (N)SE-MELD scores receivable by Belgian candidates to avoid that candidates who depend on lab-MELD for access to transplantation are crowded out of transplantation.
>
> For this, we made changes to the rules of the exception system (`simulator/magic_values/se_rules.csv`). For instance, in `BELIAC_se_rules_capped_and_slower.csv`, located in the same directory, (N)SEs in Belgium were maximized to a 50\% mortality equivalent. The 90-day upgrades were also slowed down to 5\%-points per 90 days.  
>
> In the YAML-file, `PATH_SE_SETTINGS` is set to simulator/magic_values/BELIAC_se_rules_capped_and_slower.csv` to use these modified (N)SE rules.

3. Alternative MELD scores

> For the second case study for the ELAS simulator, we simulated allocation based on ReMELD-Na instead of UNOS-MELD, with an alternative S-curve for assigning 90-day mortality equivalents. For this, we defined the `score_remeldna` score in YAML-templates as:

	score_remeldna:
	    caps:
	      bili: !!python/tuple
	      - 0.3
	      - 27
	      crea: !!python/tuple
	      - 0.7
	      - 2.5
	      inr: !!python/tuple
	      - 0.1
	      - 2.6
	      sodium: !!python/tuple
	      - 120
	      - 138.6
	    coef:
	      bili: 2.968
	      crea: 9.025
	      inr: 9.518
	      sodium: 0.392
	      revsodiumlncrea: -0.3509

> This parametrization is the ReMELD-Na score as described by Goudsmit et al. (2021). We also indicate in the YAML-file that the laboratory-MELD score should be based on this score, instead of UNOS-MELD:
	
	LAB_MELD: score_remeldna
	
> To base exception-MELD scores on an S-curve developed specifically for ReMELD-Na, we also change the settings for MELD exceptions:

	EXC_SLOPE: 0.2216
	EXC_MELD10_EQUIVALENT: 0.9745
	EXC_MAX: 36
	

4. Measures to rectify sex disparity

> One option to rectify sex disparity would be to give points directly based on female sex, or extra points to short candidates who are predominantly female. For this, one could replace the laboratory-MELD score by a composite score, which prioritizes on factors beyond the laboratory-MELD score. Such a composite-score can be used in the ELAS simulator by adding new coefficients to `ALLOCATION_SCORE` in YAMLs.
>
> For instance, 

		LAB_MELD: score_remeldna
		ALLOCATION_SCORE:
		  coef:
		    patient_sex: 1.3
		    r_height: 0
		    labmeld: 1
		  intercept: 0
		  score_limits: !!python/tuple
		  - 6
		  - 40
		  score_round: true

> would have allocation based on ReMELD-Na, but would give 1.3 extra MELD points to female candidates. 
>
> To include attributes not present in `MatchRecordCurrentELAS` in a composite allocation score, code may have to be adapted (see `simulator/code/current_ELAS/CurrentELAS.py`).


# Data

The ELAS simulator is a data-driven simulator, and simulations are preferably based on data originating from the ET registry. For ELAS simulations, three main data input files are important:

- `patients file` (by default `recipients.csv`):
	+ This defines the static information relating to listings of transplantation candidates. Listings are uniquely defined by a registration ID (`id_registration`), and are coupled to a patient by a recipient ID (`id_recipient`). Information required in this file includes: 
		- the listing date (`inc_date_time`)
		- time to deregistration (`time_to_dereg`), measured in days from the registration date
		- the deregistration reason (`outg_event`)
		- the time since the candidates previous transplantation at listing (`time_since_prev_txp`), measured in days
		- whether the previous transplantation was a living transplantation (`prev_txp_living`),
		- which registration it is for the candidate (`kth_registration`)
		- how many previous transplantations the candidate has had (`n_previous_transplants`)
		- in which center the candidate is registered (`recipient_center`), and in which region (`recipient_region`), and which country (`recipient_country`). 
		- candidate information such as the candidate's blood group (`r_bloodgroup`), candidate height and weight (`r_height` and `r_weight`), whether the candidate also requires a kidney transplant (`listed_kidney`), the candidate's date of birth (`r_dob`), the candidates BMI (`patient_bmi`), when the candidate exited the waitlist, whether the candidate exited the waiting list, and the type of exception the candidate (`type_e`) and since when (`e_since`). 
		
- `donor file` (by default `donors.csv`): 

	+	This defines static information relating to the donors. This includes:
		- the date of the donor's registration or donation (`d_date`)
		- unique identifier assigned to each donor (`id_donor`)
		- the country where the donor is located (`donor_country`)
		- the specific region within the donor's country (`donor_region`)
		- the center responsible for the procurement of the donor's organ (`donor_procurement_center`)
		- the allocation center that managed the distribution of the donor's organ (`donor_alloc_center`). This may differ from the procurement center if the graft is re-allocated through Eurotransplant, for instance after a splitting procedure.
		- the hospital where the donor's organ was procured (`donor_hospital`)
		- detailed type of offer for the organ (`type_offer_detailed`). This is equal to  `4` for a whole liver offer, and equal to `31` through `34 for a split offer (31`: left lateral segment, `32`: extended right lobe, `33`: right lobe, `34`: left lobe). 
		- general type of offer for the organ (`type_offer`): same as type_offer detailed but in text.
		
		- the kth-time this graft was offered (`kth_offer`): may equal 2 for split grafts.
		- the age of the donor (`donor_age`), the donor's blood group (`d_bloodgroup`), the donor's weight (`d_weight`), the donor's height (`d_height`), and the donor's sex (`d_sex`)
		- profile information, such as whether the graft tested positive for HBsAg (`graft_hbsag`), HCV antibodies (`graft_hcvab`), or HBcAb (`graft_hbcab`), whether the donor had sepsis (`graft_sepsis`) or meningitis (`graft_meningitis`), whether the donor had a malignancy (`donor_malignancy`) or drug abuse (`donor_drug_abuse`), whether the donor is marginal according to free text (`donor_marginal_free_text`), whether the donor had a history of tumors (`donor_tumor_history`), whether the donor was marginal (`donor_marginal`), whether the donor was a DCD donor (`graft_dcd`), whether donation followed euthanasia which is a form of DCD (`graft_euthanasia`), the death cause group for the donor (`donor_death_cause_group`), whether the donor has a history of smoking or alcohol abuse (`graft_alcohol_abuse`), and whether the donor had diabetes (`graft_diabetes`)
		- whether the graft was obtained through a rescue operation (`d_rescue`). This item is ignored in case rescue allocation is simulated (by default, rescue allocation is simulated).
	
	
- `patients status updates` file (by default `patstat1.csv`):
	+ Information in this file includes updates relating to most status updates. Required columns for these files are:
		* `id_registration`, which is used to couple status updates to patient listings,
		* `tstart`, which indicates when an update was reported relative to the listing date (in days),
		* `type_status`, which indicates what type of status update it is.
		* `urgency code`, which indicates the candidates current urgency code,
		* `urgency_reason`, which indicates the current urgency reason (for non-transplantable) 
		* `dial_biweekly`, which indicates whether the candidate received dialysis twice in the week before measuring the MELD biomarkers, `crea`, `bili` and `INR`, `sodium`, and `albu`, which are measured creatinine, bilirubin, and INR, sodium, and albumin respectively,
		* `removal_reason` which indicates the reason why the candidate was delisted, if applicable,
		* `variable_status`, which indicates whether a status was activated (1) or deactivated (0),
		* `variable_value`, which gives the value of the update, depending on the type of update, 
		* `variable_detail`, which gives extra information for the update,
		* `inc_date_time`, which indicates the candidates listing date.
	

	+ The type of updates implemented for the ELAS simulator are:
		* `LAB`, which is an update to lab-MELD biomarkers. For lab-MELD updates, variable value indicates the calculated calculated UNOS-MELD score.
		* `NSE`, `SE`, and `PED`, which are changes to the candidates (N)SE-MELD or PED-MELD score. For these updates, `variable_value` indicates the awarded 90-day mortality equivalent and `variable_detail` is the `(N)SE/PED-MELD`-identifier. This identifier has to be defined in the input file for exceptions
		* `URG`, which are changes to the candidates urgency score. This includes exit status (`R` or `D`). The variable value is equal to the urgency status (`NT`: non-transplantable, `T`: transplantable, `D`: waitlist-death, `FU`: transplanted). The variable detail is the reason why a candidate was reported to be non-transplantable.
		* `DM`, which indicates a lab-MELD score was downmarked because the lab-MELD score was not re-certified.
		* `ACO`, which indicates the candidate received an ACO status. ACO statuses are supplied via an external file (by default `aco_statuses.csv`)
		* `PRF`, which indicates it is a status update for a candidate's donor profile. These donor profiles are used to signal that a candidate does not want to be considered for certain donors. Profile status updates are included in a separate file (by default `profiles.csv`).
		* `DIAG`, which are updates to a candidates primary disease group. These are supplied via an external file (by default `tv_diseases.csv`). Note that centers can change the disease group of a candidate during registration, such that these are implemented as a status update.


Finally, an optional input file is:

- `Historic obligations` (by default `obligations.csv`): these contain historic obligations. By supplying this file, the system state of obligations may be initialized to the actual state of obligations on the simulation start date.

Unfortunately, Eurotransplant is not allowed to publicly release these datasets. Instead, synthetic minimal datasets are included in this publicly available repository (`fake_data.zip`), which all have the prefix `fake_`. These synthetic datasets do not contain information from real patients or donors from ET, although distributions of baseline characteristics should be similar. The number of patients and donors per center were also randomized, so these data cannot be used for research. The sole purpose of these datasets is to allow external researchers to work with the ELAS simulator if they do not have real data from Eurotransplant available.

Researchers in simulation of ELAS with realistic data may send a study proposal to the Eurotransplant Liver and Intestine Advisory Committee (ELIAC).


# YAML-files

The YAML file has to specify paths to the following datasets:

- `PATH_RECIPIENTS`, which is the path to candidates
- `PATH_STATUS_UPDATES` which is the path to the status update input stream used for simulation
- `PATH_DONORS`: which is the path to the donor input stream
- `PATH_ACOS`: which is the path to the ACO status input stream
- `PATH_DIAGS`, which is the path to the primary diagnosis groups
- `PATH_PROFILES`, which is the path to candidate donor profiles
- `PATH_SE_SETTINGS`: which is the path to the (N)SE/PED-MELD rules used for simulation


The YAML-file also contains simulation settings, modifiable by the end user. Important settings in this file are:

- `SEED`: which sets the seed for the simulation.
- `SIM_START_DATE` and `SIM_END_DATE`: which define the simulation window
- `RESULTS_FOLDER`: folder to which to write simulation outputs.
- `SAVE_MATCH_LIST`: boolean value whether to write match lists to output files.

- `LAB_MELD`: which sets which score is the laboratory-MELD score
- `ALLOCATION_SCORE`: which defines a formula for the allocation score. By default, this allocation score is equal to the laboratory-MELD score (ALLOCATION_SCORE = 1*LAB_MELD). Other attributes of match records may be added to this formula (for instance, `patient_sex` or `r_height`). 
- SIMULATION_SCORES: Settings for allocation scores which are to be calculated for the simulations. These include UNOS-MELD and ReMELD-Na. 


- `EXC_SLOPE`: which is the slope on MELD for the S-curve used to convert 90-day mortality equivalents to the MELD scale.
- `EXC_MELD10_EQUIVALENT`: which is the 90-day mortality equivalent for a MELD score of 10. This is used to convert 90-day mortality equivalents to the MELD scale.
- `PATH_SE_SETTINGS`: which defines the path to the standard exception rules used to initialize the exception score module.

- `PATIENT_ACC_POLICY` and `CENTER_ACC_POLICY`: by default, 'LR' which means that both patient and center offer acceptance is simulated with logistic regressions. Alternatively, they may be set to 'always', which means that candidates and centers always accept graft offers.
- `SIMULATE_RANDOM_EFFECTS`: whether to simulate random effects for prediction of graft offer acceptance decisions.
- VARCOMPS_RANDOM_EFFECTS: variance components used to simulate random effects for prediction of graft offer acceptance decisions.

- `LOAD_RETXS_FROM` and `LOAD_RETXS_TO`: these define the time window from which repeat transplantations are loaded. These re-transplantations are used to construct synthetic re-registrations for candidates in case they are listed for a repeat transplantation within the simulation window.

Finally, the YAML-files contain filenames for the output files. File names which are to be specified are:

- `PATH_TRANSPLANTATIONS`, to which information on transplantations is written at the end of simulations,
- `PATH_EXITS`, to which waitlist exits are written (deaths and removals),
- `PATH_FINAL_PATIENT_STATUS`, to which all patient statuses are written at the end of simulation,
- `PATH_DISCARDS`, to which discards are written at the end of simulation (if applicable),
- `PATH_MATCH_LISTS`, to which match lists are written if `SAVE_MATCH_LISTS` is set to True

**YAML-templates**

We anticipate that end-users of the ELAS simulator will want to a simulation scenario multiple times to assess variability in outcomes. To keep simulation settings the same across yaml-files, it is helpful to use yaml-templates as is illustrated in the `sim_yamls/templates/` subfolder. Attribute values in these templates are parametrized with double brackets. An example of such a template is:


			SEED: {{seed}}
			KTH_STATUS_FILE: {{kth_stat}}
			PATH_TRANSPLANTATIONS: transplantations_k{{kth_stat}}_s{{sim_seed}}.csv

For a seed of 1 and the first status update file, this template will lead to the following yaml-file:

			SEED: 1
			KTH_STATUS_FILE: 1
			PATH_TRANSPLANTATIONS: transplantations_k1_s1.csv

There are three \*.Rmd in the `sim_yamls/` directory, which construct YAML-files necessary for the validation of the ELAS simulator, the two policy evaluation case studies included in the manuscript, and validation with the fake data.
