#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

blood group compatibility rules

@author: H.C. de Ferrante
"""

import sys
sys.path.append('./')

# Blood group compatibilities
BGC_FULL = 'bgc_full'
BGC_TYPE1 = 'bgc_type1'
BGC_TYPE2 = 'bgc_type2'

# Column names for standard exception rules.
ID_SE = 'id_se'
TYPE_E = 'type_e'
E_SINCE = 'e_since'
TIME_IN_E = 'time_in_e'
DISEASE_SE = 'disease_se'
DISEASE_GROUP = 'disease_group'
DISEASE_SINCE = 'disease_since'
SE_COUNTRY = 'se_country'
SE_UPGRADE = 'upgrade'
SE_UPGRADE_INTERVAL = 'upgrade_interval'
LAB_MELD_BONUS = 'lab_meld_bonus'
LAB_MELD_BONUS_AMOUNT = 'lab_meld_bonus_amount'
MAX_EQUIVALENT = 'max_equivalent'
INITIAL_EQUIVALENT = 'initial_equivalent'
ALLOWED_EQUIVALENTS_BY = 'allowed_eqs_by'
ALLOWED_EQS_OFFSET = 'allowed_eqs_offset'
MAX_AGE_UPGRADE = 'max_age_upgrade'
REPLACEMENT_SE = 'replacement_se'

# Match list information
RECIPIENT_OFFERED = 'recipient_offered'
GEOGRAPHY_MATCH = 'match_geography'
LOCAL_MATCH_CROATIA = 'local_match_croatia'
MATCH_ABROAD = 'match_abroad'
CENTER_OFFER = 'center_offer'
ALLOCATION_LOC = 'alloc_loc'
ALLOCATION_REG = 'alloc_reg'
ALLOCATION_NAT = 'alloc_nat'
ALLOCATION_INT = 'alloc_int'
R_PED = 'r_pediatric'
D_PED = 'd_pediatric'
BG_IDENTICAL = 'bg_identical'
R_LOWWEIGHT = 'r_lowweight'
ID_MTR = 'id_mtr'
MATCH_DATE = 'date_match'
ID_TXP = 'id_txp'
TYPE_TRANSPLANTED = 'txp_liver_transplanted'
N_OFFERS = 'n_offers'
N_PROFILE_TURNDOWNS = 'n_profile_turndowns'
N_CENTER_OFFERS = 'n_center_offers'

ID_DONOR = 'id_donor'
FIRST_OFFER_TYPE = 'first_offer_type'
D_DCD = 'graft_dcd'
D_WEIGHT = 'd_weight'
D_HEIGHT = 'd_height'
D_AGE = 'donor_age'
D_BLOODGROUP = 'd_bloodgroup'
TYPE_OFFER = 'type_offer'
TYPE_OFFER_DETAILED = 'type_offer_detailed'
D_COUNTRY = 'donor_country'
D_REGION = 'donor_region'
D_ALLOC_COUNTRY = 'donor_alloc_country'
D_ALLOC_REGION = 'donor_alloc_region'
D_ALLOC_CENTER = 'donor_alloc_center'
D_PROC_CENTER = 'donor_procurement_center'
D_HOSPITAL = 'donor_hospital'
MTCH_TIER = 'mtch_tier'
MTCH_LAYER = 'mtch_layer'
MTCH_LAYER_REG = 'mtch_layer_reg'
MTCH_OBL = 'mtch_obl'
MTCH_LAYER_MELD = 'mtch_layer_meld'
MTCH_LAYER_WT = 'mtch_layer_wt'
MTCH_DATE_TIEBREAK = 'mtch_date_tiebreak'
MTCH_CODE = 'mtch_code'
REC_OFFERED = 'recipient_offered'
R_DOB = 'r_dob'
R_MATCH_AGE = 'r_match_age'
MATCH_CRITERIUM = 'match_criterium'
RESCUE_PRIORITY = 'rescue_priority'
URGENCY_CODE = 'urgency_code'
INIT_URG_CODE = 'init_urg_code'
ANY_ACTIVE = 'any_active'
ANY_HU = 'any_hu'
R_ACO = 'r_aco'
INTERREGIONAL_ACO = 'interregional_aco'
ACO_SINCE = 'aco_since'
R_BLOODGROUP = 'r_bloodgroup'
REC_REQ = 'rec_req'
REQ_INT = 'req_int'
RECIPIENT_COUNTRY = 'recipient_country'
RECIPIENT_REGION = 'recipient_region'
RECIPIENT_CENTER = 'recipient_center'
R_WEIGHT = 'r_weight'
R_HEIGHT = 'r_height'
PATIENT_BMI = 'patient_bmi'
RNK_OTHER = 'rnk_other'
RNK_ACCEPTED = 'rnk_accepted'
ACCEPTED = 'accepted'
ACCEPTANCE_REASON = 'acceptance_reason'
PROB_ACCEPT_P = 'prob_accept_p'
DRAWN_PROB = 'drawn_prob_p'
DRAWN_PROB_C = 'drawn_prob_c'
PROB_ACCEPT_C = 'prob_accept_c'
PROB_TILL_RESCUE = 'prob_till_rescue'
CBH_RESCUE = 'cum_bh'
N_OFFERS_TILL_RESCUE = 'n_offers_till_rescue'
ACTION_CODE = 'action_code'
ACTION_CODE_DESC = 'action_code_desc'
FINAL_REC_URG_AT_TRANSPLANT = 'final_rec_urg_at_transplant'
LAST_NONNT_HU = 'last_nonnt_hu'
MATCH_COMMENT = 'match_comment'
MELD_NAT_MATCH = 'meld_nat_match'
MELD_INT_MATCH = 'meld_int_match'
ALLOC_SCORE_NAT = 'alloc_score_nat'
ALLOC_SCORE_INT = 'alloc_score_int'
NSE_DELAY = 'nse_delay'

OFFERED_TO = 'offered_to'

# Additional information needed for post-txp survival model
DELTA_WEIGHT_NOSPLIT_MA30 = 'delta_weight_nosplit_ma30'
YEAR_TXP_RT2014 = 'year_txp_rt2014'
POSTTXP_EVENT_TYPE = 'event_type'

# Aco info
ACO_DATE = 'aco_date'
ACO_STATUS = 'aco_status'

# Recipient information
ID_RECIPIENT = 'id_recipient'
ID_REGISTRATION = 'id_registration'
ID_REREGISTRATION = 'id_reregistration'
TIME_REGISTRATION = 'inc_date_time'
NTH_TRANSPLANT = 'n_previous_transplants'
KTH_REG = 'kth_registration'
LISTING_DATE = 'inc_date_time'
TIME_TO_DEREG = 'time_to_dereg'
TIME_WAITED = 'time_waited'
TIME_SINCE_PREV_TXP = 'time_since_prev_txp'
TIME_SINCE_PREV_TXP_CAT = 'time_since_prev_txp_cat'
PREV_TXP_LIVING = 'prev_txp_living'
HU_ELIGIBLE_REREG = 'hu_eligible_rereg'
LISTED_KIDNEY = 'listed_kidney'
PATIENT_SEX = 'patient_sex'
PATIENT_FEMALE = 'patient_female'

# Retransplantation information
TYPE_RETX = 'type_retx'
NO_RETRANSPLANT = 'retx_none'
RETRANSPLANT = 'retx_real'
IS_RETRANSPLANT = 'retransplant'
DATE_RETRANSPLANT = 'date_retx'
RETRANSPLANT_DURING_SIM = 'retx_simperiod'
PREV_TX_DATE = 'prev_tx_date'
PATIENT_RELISTING = 'rereg'
PATIENT_FAILURE = 'patient_failure'
PATIENT_RELISTING_DATE = 'patient_relist_date'
PATIENT_FAILURE_DATE = 'patient_failure_date'
TIME_TO_REALEVENT = 'time_to_realevent'
TIME_LIST_TO_REALEVENT = 'time_list_to_realevent'
TIME_TO_REREG = 'time_to_rereg'
EXIT_STATUS_REREG = 'exit_status_rereg'
RETRANSPLANTED = 'retx'
EVENT = 'event'
TIME_TO_EVENT = 'time_to_event'
TIME_TO_PATIENT_FAILURE = 'time_to_patient_failure'
TIME_TO_RETX = 'time_to_retx'
TIME_TO_CENS = 'time_to_cens'

# Historic obligatons
OBL_INSERT_DATE = 'obl_insert_date'
OBL_START_DATE = 'obl_start_date'
OBL_END_DATE = 'obl_end_date'
OBL_BG = 'obl_bg'

# Retransplantation information for post-transplant module
AGE_PREV_TXP = 'age_prev_txp'
OFFER = 'offer'

# Donor information
D_DATE = 'd_date'
KTH_OFFER = 'kth_offer'
D_HBSAG = 'graft_hbsag'
D_HBSAG = 'graft_hbsag'
D_HCVAB = 'graft_hcvab'
D_HBCAB = 'graft_hbcab'
D_SEPSIS = 'graft_sepsis'
D_MENINGITIS = 'graft_meningitis'
D_MALIGNANCY = 'donor_malignancy'
D_DRUG_ABUSE = 'donor_drug_abuse'
D_MARGINAL = 'donor_marginal'
D_MARGINAL_FREE_TEXT = 'donor_marginal_free_text'
D_TUMOR_HISTORY = 'donor_tumor_history'
D_DCD = 'graft_dcd'
D_EUTHANASIA = 'graft_euthanasia'
DONOR_DEATH_CAUSE_GROUP = 'donor_death_cause_group'
D_SEX = 'd_sex'
D_SMOKING = 'graft_smoking'
D_ALCOHOL_ABUSE = 'graft_alcohol_abuse'
D_DIABETES = 'graft_diabetes'
D_ET_DRI = 'graft_et_dri'
D_ET_DRI_TRUNC = 'graft_et_dri_trunc'
D_RESCUE = 'd_rescue'
RESCUE = 'rescue'
D_GGT = 'graft_gamma_gt_u_per_l'
DONOR_BMI_CAT = 'donor_bmi_cat'
DONOR_BMI = 'donor_bmi'
GRAFT_SPLIT = 'graft_split'
TX_BLOODGROUP_MATCH = 'txp_blood_group_match'

# Status update information
TYPE_UPDATE = 'type_status'
EXIT_DATE = 'exit_date'
EXIT_STATUS = 'exit_status'
TSTART = 'tstart'
URGENCY_REASON = 'urgency_reason'
REMOVAL_REASON = 'removal_reason'
DIAL_BIWEEKLY = 'dial_biweekly'
MELD_LAB = 'meld_lab'
CREA = 'crea'
CREA_NODIAL = 'crea_nodial'
BILI = 'bili'
INR = 'inr'
SODIUM = 'sodium'
ALBU = 'albu'
R_DOB = 'r_dob'
R_AGE_LISTING = 'age_at_listing'
STATUS_ACTION = 'variable_status'
STATUS_VALUE = 'variable_value'
STATUS_DETAIL = 'variable_detail'

# Status update types
SE = 'SE'
NSE = 'NSE'
PED = 'PED'
DM = 'DM'
LAB = 'LAB'
URG = 'URG'
FU = 'FU'
R = 'R'
D = 'D'
FU = 'FU'
SFU = 'SFU'
HU = 'HU'
ACO = 'ACO'
DIAG = 'DIAG'
PRF = 'PRF'
T = 'T'
NT = 'NT'

T1 = 'T1'
T3 = 'T3'
FP = 'FP'
RR = 'RR'
CR = 'CR'

# Match criteria
LOC = 'LOC'
REG = 'REG'
NAT = 'NAT'
INT = 'INT'

# Match geography
A = 'A'  # abroad
H = 'H'  # home
R = 'R'  # regional
L = 'L'  # local

# Event types
PAT = 'patient'
DON = 'don'

# Indicator columns for blood group rules.
MELD_GE_ETTHRESH = 'MELD_GE_ETTHRESH'
PATIENT_IS_HU = 'patient_is_hu'
ANY_OBL = 'any_obl'

# Table names for BG
TAB1 = 'tab1'
TAB2 = 'tab2'
TAB3 = 'tab3'
BG_TAB_COL = 'elig_tab'  # Rule based on table
BG_RULE_COL = 'bg_rule'  # Applied rule
BG_COMP = 'bg_comp'      # Whether donor/patient is BG compatible
BG_PRIORITY = 'bg_priority'

# Columns for obligations
N_OPEN_OBL = 'n_open_obl'
OBL_DEBTOR = 'obl_debtor'
OBL_CREDITOR = 'obl_creditor'

# Match rank tiers
TIER_A = 'A'
TIER_B = 'B'
TIER_C = 'C'
TIER_D = 'D'
TIER_E = 'E'
TIER_F = 'F'

SPLIT = 'Split'
WLIV = 'WLiv'

# Type of MELD scores
SCORE_DYNREMELD = 'score_dynremeld'
SCORE_REMELD = 'score_remeld'
SCORE_REMELDNA = 'score_remeldna'
SCORE_UNOSMELD = 'score_unosmeld'
SCORE_LABHISTORIC = 'score_historic'

# Modified MELD score names for saving outputs
MELD_SE = 'meld_se'
MELD_NSE = 'meld_nse'
MELD_PED = 'meld_ped'

# For calculation of biomarkerss
INTERCEPT = 'intercept'
COEF = 'coef'
TRAFOS = 'trafos'
CAPS = 'caps'
SCORE_LIMITS = 'score_limits'
SCORE_ROUND = 'score_round'

# Transformations
IDENTITY_TRAFO = 'i'
LOG = 'log'

# Center offer columns. The first 2 are local offers,
# the second 2 are non-local offers
# (obligations & non-local DCDs in Belgium)
LOCAL_PEDIATRIC_CENTER_OFFER = 'ped_center_offer'
LOCAL_GENERAL_CENTER_OFFER = 'center_offer'

PEDIATRIC_CENTER_OFFER_INT = 'ped_center_offer_int'
GENERAL_CENTER_OFFER_INT = 'center_offer_int'

# Distinguish between donor and center offer groups.
DON_CENTER_OFFER_GROUP = 'don_center_offer_group'
REC_CENTER_OFFER_GROUP = 'rec_center_offer_group'
OBL_APPLICABLE = 'obl_applicable'
PATIENT_RANK = 'recipient_rank'
CENTER_RANK = 'center_rank'
RANK = 'rnk_other'

# Blood groups
BG_A = 'A'
BG_AB = 'AB'
BG_O = 'O'
BG_B = 'B'

# Profile variables
PRF_TYPE = 'prf_type'
PROFILE_COMPATIBLE = 'prof_compatible'
PROFILE_MIN_DONOR_AGE = 'profile_min_donor_age'
PROFILE_MAX_DONOR_AGE = 'profile_max_donor_age'
PROFILE_MIN_WEIGHT = 'profile_min_weight'
PROFILE_MAX_WEIGHT = 'profile_max_weight'
PROFILE_DCD = 'profile_dcd'
PROFILE_HBSAG = 'profile_hbsag'
PROFILE_HCVAB = 'profile_hcvab'
PROFILE_HBCAB = 'profile_hbcab'
PROFILE_EXPLANTED_LIVER = 'profile_explanted_liver'
PROFILE_SEPSIS = 'profile_sepsis'
PROFILE_MENINGITIS = 'profile_meningitis'
PROFILE_MALIGNANCY = 'profile_malignancy'
PROFILE_DRUG_ABUSE = 'profile_drug_abuse'
PROFILE_MARGINAL = 'profile_marginal'
PROFILE_RESCUE = 'profile_rescue'
PROFILE_LLS = 'profile_lls'
PROFILE_ERL = 'profile_erl'
PROFILE_RL = 'profile_rl'
PROFILE_LL = 'profile_ll'
PROFILE_EUTHANASIA = 'profile_euthanasia'

# Transport information
TRAVEL_MODE = 'travel_mode'
TRAVEL_TIME = 'travel_time'
FROM_HOSPITAL = 'from_hospital'
TO_CENTER = 'to_center'
DRIVE = 'drive'
PLANE = 'plane'
DRIVING_DISTANCE = 'driving_distance'
DRIVING_TIME = 'driving_time'
TOTAL_TIME_BYPLANE = 'total_time_byplane'
