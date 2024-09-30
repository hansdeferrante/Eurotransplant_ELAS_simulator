#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

Magic values for the simulator. Only magic values
which are to some extent immodifiable are included.

@author: H.C. de Ferrante
"""

from datetime import timedelta, datetime
from typing import Any
import numpy as np
import simulator.magic_values.magic_values_rules as mgr
import simulator.magic_values.column_names as cn
import simulator.magic_values.column_groups as cg

DAYS_PER_YEAR = timedelta(days=365.25)
DEFAULT_DATE_TIME = '%Y-%m-%d %H:%M:%S'

# Transformations
def identity(__x: Any):
    """Identitity transformation function"""
    return __x


def square(__x: Any):
    """Identitity transformation function"""
    return __x**2


def revNa(__x: Any, max=138.6):
    if np.isnan(__x):
        return 0
    elif __x:
        return max - __x
    else:
        return 0


TRAFOS = {
    cn.IDENTITY_TRAFO: identity,
    cn.LOG: np.log,
    'rev': revNa
}
REFERENCE = 'reference'

# Directory with simulation settings
DIR_SIM_SETTINGS = 'simulator/sim_yamls/'
DIR_ACCEPTANCE_COEFS = 'simulator/magic_values/acceptance/'
DIR_POSTTXP_COEFS = 'simulator/magic_values/post_txp/'
DIR_TEST_LR = 'data/test/'

# Path to rescue probs
PATH_RESCUE_PROBABILITIES = (
    'simulator/magic_values/probabilities_rescue_triggered.csv'
)
PATH_RESCUE_COX_BH = (
    'simulator/magic_values/acceptance/bh_rescue_triggered.csv'
)
PATH_RESCUE_COX_COEFS = (
    'simulator/magic_values/acceptance/coefs_rescue_triggered.csv'
)

# Paths to files with travel information
PATH_DRIVING_DISTANCE = (
    'simulator/magic_values/travel_distances/driving_distances.csv'
)
PATH_DRIVING_TIMES = (
    'simulator/magic_values/travel_distances/driving_times.csv'
)
PATH_FLIGHT_TIMES = (
    'simulator/magic_values/travel_distances/total_flight_times.csv'
)

# Time / km above which plane is transport mode of choice.
# Note, the acceptance model was calibrated on these.
MAX_DRIVE_TIME: float = 5
MAX_DRIVE_KM: float = 300

# Paths relevant for the acceptance module
ACCEPTANCE_PATHS = {
    k: DIR_ACCEPTANCE_COEFS + v for k, v in {
        'cd': 'coefs_center_and_obligations.csv',
        'rd': 'coefs_recipient_driven.csv',
        'rd_huaco': 'coefs_recipient_driven_huaco.csv',
        'rd_reg': 'coefs_recipient_driven_reg.csv',
        'rd_adult_huaco': 'coefs_adult_huaco.csv',
        'rd_adult_reg': 'coefs_adult_reg.csv',
        'rd_ped_huaco': 'coefs_ped_huaco.csv',
        'rd_ped_reg': 'coefs_ped_reg.csv',
        'pcd_ped': 'coefs_pcd_ped_reg.csv',
        'pcd_adult': 'coefs_pcd_adult_reg.csv'
    }.items()
}
LR_TEST_FILES = {
    k: DIR_TEST_LR + v for k, v in {
        'cd': 'acceptance_cd.csv',
        'rd': 'acceptance_pd.csv',
        'rd_huaco': 'acceptance_pd_huaco.csv',
        'rd_reg': 'acceptance_pd_reg.csv',
        'rd_adult_huaco': 'acceptance_pd_huaco_adult.csv',
        'rd_adult_reg': 'acceptance_pd_reg_adult.csv',
        'rd_ped_huaco': 'acceptance_pd_huaco_ped.csv',
        'rd_ped_reg': 'acceptance_pd_reg_ped.csv',
        'pcd_ped': 'acceptance_pcd_reg_ped.csv',
        'pcd_adult': 'acceptance_pcd_reg_adult.csv'
    }.items()
}
PATH_COEFFICIENTS_SPLIT = 'simulator/magic_values/acceptance/coefs_split.csv'

# Paths relevant for post-transplant survival predictions
POSTTXP_RELISTPROB_PATHS = {
    k: DIR_POSTTXP_COEFS + v for k, v in {
        cn.T: 'prob_relist_T.csv',
        cn.HU: 'prob_relist_HU.csv'
    }.items()
}
POSTTXP_SURV_PATHS = {
    k: DIR_POSTTXP_COEFS + v for k, v in {
        cn.T: 'posttx_coefs_T.csv',
        cn.HU: 'posttx_coefs_HU.csv'
    }.items()
}
POSTTXP_SURV_TESTPATHS = {
    k: DIR_TEST_LR + v for k, v in {
        cn.T: 'posttx_testcases_T.csv',
        cn.HU: 'posttx_testcases_HU.csv'
    }.items()
}

CENTER_OFFER_INHERIT_COLS = {
    cn.ID_MTR, cn.MATCH_DATE, cn.N_OPEN_OBL,
    cn.ANY_OBL, cn.D_ALLOC_CENTER,
    cn.D_ALLOC_REGION, cn.D_ALLOC_COUNTRY,
    cn.TYPE_OFFER, cn.TYPE_OFFER_DETAILED,
    cn.MATCH_CRITERIUM, cn.GEOGRAPHY_MATCH,
    cn.MATCH_ABROAD, cn.LOCAL_PEDIATRIC_CENTER_OFFER,
    cn.PEDIATRIC_CENTER_OFFER_INT,
    cn.LOCAL_GENERAL_CENTER_OFFER, cn.GENERAL_CENTER_OFFER_INT,
    cn.D_PED, cn.OBL_APPLICABLE, cn.RECIPIENT_COUNTRY,
    cn.ALLOCATION_REG,
    cn.REC_CENTER_OFFER_GROUP, cn.DON_CENTER_OFFER_GROUP
}

OFFER_INHERIT_COLS = {
    'donor': [
        cn.D_AGE, cn.DONOR_BMI_CAT, cn.DONOR_DEATH_CAUSE_GROUP,
        cn.D_TUMOR_HISTORY, cn.D_MARGINAL_FREE_TEXT,
        cn.D_DCD, cn.D_DIABETES, cn.RESCUE, cn.DONOR_BMI
    ],
    'patient': [
        cn.LISTED_KIDNEY,
        cn.PATIENT_SEX, cn.TYPE_E,
        cn.URGENCY_CODE, cn.PATIENT_BMI,
        cn.IS_RETRANSPLANT,
        cn.ID_REGISTRATION
    ]
}

# Settings for post-transplant module
POSTTXP_DISCRETE_MATCH_VARS = [cn.HU_ELIGIBLE_REREG, cn.RECIPIENT_COUNTRY]
POSTTXP_CONTINUOUS_MATCH_VARS = {
    cn.RETRANSPLANT: [
        cn.AGE_PREV_TXP, cn.TIME_SINCE_PREV_TXP, cn.TIME_LIST_TO_REALEVENT
    ],
    cn.OFFER: [cn.R_MATCH_AGE, cn.TIME_TO_REREG, cn.TIME_LIST_TO_REALEVENT]
}
POSTTXP_MIN_MATCHES = 5
POSTTXP_MATCH_CALIPERS = [20.0, 1.0, 1.0]


def log_ceil(x):
    return np.log(np.ceil(x+2))


POSTTXP_TRANSFORMATIONS = [identity, log_ceil, log_ceil]

POSTTXP_COPY_VARS = [
    cn.ID_RECIPIENT, cn.R_BLOODGROUP, cn.R_WEIGHT,
    cn.R_HEIGHT, cn.PATIENT_BMI, cn.RECIPIENT_CENTER,
    cn.RECIPIENT_COUNTRY, cn.RECIPIENT_REGION,
    cn.REC_CENTER_OFFER_GROUP, cn.R_DOB, cn.PATIENT_SEX,
    'profile'
]


# Allowed values for various factors.
ALLOWED_BLOODGROUPS = set(['A', 'B', 'AB', 'O'])
ALLOWED_STATUSES = set([cn.NT, cn.T, cn.HU])
EXIT_STATUSES = set([cn.FU, cn.R, cn.D])
ACTIVE_STATUSES = set([cn.T, cn.HU])
ALL_STATUS_CODES = ALLOWED_STATUSES.union(EXIT_STATUSES)


# Countries & specification of country rules.
ET_COUNTRIES = {
        mgr.NETHERLANDS, mgr.BELGIUM, mgr.GERMANY, mgr.AUSTRIA, mgr.HUNGARY,
        mgr.SLOVENIA, mgr.CROATIA
}
ET_COUNTRIES_OR_OTHER = ET_COUNTRIES.union(set(mgr.OTHER))
RECIPIENT_DRIVEN_COUNTRIES = {mgr.GERMANY, mgr.BELGIUM, mgr.NETHERLANDS}
CENTER_DRIVEN_COUNTRIES = {mgr.SLOVENIA, mgr.CROATIA, mgr.HUNGARY, mgr.AUSTRIA}
OBLIGATION_COUNTRIES = {
    mgr.NETHERLANDS, mgr.BELGIUM, mgr.GERMANY, mgr.HUNGARY,
    mgr.SLOVENIA, mgr.CROATIA
}
OBLIGATION_CENTERS = {
    mgr.CENTER_INNSBRUCK, mgr.CENTER_GRAZ, mgr.CENTER_VIENNA
}
OBLIGATION_PARTIES = OBLIGATION_COUNTRIES.union(OBLIGATION_CENTERS)

# DCD accepting countries
DCD_COUNTRIES = set([mgr.NETHERLANDS, mgr.AUSTRIA, mgr.BELGIUM])
DCD_ACCEPTING_COUNTRIES = DCD_COUNTRIES.union(mgr.SLOVENIA)

# Countries which use centers for obligations
CNTR_OBLIGATION_CNTRIES = set([mgr.AUSTRIA])

# Save transplantation probabilities
WINDOWS_TRANSPLANT_PROBS = (
    28, 90, 365
)
PREFIX_SURVIVAL = 'psurv_posttxp'
COLS_TRANSPLANT_PROBS = list(
    f'{PREFIX_SURVIVAL}_{w}' for w in WINDOWS_TRANSPLANT_PROBS
)

# Pre-specify columns for simulation results
OUTPUT_COLS_DISCARDS = (
    'reporting_date',
    cn.ID_DONOR, cn.D_WEIGHT, cn.D_DCD,
    'donor_age', cn.D_BLOODGROUP, cn.N_OPEN_OBL,
    cn.N_OFFERS, cn.N_CENTER_OFFERS, cn.N_PROFILE_TURNDOWNS,
    cn.TYPE_OFFER_DETAILED
)
OUTPUT_COLS_PATIENTS = (
    cn.ID_RECIPIENT, cn.ID_REGISTRATION,
    cn.RECIPIENT_CENTER, cn.LISTING_DATE,
    cn.EXIT_STATUS, cn.EXIT_DATE, cn.URGENCY_REASON,
    cn.FINAL_REC_URG_AT_TRANSPLANT,
    cn.LAST_NONNT_HU,
    cn.TIME_SINCE_PREV_TXP,
    cn.TYPE_RETX,
    cn.RECIPIENT_COUNTRY,
    cn.TYPE_E, cn.E_SINCE,
    cn.R_ACO, cn.ANY_ACTIVE,
    cn.R_DOB, cn.ANY_HU,
    cn.PATIENT_SEX,
    cn.INIT_URG_CODE,
    cn.R_BLOODGROUP,
    cn.DISEASE_SINCE,
    cn.DISEASE_GROUP,
    cn.URGENCY_CODE,
    cn.MELD_INT_MATCH,
    cn.MELD_LAB,
    cn.DIAL_BIWEEKLY, cn.LISTED_KIDNEY,
    cn.R_HEIGHT
)
OUTPUT_COLS_EXITS = (
    cn.ID_RECIPIENT, cn.ID_REGISTRATION,
    cn.TYPE_RETX, cn.ID_DONOR, cn.D_DCD, cn.EXIT_STATUS,
    cn.URGENCY_REASON, cn.LISTING_DATE,
    cn.EXIT_DATE, cn.MATCH_CRITERIUM, cn.GEOGRAPHY_MATCH,
    cn.MATCH_ABROAD, cn.RECIPIENT_CENTER,
    cn.RECIPIENT_COUNTRY, cn.R_BLOODGROUP, cn.D_BLOODGROUP,
    cn.MATCH_DATE, cn.PATIENT_RANK,
    cn.RANK, cn.MELD_LAB,
    cn.D_ALLOC_CENTER, cn.D_ALLOC_COUNTRY,
    cn.R_MATCH_AGE, cn.R_PED,
    cn.MELD_NAT_MATCH, cn.MELD_INT_MATCH,
    cn.ALLOC_SCORE_INT, cn.ALLOC_SCORE_NAT,
    cn.SE, cn.NSE, cn.PED,
    cn.TYPE_E, cn.ID_SE, cn.E_SINCE,
    cn.R_ACO, cn.PATIENT_IS_HU,
    cn.TYPE_TRANSPLANTED, cn.PATIENT_SEX,
    cn.D_ET_DRI, cn.D_ET_DRI_TRUNC,
    cn.ANY_HU, cn.ACCEPTANCE_REASON,
    cn.OFFERED_TO, cn.DISEASE_GROUP, cn.DISEASE_SINCE,
    cn.DIAL_BIWEEKLY, cn.LISTED_KIDNEY,
    cn.PROFILE_COMPATIBLE, cn.TYPE_OFFER_DETAILED,
    cn.PROB_ACCEPT_C, cn.PROB_ACCEPT_P, cn.DRAWN_PROB, cn.DRAWN_PROB_C
 ) + tuple(cg.MTCH_COLS) + tuple(COLS_TRANSPLANT_PROBS)

OUTPUT_COLS_EXIT_CONSTRUCTED = (
    cn.ID_REREGISTRATION, cn.MTCH_CODE,
    cn.TIME_WAITED, cn.TIME_IN_E, cn.CENTER_RANK
)

MATCH_INFO_COLS = (
    cn.ID_MTR, cn.MATCH_DATE, cn.OFFERED_TO, cn.N_OFFERS, cn.ID_DONOR,
    cn.D_DCD, cn.D_COUNTRY, cn.D_ALLOC_COUNTRY,
    cn.D_ALLOC_REGION, cn.D_ALLOC_CENTER,
    cn.D_WEIGHT, cn.D_PED, cn.D_AGE,
    cn.TYPE_OFFER, cn.TYPE_OFFER_DETAILED, cn.OBL_APPLICABLE,
    cn.N_OPEN_OBL, cn.RECIPIENT_CENTER, cn.ID_RECIPIENT, cn.ID_REGISTRATION,
    cn.LISTING_DATE, cn.RECIPIENT_COUNTRY, cn.R_BLOODGROUP,
    cn.D_BLOODGROUP, cn.BG_PRIORITY, cn.R_MATCH_AGE, cn.R_WEIGHT,
    cn.MELD_INT_MATCH, cn.MELD_NAT_MATCH,
    cn.ALLOC_SCORE_INT, cn.ALLOC_SCORE_NAT,
    cn.R_PED, cn.R_LOWWEIGHT, cn.MATCH_CRITERIUM, cn.GEOGRAPHY_MATCH,
    cn.REC_CENTER_OFFER_GROUP, cn.DON_CENTER_OFFER_GROUP,
    cn.DONOR_DEATH_CAUSE_GROUP, cn.D_TUMOR_HISTORY,
    cn.LISTED_KIDNEY, cn.PATIENT_SEX, cn.TYPE_E, cn.PATIENT_RANK,
    cn.RANK, cn.SE, cn.NSE, cn.PED,
    cn.MELD_LAB, cn.ACCEPTED, cn.ACCEPTANCE_REASON,
    cn.PROB_ACCEPT_C, cn.PROB_ACCEPT_P, cn.DRAWN_PROB,
    cn.DRAWN_PROB_C, cn.TRAVEL_TIME, cn.TRAVEL_MODE,
    cn.PROFILE_COMPATIBLE, cn.D_HEIGHT, cn.R_HEIGHT
) + tuple(cg.MTCH_COLS) + cg.PED_CENTER_OFFERS + cg.ADULT_CENTER_OFFERS

MATCH_INFO_COLS_SET = set(MATCH_INFO_COLS)

# Recoding of Austrian centers for obligations.
AUSTRIAN_CENTER_GROUPS = {
    'AWDTP': 'AWGTP'
}

# Centers which can split organs (>1% of livers splitted & at least 1)
CENTERS_WHICH_SPLIT = [
    'GMLTP', 'AIBTP', 'GGOTP', 'GBOTP', 'GBCTP', 'GHBTP',
    'BLATP', 'GESTP', 'BGETP', 'GKITP', 'NGRTP', 'GTUTP',
    'CZPTP', 'GHOTP', 'GRBTP', 'GHGTP'
]

# Cut-off for HU eligible retransplantation
CUTOFF_RETX_HU_ELIGIBLE = 15

# Add type statuses
STATUS_TYPES = [
    cn.DM, cn.LAB, cn.NSE, cn.PED, cn.SE,
    cn.URG, cn.FU, cn.ACO, cn.PRF, cn.DIAG
    ]
STATUSES_UPDATE_BIOMARKERS = (cn.LAB, cn.FU)
STATUSES_EXCEPTION = (cn.NSE, cn.SE)

# Event types
EVENT_TYPES = [cn.PAT, cn.DON]

# Add removed SEs
PSC_BONUS_SE = '142'
REPLACING_BONUS_SES = [PSC_BONUS_SE]
SE_OFFSET = 1000

# Terminal status updates
TERMINAL_STATUSES = [cn.R, cn.D]
EXITING_STATUSES = [cn.R, cn.D, cn.FU]

# Type offer whole liver
TYPE_OFFER_WLIV = 4
TYPE_OFFER_ERL = 32
TYPE_OFFER_LLS = 31
TYPE_OFFER_RL = 33
TYPE_OFFER_LL = 34

# Simple rules which splits are being done
WEIGHT_ABOVE_FULL_SPLIT = 50
AGE_BELOW_CLASSIC_SPLIT = 12

# Which organs remain after split
ORGAN_AVAILABILITY_SPLIT = {
    31: 32,
    32: 31,
    33: 34,
    34: 33
}

# Tiers in which profiles are ignored
IGNORE_PROFILE_TIERS = [cn.TIER_E]

# HU/ACO tiers (after reversing the alphabetical order of tiers)
TIERS_HUACO = [cn.TIER_D, cn.TIER_E]
TIERS_HUACO_SPLIT = [cn.TIER_B, cn.TIER_C, cn.TIER_E, cn.TIER_F]

# Implemented acceptance module policies
PATIENT_ACCEPTANCE_POLICIES = ['Always', 'LR']
CENTER_ACCEPTANCE_POLICIES = ['Always', 'LR']

# Date MELD was introduced
INTRODUCTION_MELD = datetime(year=2006, month=12, day=26)

# ET-DRI components
ET_DRI_DICT = {
    cn.D_AGE: {
        0: (lambda x: x < 40),
        0.154*0.960: (lambda x: x >= 40 & x < 50),
        0.274*0.960: (lambda x: x >= 50 & x < 60),
        0.424*0.960: (lambda x: x >= 60 & x < 70),
        0.501*0.960: (lambda x: x >= 70)
    },
    cn.DONOR_DEATH_CAUSE_GROUP: {
        0.079*0.960: (lambda x: x == 'Anoxia'),
        0.145*0.960: (lambda x: x == 'CVA'),
        0.184*0.960: (lambda x: x == 'Other'),
        0: (lambda x: x == 'Head trauma' or x == 'CNS tumor')
    },
    cn.D_DCD: {
        0.411*0.960: (lambda x: x),
        0: (lambda x: True)
    },
    cn.TYPE_OFFER_DETAILED: {
        0: (lambda x: x == TYPE_OFFER_WLIV),
        0.422*0.960: (lambda x: True)
    },
    cn.MATCH_CRITERIUM: {
        0.105*0.960: (lambda x: x == cn.REG),
        0.105*0.960: (lambda x: x == cn.NAT),
        0.244*0.960: (lambda x: x == cn.INT),
        0: (lambda x: True)
    }
}


# Required donor and patient information for match records
MTCH_RCRD_DONOR_COLS = (
    cn.ID_DONOR, cn.D_COUNTRY, cn.D_PROC_CENTER,
    cn.D_HOSPITAL, cn.D_BLOODGROUP,
    cn.D_WEIGHT, cn.D_DCD
)

MTCH_RCRD_PAT_COLS = (
    cn.ID_RECIPIENT, cn.R_BLOODGROUP, cn.RECIPIENT_COUNTRY,
    cn.RECIPIENT_CENTER, cn.REC_CENTER_OFFER_GROUP, cn.R_WEIGHT,
    cn.R_ACO, cn.PATIENT_IS_HU
)
