#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

Magic values for the simulator

@author: H.C. de Ferrante
"""

import simulator.magic_values.column_names as cn
import numpy as np
import pandas as pd

DEFAULT_DATE_FORMAT = '%Y-%m-%d'
DEFAULT_DMY_HMS_FORMAT = '%d-%m-%Y %H:%M:%S'


DTYPE_OFFERLIST = {
    cn.RECIPIENT_OFFERED: 'Int64',
    cn.R_DOB: 'object',
    cn.ID_MTR: 'Int64',
    cn.MATCH_DATE: 'object',
    cn.ID_TXP: 'Int64',
    cn.D_DCD: 'Int64',
    cn.D_WEIGHT: 'float64',
    cn.D_BLOODGROUP: 'object',
    cn.TYPE_OFFER: 'Int64',
    cn.D_COUNTRY: 'object',
    cn.D_REGION: 'object',
    cn.D_ALLOC_CENTER: 'object',
    cn.MTCH_TIER: 'object',
    cn.MTCH_LAYER: 'Int64',
    cn.MTCH_OBL: 'Int64',
    cn.MTCH_LAYER_MELD: 'Int64',
    cn.MTCH_LAYER_WT: 'Int64',
    cn.R_MATCH_AGE: 'float64',
    cn.MATCH_CRITERIUM: 'object',
    cn.URGENCY_CODE: 'object',
    cn.R_ACO: 'Int64',
    cn.R_BLOODGROUP: 'object',
    cn.REC_REQ: 'object',
    cn.REQ_INT: 'Int64',
    cn.RECIPIENT_COUNTRY: 'object',
    cn.RECIPIENT_CENTER: 'object',
    cn.RECIPIENT_REGION: 'object',
    cn.R_WEIGHT: 'float64',
    cn.R_HEIGHT: 'float64',
    cn.RNK_OTHER: 'Int64',
    cn.RNK_ACCEPTED: 'Int64',
    cn.ACCEPTED: bool,
    cn.ACTION_CODE: 'object',
    cn.ACTION_CODE_DESC: 'object',
    cn.FINAL_REC_URG_AT_TRANSPLANT: 'object',
    cn.MATCH_COMMENT: 'object',
    cn.TYPE_TRANSPLANTED: 'str'
}


DTYPE_PATIENTLIST = {
    cn.ID_RECIPIENT: int,
    cn.ID_REGISTRATION: int,
    cn.TIME_REGISTRATION: 'object',
    cn.TIME_TO_DEREG: 'Int64',
    cn.TIME_SINCE_PREV_TXP: 'Int64',
    cn.PREV_TXP_LIVING: int,
    cn.KTH_REG: int,
    cn.NTH_TRANSPLANT: int,
    cn.RECIPIENT_COUNTRY: str,
    cn.RECIPIENT_CENTER: str,
    cn.RECIPIENT_REGION: str,
    cn.R_BLOODGROUP: str,
    cn.R_WEIGHT: float,
    cn.R_HEIGHT: float,
    cn.R_DOB: 'object',
    cn.LISTED_KIDNEY: int,
    cn.PATIENT_SEX: str
}

DTYPE_OBLIGATIONS = {
    cn.OBL_INSERT_DATE: object,
    cn.OBL_START_DATE: object,
    cn.OBL_END_DATE: object,
    cn.OBL_BG: str,
    cn.OBL_CREDITOR: str,
    cn.OBL_DEBTOR: str
}

DTYPE_DONORLIST = {
    cn.D_DATE: object,
    cn.ID_DONOR: int,
    cn.KTH_OFFER: int,
    cn.TYPE_OFFER: str,
    cn.TYPE_OFFER_DETAILED: int,
    cn.D_COUNTRY: str,
    cn.D_PROC_CENTER: str,
    cn.D_ALLOC_CENTER: str,
    cn.D_HOSPITAL: str,
    cn.D_REGION: str,
    cn.D_AGE: int,
    cn.D_WEIGHT: float,
    cn.D_BLOODGROUP: str,
    cn.D_HBSAG: 'Int64',
    cn.D_HCVAB: 'Int64',
    cn.D_HBCAB: 'Int64',
    cn.D_SEPSIS: 'Int64',
    cn.D_MENINGITIS: 'Int64',
    cn.D_MALIGNANCY: 'Int64',
    cn.D_DRUG_ABUSE: 'Int64',
    cn.D_MARGINAL: 'Int64',
    cn.D_DCD: 'Int64',
    cn.D_EUTHANASIA: 'Int64',
    cn.DONOR_DEATH_CAUSE_GROUP: str,
    cn.D_TUMOR_HISTORY: 'Int64',
    cn.D_MARGINAL_FREE_TEXT: 'Int64',
    cn.D_SEX: str,
    cn.D_SMOKING: 'Int64',
    cn.D_ALCOHOL_ABUSE: 'Int64',
    cn.D_DIABETES: 'Int64',
    cn.D_RESCUE: 'Int64',
    cn.D_DIABETES: 'Int64'
}


DTYPE_DONOR_FILL_NAS = {
    cn.D_HBSAG: 0,
    cn.D_HCVAB: 0,
    cn.D_HBCAB: 0,
    cn.D_SEPSIS: 0,
    cn.D_MENINGITIS: 0,
    cn.D_MALIGNANCY: 0,
    cn.D_DRUG_ABUSE: 0,
    cn.D_MARGINAL: 0,
    cn.D_DCD: 0,
    cn.D_EUTHANASIA: 0,
    cn.D_RESCUE: 0,
    cn.D_DIABETES: 0
}


DTYPE_ACOS = {
    cn.ID_REGISTRATION: int,
    cn.TSTART: float,
    cn.ACO_STATUS: int
}


DTYPE_DIAGS = {
    cn.ID_REGISTRATION: int,
    cn.TSTART: float,
    cn.DISEASE_GROUP: str
}


DTYPE_PROFILES = {
    cn.ID_REGISTRATION: int,
    cn.TSTART: float,
    cn.PRF_TYPE: str,
    cn.PROFILE_MIN_DONOR_AGE: int,
    cn.PROFILE_MAX_DONOR_AGE: int,
    cn.PROFILE_MIN_WEIGHT: int,
    cn.PROFILE_MAX_WEIGHT: int,
    cn.PROFILE_DCD: bool,
    cn.PROFILE_HBSAG: bool,
    cn.PROFILE_HCVAB: bool,
    cn.PROFILE_HBCAB: bool,
    cn.PROFILE_SEPSIS: bool,
    cn.PROFILE_MENINGITIS: bool,
    cn.PROFILE_MALIGNANCY: bool,
    cn.PROFILE_DRUG_ABUSE: bool,
    cn.PROFILE_MARGINAL: bool,
    cn.PROFILE_RESCUE: bool,
    cn.PROFILE_LLS: bool,
    cn.PROFILE_ERL: bool,
    cn.PROFILE_RL: bool,
    cn.PROFILE_LL: bool,
    cn.PROFILE_EUTHANASIA: bool
}


DTYPE_STATUSUPDATES = {
    cn.ID_REGISTRATION: int,
    cn.TSTART: float,
    cn.TYPE_UPDATE: 'category',
    cn.URGENCY_CODE: 'category',
    cn.URGENCY_REASON: str,
    cn.DIAL_BIWEEKLY: 'Int64',
    cn.CREA: float,
    cn.BILI: float,
    cn.INR: float,
    cn.SODIUM: float,
    cn.ALBU: float,
    cn.STATUS_ACTION: 'Int64',
    cn.REMOVAL_REASON: 'category',
    cn.STATUS_VALUE: str,
    cn.STATUS_DETAIL: str,
    cn.LISTING_DATE: object
}


DTYPE_SE_RULES = {
    cn.ID_SE: str,
    cn.TYPE_E: str,
    cn.DISEASE_SE: str,
    cn.SE_COUNTRY: str,
    cn.SE_UPGRADE: float,
    cn.SE_UPGRADE_INTERVAL: int,
    cn.LAB_MELD_BONUS: bool,
    cn.LAB_MELD_BONUS_AMOUNT: int,
    cn.MAX_EQUIVALENT: int,
    cn.INITIAL_EQUIVALENT: int,
    cn.ALLOWED_EQUIVALENTS_BY: float,
    cn.ALLOWED_EQS_OFFSET: 'Int64',
    cn.REPLACEMENT_SE: str,
    cn.MAX_AGE_UPGRADE: 'Int64'
}


# Capitalize if read from "raw_data"
DTYPE_SE_RULES = {
    k.upper(): v for
    k, v in DTYPE_SE_RULES.items()
    }

DTYPE_DONORLIST = {
    k: v for
    k, v in DTYPE_DONORLIST.items()
    }

DTYPE_OFFERLIST = {
    k.upper(): v for
    k, v in DTYPE_OFFERLIST.items()
    }

DTYPE_ACOS = {
    k.upper(): v for
    k, v in DTYPE_ACOS.items()
    }

DTYPE_PROFILES = {
    k.upper(): v for
    k, v in DTYPE_PROFILES.items()
    }
