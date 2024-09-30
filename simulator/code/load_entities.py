#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:33:44 2022

Scripts to read in input files.

@author: H.C. de Ferrante
"""

import sys
from typing import List, Dict, Dict, Tuple, Any, Iterator, Hashable, Optional
from datetime import timedelta
from itertools import groupby
import pandas as pd
import numpy as np
import warnings

sys.path.append('./')
if True:  # noqa 402
    from simulator.code.utils import DotDict, round_to_decimals, round_to_int
    import simulator.magic_values.column_names as cn
    import simulator.magic_values.column_names as cn
    import simulator.code.read_input_files as rdr
    from simulator.code.entities import Donor, Patient, Profile, \
        ExceptionSystem, Obligation
    from simulator.code.StatusUpdate import StatusUpdate, ProfileUpdate
    from simulator.code.PatientStatusQueue import PatientStatusQueue


def _read_patients_rich(
        sim_set: DotDict,
        start_date_col: Optional[str] = None,
        end_date_col: Optional[str] = None,
        **kwargs
) -> pd.DataFrame:
    """
        Read patients in with whether it is a retransplantation
        between the start and end date times.
    """

    # Read in patient data
    d_patients: pd.DataFrame = rdr.read_patients(
        sim_set.PATH_RECIPIENTS, **kwargs
    )

    # Read only patient data in before end of simulation.
    end_date = (
        sim_set.SIM_END_DATE if end_date_col is None
        else sim_set.__dict__[end_date_col]
    )
    d_patients = d_patients.loc[
        d_patients[cn.LISTING_DATE] <= end_date,
        :
    ]

    # Add whether patient is re-transplanted.
    d_patients[cn.RETRANSPLANT] = d_patients[cn.NTH_TRANSPLANT] > 0
    idx = d_patients.loc[:, cn.RETRANSPLANT]
    d_patients.loc[idx, cn.RETRANSPLANT_DURING_SIM] = (
        d_patients.loc[idx, cn.PREV_TX_DATE] >= sim_set.SIM_START_DATE
    )
    d_patients.loc[~idx, cn.RETRANSPLANT_DURING_SIM] = False

    # Add column for type re-transplantation
    d_patients[cn.TYPE_RETX] = np.select(
        condlist=[
            d_patients[cn.RETRANSPLANT_DURING_SIM].astype(bool).to_numpy(),
            d_patients[cn.RETRANSPLANT].to_numpy()
        ],
        choicelist=[
            cn.RETRANSPLANT_DURING_SIM,
            cn.RETRANSPLANT
            ],
        default=cn.NO_RETRANSPLANT
    )

    # Remove patients deregistered before simulation start
    start_date = (
        sim_set.SIM_START_DATE if start_date_col is None
        else sim_set.__dict__[start_date_col]
    )
    d_patients = d_patients.loc[
        (
            d_patients.apply(
                lambda x:
                x[cn.LISTING_DATE] + timedelta(days=x[cn.TIME_TO_DEREG]),
                axis=1
            ) >= start_date
        ),
        :
    ]

    return d_patients


def _rcrd_to_patient(sim_set: DotDict, rcrd: dict[Hashable, Any]) -> Patient:
    """Convert a record to a Patient object"""
    return Patient(
        id_recipient=rcrd[cn.ID_RECIPIENT],
        recipient_country=rcrd[cn.RECIPIENT_COUNTRY],
        recipient_center=rcrd[cn.RECIPIENT_CENTER],
        recipient_region=rcrd[cn.RECIPIENT_REGION],
        bloodgroup=rcrd[cn.R_BLOODGROUP],
        listing_date=rcrd[cn.TIME_REGISTRATION],
        urgency_code='NT',
        weight=rcrd[cn.R_WEIGHT],
        height=rcrd[cn.R_HEIGHT],
        r_dob=rcrd[cn.R_DOB],
        lab_meld=None,
        r_aco=False,
        sim_set=sim_set,
        listed_kidney=rcrd[cn.LISTED_KIDNEY],
        sex=rcrd[cn.PATIENT_SEX],
        type_retx=rcrd[cn.TYPE_RETX],
        time_since_prev_txp=rcrd[cn.TIME_SINCE_PREV_TXP],
        id_reg=rcrd[cn.ID_REGISTRATION],
        time_to_dereg=rcrd[cn.TIME_TO_DEREG],
        seed=sim_set.SEED
    )


def load_patients(sim_set: DotDict, **kwargs) -> dict[int, Patient]:
    """Load list of patients"""

    # Load patients. The start and end date
    # are simulation start and end by default.
    d_patients = _read_patients_rich(
        sim_set=sim_set
    )

    # If we simulate re-transplantations, do not load patient registrations
    # that are future re-transplantations for patients not yet transplanted.
    patient_dict = {
        rcrd[cn.ID_REGISTRATION]: _rcrd_to_patient(sim_set, rcrd)
        for rcrd in d_patients.to_dict(orient='records')
        if (
            not sim_set.SIM_RETX or not (
                sim_set.SIM_RETX and
                rcrd[cn.TYPE_RETX] == cn.RETRANSPLANT_DURING_SIM and
                rcrd[cn.PREV_TXP_LIVING] == 0
            )
        )
    }

    return patient_dict


def load_retransplantations(sim_set: DotDict, **kwargs) -> dict[int, Patient]:
    """Load list of patients"""

    # Load patients. The start and end date
    # are simulation start and end by default.
    d_patients = _read_patients_rich(
        sim_set=sim_set
    )

    # Do not include patients listed outside the transplantation registration
    # window; gives bias.
    d_patients = d_patients.loc[
        (
            d_patients.loc[:, cn.LISTING_DATE] >=
            sim_set.__dict__['LOAD_RETXS_FROM']
        ) & (
            d_patients.loc[:, cn.LISTING_DATE] <=
            sim_set.__dict__['LOAD_RETXS_TO']
        ),
        :
    ]

    # Construct a list of patients from patient data
    patient_dict = {
        rcrd[cn.ID_REGISTRATION]: _rcrd_to_patient(sim_set, rcrd)
        for rcrd in d_patients.to_dict(orient='records')
        if rcrd[cn.TYPE_RETX] != cn.NO_RETRANSPLANT
    }

    return patient_dict


def load_obligations(
        sim_set: DotDict,
        empty_obl_dict: Dict[str, Dict[Tuple[str, str], List[Any]]]
) -> Dict[str, Dict[Tuple[str, str], List[Any]]]:
    """Load list of donors"""

    # Read in obligations. Select only obligations inserted before the
    # simulation start date which did not end yet.
    d_obl = rdr.read_obligations(sim_set.PATH_OBLIGATIONS)
    d_obl = d_obl.loc[
        (d_obl.loc[:, cn.OBL_INSERT_DATE] <= sim_set.SIM_START_DATE) &
        (d_obl.loc[:, cn.OBL_END_DATE] >= sim_set.SIM_START_DATE),
        :
    ]

    obligations = empty_obl_dict
    for rcrd in d_obl.to_dict(orient='records'):
        obligations[
            rcrd[cn.OBL_BG]
            ][
                (rcrd[cn.OBL_DEBTOR], rcrd[cn.OBL_CREDITOR])
            ].append(
                Obligation(
                    bloodgroup=rcrd[cn.OBL_BG],
                    creditor=rcrd[cn.OBL_CREDITOR],
                    debtor=rcrd[cn.OBL_DEBTOR],
                    obl_insert_date=rcrd[cn.OBL_INSERT_DATE],
                    start_date=rcrd[cn.OBL_START_DATE]
                )
            )

    return obligations


def load_donors(
        sim_set: DotDict,
        load_splits: bool = False,
        **kwargs
) -> dict[int, Donor]:
    """Load list of donors"""

    # Read in donor data data
    d_don = rdr.read_donors(sim_set.PATH_DONORS, **kwargs)
    d_don = d_don.loc[
        (d_don[cn.D_DATE] >= sim_set.SIM_START_DATE) &
        (d_don[cn.D_DATE] <= sim_set.SIM_END_DATE),
        :
    ]

    # Drop split offers (do not use real splits).
    if not load_splits:
        d_don = d_don.loc[
            d_don[cn.KTH_OFFER] == 1, :
        ]

    # Construct a list of patients from patient data
    donor_dict = {
            rcrd[cn.ID_DONOR]: Donor(
                id_donor=rcrd[cn.ID_DONOR],
                bloodgroup=rcrd[cn.D_BLOODGROUP],
                donor_country=rcrd[cn.D_COUNTRY],
                donor_center=rcrd[cn.D_ALLOC_CENTER],
                donor_proc_center=rcrd[cn.D_PROC_CENTER],
                donor_hospital=rcrd[cn.D_HOSPITAL],
                donor_region=rcrd[cn.D_REGION],
                reporting_date=rcrd[cn.D_DATE],
                weight=rcrd[cn.D_WEIGHT],
                donor_dcd=rcrd[cn.D_DCD],
                first_offer_type=rcrd[cn.TYPE_OFFER_DETAILED],
                age=rcrd[cn.D_AGE],
                hbsag=rcrd[cn.D_HBSAG],
                hcvab=rcrd[cn.D_HCVAB],
                hbcab=rcrd[cn.D_HBCAB],
                sepsis=rcrd[cn.D_SEPSIS],
                meningitis=rcrd[cn.D_MENINGITIS],
                malignancy=rcrd[cn.D_MALIGNANCY],
                drug_abuse=rcrd[cn.D_DRUG_ABUSE],
                marginal=rcrd[cn.D_MARGINAL],
                euthanasia=rcrd[cn.D_EUTHANASIA],
                rescue=(
                    rcrd[cn.D_RESCUE] if not sim_set.SIMULATE_RESCUE
                    else False
                ),
                tumor_history=rcrd[cn.D_TUMOR_HISTORY],
                donor_marginal_free_text=rcrd[cn.D_MARGINAL_FREE_TEXT],
                donor_death_cause_group=rcrd[cn.DONOR_DEATH_CAUSE_GROUP],
                diabetes=rcrd[cn.D_DIABETES]
            ) for rcrd in d_don.to_dict(orient='records')
        }

    return donor_dict


def preload_aco_statuses(
        patients: Dict[int, Patient],
        sim_set: DotDict, **kwargs
) -> None:
    """Add aco statuses to patients"""

    # Read in donor data data
    d_acos = rdr.read_acos(sim_set.PATH_ACOS, **kwargs)

    # For patient in patients
    for rcrd in d_acos.to_dict(orient='records'):
        if (
            rcrd[cn.ID_REGISTRATION] in patients and
            patients[rcrd[cn.ID_REGISTRATION]].future_statuses is not None
        ):
            patients[rcrd[cn.ID_REGISTRATION]]. \
                future_statuses.add(
                    StatusUpdate(
                        type_status=cn.ACO,
                        arrival_time=rcrd[cn.TSTART],
                        biomarkers={},
                        status_action=rcrd[cn.ACO_STATUS],
                        status_detail='',
                        status_value='',
                        sim_start_time=(
                            patients[
                                rcrd[cn.ID_REGISTRATION]
                            ].__dict__[cn.LISTING_DATE] -
                            sim_set.SIM_START_DATE
                        ) / timedelta(days=1)
                    )
                )


def preload_disease_groups(
        patients: Dict[int, Patient],
        sim_set: DotDict, **kwargs
) -> None:
    """Add aco statuses to patients"""

    # Read in donor data data
    d_diag = rdr.read_diags(sim_set.PATH_DIAGS, **kwargs)

    # For patient in patients
    for rcrd in d_diag.to_dict(orient='records'):
        if (
            rcrd[cn.ID_REGISTRATION] in patients and
            patients[rcrd[cn.ID_REGISTRATION]].future_statuses is not None
        ):
            patients[rcrd[cn.ID_REGISTRATION]]. \
                future_statuses.add(
                    StatusUpdate(
                        type_status=cn.DIAG,
                        arrival_time=rcrd[cn.TSTART],
                        biomarkers={},
                        status_action=1,
                        status_detail='',
                        status_value=rcrd[cn.DISEASE_GROUP],
                        sim_start_time=(
                            patients[
                                rcrd[cn.ID_REGISTRATION]
                            ].__dict__[cn.LISTING_DATE] -
                            sim_set.SIM_START_DATE
                        ) / timedelta(days=1)
                    )
                )


def preload_profiles(
        patients: Dict[int, Patient],
        sim_set: DotDict, **kwargs
) -> None:
    """Add profile information for patients"""

    # Read in donor data data
    d_profiles = rdr.read_profiles(sim_set.PATH_PROFILES, **kwargs)

    # For patient in patients
    for rcrd in d_profiles.to_dict(orient='records'):
        if (
            rcrd[cn.ID_REGISTRATION] in patients.keys() and
            patients[rcrd[cn.ID_REGISTRATION]].future_statuses is not None
        ):
            patients[rcrd[cn.ID_REGISTRATION]]. \
                future_statuses.add(
                    ProfileUpdate(
                        type_status=cn.PRF,
                        arrival_time=rcrd[cn.TSTART],
                        profile=Profile(
                            min_age=rcrd[cn.PROFILE_MIN_DONOR_AGE],
                            max_age=rcrd[cn.PROFILE_MAX_DONOR_AGE],
                            min_weight=rcrd[cn.PROFILE_MIN_WEIGHT],
                            max_weight=rcrd[cn.PROFILE_MAX_WEIGHT],
                            hbsag=rcrd[cn.PROFILE_HBSAG],
                            hcvab=rcrd[cn.PROFILE_HCVAB],
                            hbcab=rcrd[cn.PROFILE_HBCAB],
                            sepsis=rcrd[cn.PROFILE_SEPSIS],
                            meningitis=rcrd[cn.PROFILE_MENINGITIS],
                            malignancy=rcrd[cn.PROFILE_MALIGNANCY],
                            drug_abuse=rcrd[cn.PROFILE_DRUG_ABUSE],
                            marginal=rcrd[cn.PROFILE_MARGINAL],
                            rescue=rcrd[cn.PROFILE_RESCUE],
                            lls=rcrd[cn.PROFILE_LLS],
                            erl=rcrd[cn.PROFILE_ERL],
                            rl=rcrd[cn.PROFILE_RL],
                            ll=rcrd[cn.PROFILE_LL],
                            euthanasia=rcrd[cn.PROFILE_EUTHANASIA],
                            dcd=rcrd[cn.PROFILE_DCD]
                        ),
                        sim_start_time=(
                            patients[
                                rcrd[cn.ID_REGISTRATION]
                            ].__dict__[cn.LISTING_DATE] -
                            sim_set.SIM_START_DATE
                        ) / timedelta(days=1)
                    )
                )


def to_dict_fast(df):
    cols = list(df)
    col_arr_map = {col: df[col].astype(object).to_numpy() for col in cols}
    records = []
    for i in range(len(df)):
        record = {col: col_arr_map[col][i] for col in cols}
        records.append(record)
    return records


def preload_status_updates(
        patients: Dict[int, Patient],
        sim_set: DotDict,
        se_sys: ExceptionSystem,
        end_date_col: Optional[str] = None,
        **kwargs
) -> None:
    """ Read in status updates for selected patients,
        and preload them in the patients.
    """

    # Read in status updates for patients
    d_status_updates = rdr.read_status_updates(
        sim_set.PATH_STATUS_UPDATES,
        end_date_col=end_date_col,
        sim_set=sim_set,
        **kwargs
    )

    # Delay standard exceptions, if we simulate with a delay.
    for type_e in se_sys.exceptions.keys():
        se_recode_dict = {
            se_num: se.nse_delay
            for se_num, se in se_sys.exceptions[type_e].items()
            if se.nse_delay != 0
        }
        for se_key, se_delay in se_recode_dict.items():
            exc = se_sys.exceptions[type_e][se_key]
            print(f'Simulating {exc} with a {se_delay}-day delay')

        # TODO: Maybe add in delaying only for candidates registered after
        # the simulation start date (to see transient effects).
        if se_recode_dict:
            sel_stats = (
                (d_status_updates.loc[:, cn.TYPE_UPDATE] == type_e) &
                d_status_updates.loc[:, cn.STATUS_DETAIL].isin(se_recode_dict)
            )
            d_status_updates.loc[sel_stats, cn.TSTART] = (
                    d_status_updates.loc[sel_stats, cn.TSTART] +
                    d_status_updates.loc[sel_stats, cn.STATUS_DETAIL].replace(
                        se_recode_dict
                    )
            )

    # Check whether imputation procedure was run correctly
    # (i.e. almost all patients have a terminal status)
    d_last_statuses = d_status_updates.groupby(
        cn.ID_REGISTRATION
    ).last().loc[:, cn.URGENCY_CODE]
    fb_last_statuses = (
        d_last_statuses.loc[~ d_last_statuses.isin([cn.R, cn.D])].index
    )
    if len(fb_last_statuses) > 0:
        d_status_updates = d_status_updates.loc[
                    ~ d_status_updates.loc[:, cn.ID_REGISTRATION].isin(
                        fb_last_statuses
                    ),
                    :
                ]
    if len(fb_last_statuses) > 5 and len(fb_last_statuses) < 20:
        warnings.warn(
            f'Removed {len(fb_last_statuses)} registrations '
            f'without R/D as last status update'
            )
    elif len(fb_last_statuses) > 20:
        raise ValueError(
            f'There are {len(fb_last_statuses)} registrations without '
            f'terminal R/D status. All registrations must end with R/D.'
        )

    for id_reg, status_upd in groupby(
        to_dict_fast(
            d_status_updates.loc[
                d_status_updates[cn.ID_REGISTRATION].isin(
                    patients.keys()
                )
            ]
        ),
            lambda x: x[cn.ID_REGISTRATION]
    ):
        if int(id_reg) in patients:
            patients[int(id_reg)].preload_status_updates(
                fut_stat=dstat_to_queue(
                    rcrds=status_upd,
                    sim_set=sim_set
                )
            )


def dstat_to_queue(
        rcrds: Iterator[Dict[Hashable, Any]],
        sim_set: DotDict,
        ) -> PatientStatusQueue:
    """Read in statuses and convert to a heap queue instance"""

    status_queue = PatientStatusQueue(
        initial_events=[
            StatusUpdate(
                    type_status=rcrd[cn.TYPE_UPDATE],
                    arrival_time=rcrd[cn.TSTART],
                    biomarkers={
                        cn.CREA: rcrd[cn.CREA],
                        cn.BILI: rcrd[cn.BILI],
                        cn.INR: rcrd[cn.INR],
                        cn.SODIUM: rcrd[cn.SODIUM],
                        cn.ALBU: rcrd[cn.ALBU],
                        cn.DIAL_BIWEEKLY: rcrd[cn.DIAL_BIWEEKLY]
                    },
                    status_action=rcrd[cn.STATUS_ACTION],
                    status_detail=rcrd[cn.STATUS_DETAIL],
                    status_value=rcrd[cn.STATUS_VALUE],
                    sim_start_time=(
                        rcrd[cn.LISTING_DATE] -
                        sim_set.SIM_START_DATE
                    ) / timedelta(days=1)
                ) for rcrd in rcrds
        ]
    )

    return status_queue
