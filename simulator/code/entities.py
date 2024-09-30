#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

@author: H.C. de Ferrante
"""

from typing import List, Dict, Tuple, Optional, Callable, Generator, Union, Any
from datetime import timedelta, datetime
from math import floor, isnan, ceil
from itertools import count, product
from copy import deepcopy
from warnings import warn
from collections import defaultdict

import pandas as pd
import numpy as np
import simulator.magic_values.magic_values_rules as mgr

from simulator.code.utils import DotDict, round_to_decimals, round_to_int
from simulator.code.exception_system import ExceptionScore, ExceptionSystem
import simulator.magic_values.elass_settings as es
from simulator.magic_values.rules import (
    CENTER_OFFER_GROUPS
)
from simulator.code.StatusUpdate import StatusUpdate, ProfileUpdate
from simulator.code.read_input_files import read_se_rules
from simulator.code.PatientStatusQueue import PatientStatusQueue
from simulator.code.SimResults import SimResults
import simulator.magic_values.column_names as cn
from simulator.code.ScoringFunction import AllocationScore, clamp


def isDigit(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


class Donor:
    """Class which implements a donor

    ...

    Attributes   #noqa
    ----------
    id_donor: int
        integer for the donor ID
    donor_hospital: int
        Donor hospital (tied to travel time)
    donor_country: str
        name of country of donation
    don_region: str
        name of donor region
    donor_center: str
        name of donating center (5-letter code).
    donor_age: int
        Donor age
    weight: float
        weight of the donor
    bloodgroup: str
        recipient blood group
    reporting_date: datetime
        Time of entry on the liver waitlist


    Methods
    -------
    arrival_at():
        time at which donor arrives, in time from simulation date.
    """

    def __init__(
            self, id_donor: int, bloodgroup: str, donor_country: str,
            donor_center: str, donor_region: str,
            reporting_date: datetime, weight: float,
            donor_dcd: int, first_offer_type: int,
            height: Optional[float] = None,
            age: Optional[int] = None, hbsag: bool = False,
            hcvab: bool = False, hbcab: bool = False,
            sepsis: bool = False, meningitis: bool = False,
            malignancy: bool = False, drug_abuse: bool = False,
            marginal: bool = False, euthanasia: bool = False,
            rescue: bool = False, dri: float = np.nan,
            donor_death_cause_group: Optional[str] = None,
            donor_marginal_free_text: Optional[int] = 0,
            tumor_history: Optional[int] = 0,
            donor_proc_center: Optional[str] = None,
            donor_hospital: Optional[int] = None,
            diabetes: Optional[int] = 0,
            ggt: Optional[float] = None
            ) -> None:

        self.id_donor = id_donor

        # Geographic information
        assert donor_country in es.ET_COUNTRIES, \
            f'listing country should be one of:\n\t' \
            f'{", ".join(es.ET_COUNTRIES)}'
        self.donor_country = donor_country
        self.donor_region = donor_region
        self.donor_center = donor_center
        self.__dict__[cn.D_PROC_CENTER] = donor_proc_center if \
            donor_proc_center else donor_center
        self.__dict__[cn.D_HOSPITAL] = str(donor_hospital) if \
            donor_hospital else '1'

        self.reporting_date = reporting_date

        # Donor blood group
        assert bloodgroup in es.ALLOWED_BLOODGROUPS, \
            f'blood group should be one of:\n\t' \
            f'{", ".join(es.ALLOWED_BLOODGROUPS)}'
        self.d_bloodgroup = str(bloodgroup)

        self.d_weight = float(weight)
        self.d_height = float(height) if height is not None else 170
        self.donor_bmi = self.d_weight / (self.d_height/100)**2
        if self.donor_bmi < 20:
            self.donor_bmi_cat = '<20'
        elif self.donor_bmi < 25:
            self.donor_bmi_cat = "20-25"
        elif self.donor_bmi < 30:
            self.donor_bmi_cat = "25-30"
        else:
            self.donor_bmi_cat = '30+'

        self.graft_dcd = bool(donor_dcd)

        self.first_offer_type = int(first_offer_type)

        # Profile info
        if age is not None:
            self.__dict__[cn.D_AGE] = age
        self.hbsag = hbsag if hbsag is not None and  \
            hbsag is not np.nan else False
        self.hcvab = hcvab if hcvab is not None and \
            hcvab is not np.nan else False
        self.hbcab = hbcab if hbcab is not None and \
            hbcab is not np.nan else False
        self.sepsis = sepsis if sepsis is not None and \
            sepsis is not np.nan else False
        self.meningitis = meningitis if meningitis is not None and \
            meningitis is not np.nan else False
        self.malignancy = malignancy if malignancy is not None and \
            malignancy is not np.nan else False
        self.marginal = marginal if marginal is not None and \
            marginal is not np.nan else False
        self.drug_abuse = drug_abuse if drug_abuse is not None and \
            drug_abuse is not np.nan else False
        self.euthanasia = euthanasia if euthanasia is not None and \
            euthanasia is not np.nan else False
        self.rescue = rescue if rescue is not None and \
            rescue is not np.nan else False
        self.__dict__[cn.D_ET_DRI] = dri if dri is not None and \
            dri is not np.nan else False
        self.donor_death_cause_group = donor_death_cause_group \
            if donor_death_cause_group is not None and \
            donor_death_cause_group is not np.nan else False
        self.__dict__[cn.D_MARGINAL_FREE_TEXT] = donor_marginal_free_text \
            if donor_marginal_free_text is not np.nan else 0
        self.__dict__[cn.D_TUMOR_HISTORY] = tumor_history \
            if tumor_history is not np.nan else 0
        self.__dict__[cn.D_DIABETES] = diabetes \
            if diabetes is not np.nan else 0
        self.__dict__[cn.D_GGT] = ggt

        self.__dict__[cn.D_PED] = None

        self._needed_match_info = None
        self._offer_inherit_cols = None

    def arrival_at(self, sim_date: datetime):
        """Retrieve calendar time arrival time."""
        return (
                self.reporting_date - sim_date
        ) / timedelta(days=1)

    def check_pediatric(self, ped_fun: Callable):
        """Check whether donor is pediatric"""
        if self.__dict__[cn.D_PED] is None:
            self.__dict__[cn.D_PED] = ped_fun(self)
        return self.__dict__[cn.D_PED]

    def __str__(self):
        return(
            f'Donor {self.id_donor}, reported on '
            f'{self.reporting_date}, dcd: {self.graft_dcd}'
            )

    def __repr__(self):
        return(
            f'Donor {self.id_donor}, reported on '
            f'{self.reporting_date}, dcd: {self.graft_dcd}'
            )

    @property
    def needed_match_info(self):
        if not self._needed_match_info:
            self._needed_match_info = {
                k: self.__dict__[k]
                for k
                in es.MTCH_RCRD_DONOR_COLS
            }
        return self._needed_match_info

    @property
    def offer_inherit_cols(self):
        if not self._offer_inherit_cols:
            self._offer_inherit_cols = {
                k: self.__dict__[k] for k in es.OFFER_INHERIT_COLS['donor']
            }
        return self._offer_inherit_cols


class Patient:
    """Class which implements a patient
    ...

    Attributes   #noqa
    ----------
    id_recipient: int
        integer for the recipient ID
    r_dob: datetime
        patient DOB
    age_at_listing: int
        age in days at listing
    recipient_country: str
        name of country of listing (not nationality!)
    recipient_center: str
        name of listing center (5-letter code).
    bloodgroup: str
        recipient blood group
    r_weight: float
        patient weight
    r_height: float
        patient height
    r_aco: bool
        whether patient has ACO status
    urgency_code: str
        current patient urgency code
    urgency_reason: Optional[str]
        urgency reason
    type_e: Optional[str]
        type exception
    exception: Optional[ExceptionScore]
        exception score
    ped_status: Optional[ExceptionScore]
        pediatric status
    e_since: float
        how long SE has been active
    listing_date: datetime
        Time of entry on the liver waitlist
    urgency_code: str
        Status of listing (HU/NT/T)
    biomarkers: dict[str, float]
        biomarker dictionary
    meld_scores: dict[str, object]
        meld score dictionary
    last_update_time: float
        time at which the last update was issued
    hu_since: float
        time since registration since when patient is HU
    aco_since: float
        time since registration since when patient is ACO
    future_statuses: Optional[PatientStatusQueue]
        status queue
    exit_status: Optional[str]
        exit status
    exit_date: Optional[datetime]
        exit time
    age_nat_meld: np.ndarray
        accrued waiting time for the national MELD score
    age_int_meld: np.ndarray
        accrued waiting time for the international MELD score
    sim_set: DotDict,
        simulation settings under which pat. is created
    profile: Optional[Profile]
        patient profile
    listed_kidney: int
        whether patient is simulatenously listed for kidney.
    patient_sex: str
        sex of the patient,
    time_since_prev_txp: float
        Time elapsed since previous TXP
    id_received_donor: int
        donor ID
    id_registration: int
        registration ID [optional]
    seed: int
        Optional seed for random number generator
    time_to_dereg: float
        time to deregistration (in days from registration)
    rng_acceptance: Generator
        Random number generator for acceptance probabilities

    Methods
    -------
    age_at_t(time: float)
        Return age at time t
    next_update_at()
        Return time at which next update is issued
    get_accrued_waiting_time(
            current_cal_time: float,
            type_meld: str):
        Return accrued waiting time in int or match MELD.
    get_time_in_status(type_status):
        Return time in HU or ACO status]
    get_acceptance_prob():
        Return an acceptance probability
    """

    def __init__(
            self, id_recipient: int, recipient_country: str,
            recipient_center: str, recipient_region: str, bloodgroup: str,
            listing_date: datetime, urgency_code: str, weight: float,
            height: float, r_dob: datetime,
            r_aco: bool, sim_set: DotDict,
            lab_meld: Optional[float] = None,
            se_meld: Optional[float] = 0,
            nse_meld: Optional[float] = 0,
            ped_meld: Optional[float] = 0,
            dm_meld: Optional[float] = None,
            listed_kidney: Optional[int] = 0,
            sex: Optional[str] = None,
            type_e: Optional[str] = None,
            type_retx: str = cn.NO_RETRANSPLANT,
            time_since_prev_txp: Optional[float] = None,
            id_reg: Optional[int] = None,
            time_to_dereg: Optional[float] = None,
            seed: Optional[int] = None) -> None:

        self.id_recipient = int(id_recipient)
        self.id_registration = id_reg if id_reg else None
        self.id_received_donor = None
        assert isinstance(listing_date, datetime), \
            f'listing date should be datetime, not {type(listing_date)}'
        self.__dict__[cn.LISTING_DATE] = listing_date
        self.age_days_at_listing = round_to_int(
            (listing_date - r_dob) / timedelta(days=1)
            )
        if seed is None:
            self.seed = 1
        elif id_reg is None:
            warn(
                f'No registration ID is set. '
                f'Setting seed to the registration ID.'
            )
            self.seed = self.seed
        else:
            self.seed = seed * id_reg
        self.rng_acceptance = np.random.default_rng(self.seed)

        # Load patient bloodgroup
        assert bloodgroup in es.ALLOWED_BLOODGROUPS, \
            f'blood group should be one of:\n\t' \
            f'{", ".join(es.ALLOWED_BLOODGROUPS)}'
        self.r_bloodgroup = str(bloodgroup)

        # Insert patient height & weight
        self.r_weight = float(weight)
        self.r_height = float(height)
        self.__dict__[cn.PATIENT_BMI] = (
            self.r_weight / (self.r_height/100)**2
            )

        # Check listing status
        assert urgency_code in es.ALLOWED_STATUSES, \
            f'listing status should be one of:\n\t' \
            f'{", ".join(es.ALLOWED_STATUSES)}'
        self.urgency_code = str(urgency_code)
        self.__dict__[cn.PATIENT_IS_HU] = str(urgency_code) == cn.HU
        self.__dict__[cn.ANY_HU] = self.__dict__[cn.PATIENT_IS_HU]
        self.urgency_reason = None
        self.r_aco = bool(r_aco)

        # Geographic information
        assert recipient_country in es.ET_COUNTRIES, \
            f'listing country should be one of:\n\t' \
            f'{", ".join(es.ET_COUNTRIES)}'

        self.__dict__[cn.RECIPIENT_CENTER] = str(recipient_center)
        self.__dict__[cn.RECIPIENT_REGION] = str(recipient_region)
        self.__dict__[cn.RECIPIENT_COUNTRY] = str(recipient_country)

        self.__dict__[cn.REC_CENTER_OFFER_GROUP] = (
            CENTER_OFFER_GROUPS.get(
                recipient_center,
                recipient_center
            )
        )

        # Set DOB.
        if not isinstance(r_dob, datetime):
            raise TypeError(
                f'r_dob must be datetime, not a {type(r_dob)}'
                )
        self.r_dob = r_dob

        # Initialize SE information
        self.exception = None
        if type_e is None:
            self.type_e = 'None'
        else:
            self.type_e = type_e
        self.__dict__[cn.E_SINCE] = None

        # Initialize PED status
        self.ped_status = None

        # Status updates
        self.future_statuses = None

        # MELD variables
        self.meld_scores = {
            cn.SE: se_meld,
            cn.NSE: nse_meld,
            sim_set.LAB_MELD: None,
            cn.SCORE_LABHISTORIC: lab_meld,
            cn.DM: dm_meld,
            cn.PED: ped_meld
        }

        # Placeholders
        self.biomarkers = {
            cn.BILI: np.nan,
            cn.CREA: np.nan,
            cn.INR: np.nan,
            cn.SODIUM: np.nan,
            cn.ALBU: np.nan,
            cn.DIAL_BIWEEKLY: np.nan
        }

        self.__dict__[cn.EXIT_STATUS] = None
        self.__dict__[cn.EXIT_DATE] = None

        self.sim_set = sim_set
        self.listing_offset = (
            self.__dict__[cn.LISTING_DATE] - sim_set.SIM_START_DATE
        ) / timedelta(days=1)

        # Tracking time how long a MELD status has been active.
        self.last_update_time = None
        self.age_nat_meld = np.zeros_like(sim_set.MELD_GRID)
        self.age_int_meld = np.zeros_like(sim_set.MELD_GRID)

        # Initialize HU and ACO since
        self.hu_since = np.nan
        self.__dict__[cn.ACO_SINCE] = np.nan

        # Add time since previous transplantation
        self.__dict__[cn.TIME_SINCE_PREV_TXP] = (
            None if pd.isnull(time_since_prev_txp) else time_since_prev_txp
        )
        if self.__dict__[cn.TIME_SINCE_PREV_TXP] is not None:
            self.__dict__[cn.AGE_PREV_TXP] = (
                (
                    self.age_days_at_listing
                ) - self.__dict__[cn.TIME_SINCE_PREV_TXP]
            ) / 365.25
        else:
            self.__dict__[cn.AGE_PREV_TXP] = None

        # Add dummy indicating whether reregistration makes patient
        # HU eligible
        self.__dict__[cn.HU_ELIGIBLE_REREG] = (
            self.__dict__[cn.TIME_SINCE_PREV_TXP] <= es.CUTOFF_RETX_HU_ELIGIBLE
            if self.__dict__[cn.TIME_SINCE_PREV_TXP]
            else False
        )

        self.initialized = False
        self.profile = None
        self.listed_kidney = listed_kidney
        self.__dict__[cn.PATIENT_SEX] = sex
        self.__dict__[cn.PATIENT_FEMALE] = int(sex == 'Female')
        assert type_retx in (
            cn.NO_RETRANSPLANT, cn.RETRANSPLANT, cn.RETRANSPLANT_DURING_SIM
        )
        self.__dict__[cn.TYPE_RETX] = type_retx
        self.__dict__[cn.IS_RETRANSPLANT] = (
            0 if type_retx and type_retx == cn.NO_RETRANSPLANT
            else 1 if type_retx
            else None
        )
        self.exit_status = None

        self.disease_group = None

        self.__dict__[cn.ANY_ACTIVE] = False
        self.__dict__[cn.R_LOWWEIGHT] = None

        # Properties we may need to access rapidly
        self._meld_score = None
        self._meld_score_nat = None
        self._meld_score_int = None
        self._alloc_score = None
        self._alloc_score_nat = None
        self._alloc_score_int = None
        self._needed_match_info = None
        self._offer_inherit_cols = None
        self._active = None
        self._pediatric = None

    def check_lowweight(self, lowweight_fun: Callable):
        """Check whether recipient is low-weight"""
        if self.__dict__[cn.R_LOWWEIGHT] is None:
            self.__dict__[cn.R_LOWWEIGHT] = lowweight_fun(self)
        return self.__dict__[cn.R_LOWWEIGHT]

    def check_pediatric(self, ped_fun: Callable, match_age: float):
        """Check whether recipient is low-weight. Only check for
            candidates who were pediatric at past matches.
        """
        if self._pediatric or self._pediatric is None:
            self._pediatric = ped_fun(self, match_age)
        return self._pediatric

    @property
    def needed_match_info(self):
        if not self._needed_match_info:
            self._needed_match_info = {
                k: self.__dict__[k]
                for k
                in es.MTCH_RCRD_PAT_COLS
            }
        return self._needed_match_info

    @property
    def offer_inherit_cols(self):
        if not self._offer_inherit_cols:
            self._offer_inherit_cols = {
                k: self.__dict__[k] for k in es.OFFER_INHERIT_COLS['patient']
            }
        return self._offer_inherit_cols

    def reset_matchrecord_info(self):
        self._needed_match_info = None
        self._offer_inherit_cols = None

    def return_meld_score(
            self,
            rnd_fun: Callable = round_to_int
    ) -> Optional[Union[float, int]]:
        if (value := self.meld_scores[self.sim_set.LAB_MELD]) is not None:
            return rnd_fun(value)
        else:
            return self.sim_set.MIN_MELD_SCORE

    def return_nat_meld_score(
            self,
            rnd_fun: Callable = round_to_int
    ) -> Optional[Union[float, int]]:
        """Return the national match MELD."""
        lab_dm = self.lab_meld_score if \
            self.meld_scores[cn.DM] is None else \
            self.meld_scores[cn.DM]

        if lab_dm is not None:
            return rnd_fun(
                max(
                    lab_dm,
                    self.get_meld(cn.NSE, default=0),
                    self.get_meld(cn.SE, default=0),
                    self.get_meld(cn.PED, default=0),
                )
            )

    def return_int_meld_score(
            self,
            rnd_fun: Callable = round_to_int
            ) -> Optional[Union[float, int]]:
        """Return the international match MELD (ignores SEs and NSEs)"""
        lab_dm = self.lab_meld_score if \
            self.meld_scores[cn.DM] is None else \
            self.meld_scores[cn.DM]

        if lab_dm is not None:
            return rnd_fun(max(lab_dm, self.get_meld(cn.PED, default=0)))

    def return_nonna_int_match_meld(
            self,
            rnd_fun: Callable = round_to_int,
            default=6
    ) -> int:
        """Return international match MELD, not allowing for None"""
        if self.meld_int_match:
            return rnd_fun(self.meld_int_match)
        else:
            return default

    def return_nonna_nat_match_meld(
            self,
            rnd_fun: Callable = round_to_int,
            default=6
    ) -> int:
        """Return national match MELD, not allowing for None"""
        if self.meld_nat_match:
            return rnd_fun(self.meld_nat_match)
        else:
            return default

    def reset_match_melds(self):
        self._meld_score = None
        self._meld_score_nat = None
        self._meld_score_int = None
        self._alloc_score = None
        self._alloc_score_nat = None
        self._alloc_score_int = None

    @property
    def active(self):
        if self._active is None:
            self._active = (
                self.urgency_code in es.ACTIVE_STATUSES and
                self.meld_nat_match is not None
            )
        return self._active

    @property
    def lab_meld_score(self):
        if not self._meld_score:
            self._meld_score = self.return_meld_score()
        return self._meld_score

    @property
    def meld_nat_match(self):
        if not self._meld_score_nat:
            self._meld_score_nat = self.return_nat_meld_score()
        return self._meld_score_nat

    @property
    def meld_int_match(self):
        if not self._meld_score_int:
            self._meld_score_int = self.return_int_meld_score()
        return self._meld_score_int

    def calculate_alloc_score(
            self,
            verbose: int = 0,
            don: Optional[Donor] = None
    ) -> Union[float, int]:
        assert isinstance(self.sim_set.ALLOC_SCORE, AllocationScore)
        x_dict = {}
        if verbose:
            print('******')
            print(f'Donor: {don}')
        for key in self.sim_set.ALLOC_SCORE.raw_variables:
            if (value := self.meld_scores.get(key)) is not None:
                x_dict[key] = value
                if verbose:
                    print(f'{key}: {value}')
            elif (value := self.__dict__.get(key)) is not None:
                x_dict[key] = value
                if verbose:
                    print(f'{key}: {value}')
            elif key == self.sim_set.LAB_MELD:
                x_dict[key] = self.sim_set.MIN_MELD_SCORE
                if verbose:
                    print(f'{key}: {value}')
            elif don:
                if (value := don.__dict__.get(key)) is not None:
                    x_dict[key] = value
                if verbose:
                    print(f'{key}: {value}')
            else:
                raise Exception(
                    f'Cannot calculate allocation score, '
                    f'as {key} does not exist.'
                )

        return self.sim_set.ALLOC_SCORE.calc_score(
            x_dict,
            verbose=verbose
        )

    def return_nat_alloc_score(
        self,
        rnd_fun: Callable = round_to_int,
        verbose: int = 0,
        don: Optional[Donor] = None
    ) -> Union[float, int]:
        """Function which returns the allocation score.

        Based on stored allocation score, if not using donor characteristics
        """

        # Use a cache for allocation score, if allocation score uses donor info
        if not self.sim_set.get('ALLOCATION_SCORE_DONOR_BASED', False):
            if self._alloc_score_nat:
                return self._alloc_score_nat
            else:
                if not self._alloc_score:
                    self._alloc_score = self.calculate_alloc_score(don=don)
                score_dm = self._alloc_score if \
                    self.meld_scores[cn.DM] is None else \
                    self.meld_scores[cn.DM]
                self._alloc_score_nat = rnd_fun(
                        max(
                            score_dm,
                            self.get_meld(cn.NSE, default=0),
                            self.get_meld(cn.SE, default=0),
                            self.get_meld(cn.PED, default=0),
                        )
                    )
                return self._alloc_score_nat
        else:
            score_dm = self.calculate_alloc_score(don=don) if \
                self.meld_scores[cn.DM] is None else \
                self.meld_scores[cn.DM]

            return rnd_fun(
                max(
                    score_dm,
                    self.get_meld(cn.NSE, default=0),
                    self.get_meld(cn.SE, default=0),
                    self.get_meld(cn.PED, default=0),
                )
            )

    def return_int_alloc_score(
        self,
        rnd_fun: Callable = round_to_int,
        verbose: int = 0,
        don: Optional[Donor] = None
    ) -> Union[float, int]:
        """Function which returns the international allocation score.

        Based on stored allocation score, if not using donor characteristics
        """

        # If allocation does not use donor information, use cache
        if not self.sim_set.get('ALLOCATION_SCORE_DONOR_BASED', False):
            if self._alloc_score_int:
                return self._alloc_score_int
            else:
                if not self._alloc_score:
                    self._alloc_score = self.calculate_alloc_score(don=don)
                score_dm = self._alloc_score if \
                    self.meld_scores[cn.DM] is None else \
                    self.meld_scores[cn.DM]
                self._alloc_score_int = rnd_fun(
                    max(score_dm, self.get_meld(cn.PED, default=0))
                )
                return self._alloc_score_int
        else:
            # Otherwise, always recalculate score
            score_dm = self.calculate_alloc_score(don=don) if \
                self.meld_scores[cn.DM] is None else \
                self.meld_scores[cn.DM]

            return rnd_fun(max(score_dm, self.get_meld(cn.PED, default=0)))

    def age_at_t(self, t: float) -> int:
        """Return age at time t (in days)"""
        return floor((self.age_days_at_listing + t) / 365.25)

    def next_update_at(self) -> Optional[float]:
        """
            Return time at which next update is issued,
            offset from calendar time
        """
        if (
            self.future_statuses is not None and
            not self.future_statuses.is_empty()
        ):
            return(
                self.listing_offset +
                self.future_statuses.first().arrival_time
            )

    def update_patient(
            self, se_sys: ExceptionSystem,
            sim_results: Optional[SimResults] = None
            ):
        """Update patient with first next status"""
        if (
            self.future_statuses is not None and
            not self.future_statuses.is_empty()
        ):
            stat_update = self.future_statuses.next()
            current_wt = (
                (stat_update.arrival_time - float(self.last_update_time)) if
                self.last_update_time and self.urgency_code != cn.NT else
                0
            )
            upd_time = stat_update.arrival_time

            if not self.__dict__[cn.ANY_ACTIVE]:
                if not stat_update.before_sim_start:
                    if self.__dict__[cn.URGENCY_CODE] != cn.NT:
                        self.__dict__[cn.ANY_ACTIVE] = True

            # Keep track of accrued time in MELD scores
            self.age_nat_meld = self._upgrade_current_wt(
                meld_grid=self.sim_set.MELD_GRID,
                accrued_wt=self.age_nat_meld,
                current_wt=current_wt,
                meld_score=self.meld_nat_match
            )
            self.age_int_meld = self._upgrade_current_wt(
                meld_grid=self.sim_set.MELD_GRID,
                accrued_wt=self.age_int_meld,
                current_wt=current_wt,
                meld_score=self.meld_int_match
            )

            # Now reset cached matched MELDs; we no longer need them.
            # Also reset other information, inherited to match records
            self.reset_match_melds()
            self.reset_matchrecord_info()

            # Update the patient with the update status
            if self.sim_set.USE_REAL_FU and stat_update.synthetic:
                pass
            elif stat_update.type_status in es.STATUSES_UPDATE_BIOMARKERS:
                if not (
                    self.sim_set.USE_REAL_FU and
                    stat_update.type_status == cn.FU
                ):
                    self._update_meld_status(stat_update, excsys=se_sys)
            elif stat_update.type_status == cn.PED:
                self._update_ped_status(stat_update, excsys=se_sys)
            elif stat_update.type_status in es.STATUSES_EXCEPTION:
                self._update_e_status(stat_update, excsys=se_sys)
            elif stat_update.type_status == cn.URG:
                if stat_update.status_value == cn.FU:
                    if (
                        self.sim_set.USE_REAL_FU or
                        stat_update.before_sim_start
                    ):
                        self.set_transplanted(
                            self.__dict__[cn.LISTING_DATE] + timedelta(
                                days=stat_update.arrival_time
                                )
                        )
                        # TODO: save real transplantation?
                elif stat_update.status_value in es.TERMINAL_STATUSES:
                    self.future_statuses.clear()
                    self.__dict__[cn.FINAL_REC_URG_AT_TRANSPLANT] = (
                        self.urgency_code
                    )
                    self._update_urgency_code(
                        stat_update.status_value,
                        stat_update.status_detail
                    )
                    self.__dict__[cn.EXIT_STATUS] = stat_update.status_value
                    self.__dict__[cn.EXIT_DATE] = (
                        self.__dict__[cn.LISTING_DATE] +
                        timedelta(days=stat_update.arrival_time)
                    )
                    if sim_results:
                        sim_results.save_exit(self)
                else:
                    self._update_urgency(stat_update)
            elif stat_update.type_status == cn.SFU:
                self._update_meld_status(stat_update, excsys=se_sys)
            elif stat_update.type_status == cn.DM:
                self._update_dm(stat_update)
            elif stat_update.type_status == cn.ACO:
                self._update_aco_status(stat_update)
            elif stat_update.type_status == cn.DIAG:
                self._update_patient_diag(stat_update)
            elif isinstance(stat_update, ProfileUpdate):
                self.profile = stat_update.profile
            else:
                print(stat_update)
                print(
                    f'Stopped early for {self.id_recipient}'
                    f'due to unknown status'
                    )
                exit()

            self.last_update_time = upd_time

    def _upgrade_current_wt(
            self, meld_grid: np.ndarray,
            accrued_wt: np.ndarray, meld_score: Optional[float],
            current_wt: float
            ):
        """Update accrued waiting times

        If a patient's MELD score exceeds the score, upgrade by current_wt,
        otherwise reset to 0.
        """

        if meld_score is not None:
            return np.select(
                condlist=[meld_grid <= meld_score],
                choicelist=[accrued_wt+current_wt],
                default=0
            )
        else:
            return accrued_wt

    def get_accrued_waiting_time(
            self,
            current_cal_time: float,
            type_meld: str
            ) -> int:
        """Return time waited in current MELD score."""

        if self.r_aco:
            return (
                current_cal_time -
                self.__dict__[cn.ACO_SINCE] -
                self.listing_offset
            )
        elif self.__dict__[cn.PATIENT_IS_HU]:
            return (current_cal_time - self.hu_since - self.listing_offset)
        else:
            # Retrieve waiting time accrued at last update.
            if type_meld == cn.MELD_INT_MATCH:
                accrued_time_last_update = self.age_int_meld[
                    self.sim_set.MELD_GRID_DICT[
                        self.return_nonna_int_match_meld()
                    ]
                ]
            elif type_meld == cn.MELD_NAT_MATCH:

                accrued_time_last_update = self.age_nat_meld[
                    self.sim_set.MELD_GRID_DICT[
                        self.return_nonna_nat_match_meld()
                    ]
                ]
            else:
                raise ValueError(
                    f'type_meld should be {cn.MELD_INT_MATCH} or '
                    f'{cn.MELD_NAT_MATCH} not {type_meld}'
                )

            # Add an offset to account for the fact that time has elapsed
            # from last patient update to current calendar time.
            if self.last_update_time is not None:
                extra_accrued_time = (
                    current_cal_time - (
                        self.listing_offset + self.last_update_time
                        )
                )
            else:
                extra_accrued_time = 0
            if extra_accrued_time < -0.5:
                print(self.future_statuses)
                print(extra_accrued_time)
                raise ValueError(
                    f'Current calendar time must be greater than last '
                    f'update time.\nLast update time: '
                    f'{self.last_update_time}\n'
                    f'current time: {current_cal_time}\n'
                    f'offset: {self.listing_offset}'
                )

            return (
                ceil(accrued_time_last_update + extra_accrued_time)
            )

    def _update_dm(self, upd: StatusUpdate):
        """Update downmarked status"""
        if upd.status_action == 0:
            self.meld_scores[
                cn.DM
            ] = None
        else:
            if self.sim_set.USE_REAL_DM and isinstance(upd.status_value, str):
                self.meld_scores[
                    cn.DM
                ] = float(upd.status_value)
            else:
                self.meld_scores[
                    cn.DM
                ] = self.sim_set.DOWNMARKED_MELD

    def _update_aco_status(self, upd: StatusUpdate):
        """Update downmarked status"""
        if upd.status_action == 0:
            self.r_aco = False
            self.__dict__[cn.ACO_SINCE] = np.nan
        else:
            self.set_to_aco(upd)

    def _update_patient_diag(self, upd: StatusUpdate):
        """Update downmarked status"""
        self.set_diag(upd)

    def _update_urgency_code(self, code: str, reason: Optional[str] = None):
        """Update urgency code with string"""
        if code not in es.ALL_STATUS_CODES:
            raise Exception(
                f'listing status should be one of:\n\t'
                f'{", ".join(es.ALL_STATUS_CODES)},\n not {code}'
            )
        self.__dict__[cn.URGENCY_CODE] = code
        self.__dict__[cn.PATIENT_IS_HU] = code == cn.HU
        self._active = None
        if reason:
            self.__dict__[cn.URGENCY_REASON] = reason

    def _update_urgency(self, upd: StatusUpdate):
        """Update urgency code"""
        assert upd.status_value in es.ALLOWED_STATUSES, \
            f'listing status should be one of:\n\t' \
            f'{", ".join(es.ALLOWED_STATUSES)}, not {upd.status_value}'
        self._update_urgency_code(
            upd.status_value,
            upd.status_detail
        )

        if self.urgency_code == cn.T:
            self.__dict__[cn.LAST_NONNT_HU] = False
            if not self.__dict__[cn.ANY_ACTIVE]:
                self.__dict__[cn.ANY_ACTIVE] = True
        elif self.urgency_code == cn.HU:
            self.__dict__[cn.LAST_NONNT_HU] = True
            if not self.__dict__[cn.ANY_ACTIVE]:
                self.__dict__[cn.ANY_ACTIVE] = True
            self.__dict__[cn.ANY_HU] = True
            self.hu_since = upd.arrival_time
        else:
            self.hu_since = np.nan

    def set_to_aco(self, upd: StatusUpdate):
        """Set patient to ACO"""
        self.__dict__[cn.R_ACO] = True
        self.__dict__[cn.ACO_SINCE] = upd.arrival_time

    def set_diag(self, upd: StatusUpdate):
        if self.__dict__[cn.DISEASE_GROUP] != upd.status_value:
            self.__dict__[cn.DISEASE_GROUP] = upd.status_value
            self.__dict__[cn.DISEASE_SINCE] = upd.arrival_time

    def set_dialysis(self, status: bool = True):
        """Set patient on dialysis"""
        self.biomarkers[cn.DIAL_BIWEEKLY] = status

    def _update_meld_status(
            self, upd: StatusUpdate, excsys: ExceptionSystem
            ):
        """Update MELD status. Also update MELD-SE if bonus exception."""

        if upd.biomarkers:
            self.biomarkers.update(
                upd.biomarkers
            )
        # Save the real historic MELD lab value.
        # Do not update in case of real FU.
        if upd.type_status != cn.FU:
            self.meld_scores[
                cn.SCORE_LABHISTORIC
            ] = float(upd.status_value)

        # Calculate MELD-like scores
        for k, score_fun in self.sim_set['SCORES'].items():
            self.meld_scores[
                k
            ] = score_fun.calc_score(self.biomarkers)

        # Re-calculate bonus MELD if lab is updated.
        if (
            self.exception is not None and
            self.meld_scores[self.sim_set.LAB_MELD] is not None
        ):
            if self.exception.adds_bonus:
                self.meld_scores[
                    self.exception.type_e
                ] = excsys.calculate_bonus_meld(
                    self.meld_scores[self.sim_set.LAB_MELD],
                    bonus_amount=self.exception.adds_bonus_amount
                )

    def _update_ped_status(self, upd: StatusUpdate, excsys: ExceptionSystem):
        """Update a patients pediatric status"""
        if upd.status_action == 0:
            self.ped_status = None
            self.meld_scores[cn.PED] = 0
        else:
            if self.ped_status is not None:
                if self.sim_set.USE_REAL_PED:
                    self.ped_status.set_equivalent(
                        float(upd.status_value)
                    )
                else:
                    self.ped_status.upgrade_equivalent()
            else:
                self.ped_status = deepcopy(
                        excsys.exceptions[
                            upd.type_status
                            ][
                                upd.status_detail
                            ]
                        )

                # Initialize the bonus SE with the real assigned value.
                if self.sim_set.USE_REAL_PED:
                    self.ped_status.set_equivalent(
                        float(upd.status_value)
                    )
                else:
                    self.ped_status.set_to_initial_equivalent()

            # Update pediatric MELD score
            self.meld_scores[cn.PED] = excsys.get_equivalent_meld(
                mrt=self.ped_status.current_eq
            )

            # If no future pedatric meld update exists,
            # and current mrt. eqv. is less than 100
            # schedule a new future pediatric meld
            if (
                self.ped_status is not None and
                self.ped_status.current_eq < 100 and
                self.future_statuses and
                len(
                    self.future_statuses.return_status_types(
                        event_types=[cn.PED]
                    )
                ) == 0
            ):
                schedule_new_update = True

                # If next update time exceeds max. allowed age,
                # replace the current pediatric status by
                # the replacement SE (if applicable).
                if self.age_at_t(
                    upd.arrival_time + self.ped_status.next_update_in()
                ) >= self.ped_status.max_age_upgrade:
                    if self.ped_status.replacement_se is not np.nan:
                        eq_before = self.ped_status.current_eq
                        self.ped_status = deepcopy(
                            excsys.exceptions[
                                upd.type_status
                                ][
                                    self.ped_status.replacement_se
                                ]
                        )
                        self.ped_status.set_equivalent(
                            float(eq_before)
                        )
                        upd.status_detail = self.ped_status.replacement_se
                    else:
                        schedule_new_update = False

                if schedule_new_update:
                    upd.set_value(
                        min(
                            float(upd.status_value) +
                            self.ped_status.eq_upgrade,
                            self.ped_status.eq_max
                        )
                    )
                    upd.set_arrival_time(
                        upd.arrival_time + self.ped_status.next_update_in()
                    )
                    if self.future_statuses:
                        self.future_statuses.add(upd)

    def _delete_e_status(
        self, type_status='str',
    ) -> None:
        """Delete exception"""
        self.meld_scores[type_status] = 0
        self.__dict__[cn.E_SINCE] = None
        self.exception = None
        self.meld_scores[type_status] = 0
        self.type_e = 'None'

    def _update_e_status(
            self, upd: StatusUpdate,
            excsys: ExceptionSystem
            ):
        """Update (N)SE statuses of patient based on StatusUpdate"""
        if upd.status_action == 0:
            self._delete_e_status(upd.type_status)
        else:
            # (i) if exception remains same, upgrade,
            # (ii) If patient has no exception yet, set new exception,
            # (iii) if exception changes, but old one is bonus
            if (
                isinstance(self.exception, ExceptionScore) and
                self.exception.__dict__[cn.ID_SE] == upd.status_detail
            ):
                if (
                        (
                            (self.exception.use_real_se) or
                            (
                                upd.before_sim_start and
                                self.exception.retain_historic_se
                            )
                        ) and
                        not self.exception.adds_bonus
                ):
                    self.exception.set_equivalent(
                        float(upd.status_value)
                    )
                elif not self.exception.adds_bonus:
                    self.exception.upgrade_equivalent()
            else:
                if (
                    self.exception is None or
                    (
                        isinstance(self.exception, ExceptionScore) and
                        (
                            self.exception.__dict__[cn.ID_SE] !=
                            upd.status_detail
                        ) and (
                            excsys.exceptions[upd.type_status][
                                upd.status_detail
                            ].adds_bonus or
                            (upd.status_detail in es.REPLACING_BONUS_SES)
                        )
                    )
                ):
                    # Initialize the exception to real exception if
                    # real updates are used, otherwise set to initial eqv.
                    self.exception = deepcopy(
                        excsys.exceptions[
                            upd.type_status
                            ][
                                upd.status_detail
                            ]
                    )

                    assert isinstance(self.exception, ExceptionScore), \
                        f'Exception should be ExceptionScore'

                    # Keep track of since when patient has SE.
                    self.__dict__[cn.E_SINCE] = upd.arrival_time

                    # Initialize the bonus SE.
                    if (
                        (
                            self.exception.use_real_se and
                            not self.exception.adds_bonus
                            ) or
                        (
                            upd.before_sim_start and
                            self.exception.real_se_before_simstart
                        )
                    ) and not isnan(float(upd.status_value)):
                        self.exception.set_equivalent(
                            float(upd.status_value)
                        )
                    elif not self.exception.adds_bonus:
                        self.exception.set_to_initial_equivalent()
                else:
                    warn(
                        f'Status update {upd.type_status} '
                        f'with {upd.status_detail} scheduled, '
                        f'but patient {self.id_recipient} is '
                        f'{self.exception.__dict__[cn.ID_SE]} '
                        f'{self.exception.type_e}.'
                        f'\nNot resetting time since E.'
                    )
                    self.exception = deepcopy(
                        excsys.exceptions[
                            upd.type_status
                            ][
                                upd.status_detail
                            ]
                        )

                    self.exception.set_equivalent(
                            float(upd.status_value)
                    )
            # Set type E for patient
            if self.exception.type_e == cn.NSE:
                self.type_e = cn.NSE
            elif mgr.HCC in self.exception.dis:
                self.type_e = mgr.HCC
            else:
                self.type_e = mgr.OTHER_SE

            # Update MELD scores
            if self.exception is not None:
                if not self.exception.adds_bonus:
                    self.meld_scores[
                        self.exception.type_e
                    ] = excsys.get_equivalent_meld(
                        mrt=self.exception.current_eq
                    )
                elif self.meld_scores[self.sim_set['LAB_MELD']] is not None:
                    self.meld_scores[
                        self.exception.type_e
                    ] = excsys.calculate_bonus_meld(
                        self.meld_scores[self.sim_set['LAB_MELD']],
                        self.exception.adds_bonus_amount
                    )

            # If no future status update exists,
            # schedule a future exception status update
            if (
                self.future_statuses is not None and
                self.exception is not None and
                len(
                    self.future_statuses.return_status_types(
                        event_types=[cn.SE, cn.NSE]
                        )
                ) == 0
            ):
                # Schedule new update only if patient age
                # does not exceed maximum age
                if (
                    self.age_at_t(
                        upd.arrival_time + self.exception.next_update_in()
                    ) < self.exception.max_age_upgrade and
                    (
                        self.exception.current_eq is not None and
                        self.exception.current_eq < self.exception.eq_max
                    )
                ):
                    # Update arrival time, and whether
                    # it occurs before the listing date.
                    upd.set_arrival_time(
                        upd.arrival_time +
                        self.exception.next_update_in()
                    )

                    if (
                        self.sim_set.USE_REAL_SE and
                        isinstance(upd.status_value, str) and
                        isDigit(upd.status_value)
                    ):
                        upd.set_value(
                            float(upd.status_value) +
                            self.exception.eq_upgrade
                        )
                    else:
                        upd.set_value(
                            float(self.exception.current_eq) +
                            self.exception.eq_upgrade
                        )

                    self.future_statuses.add(upd)

    def set_transplanted(
            self, tx_date: datetime,
            donor: Optional[Donor] = None,
            sim_results: Optional[SimResults] = None,
            match_record: Optional[
                'simulator.code.AllocationSystem.MatchRecord'
            ] = None
    ):
        """Set patient to transplanted"""
        if self.future_statuses:
            self.future_statuses.clear()
        self.__dict__[cn.EXIT_DATE] = tx_date
        self.__dict__[cn.EXIT_STATUS] = cn.FU
        self.__dict__[cn.FINAL_REC_URG_AT_TRANSPLANT] = (
            self.__dict__[cn.URGENCY_CODE]
        )
        self._update_urgency_code(cn.FU)
        if donor:
            self.id_received_donor = donor.id_donor
        else:
            self.id_received_donor = 'Donor'
        if sim_results:
            sim_results.save_transplantation(
                pat=self, matchr=match_record
            )

    def preload_status_updates(
            self,
            fut_stat: PatientStatusQueue
            ):
        """Initialize status updates for patient"""
        self.initialized = True
        self.__dict__[cn.TIME_LIST_TO_REALEVENT] = (
            fut_stat.return_time_to_exit(
                exit_statuses=es.EXITING_STATUSES
            )
        )
        self.future_statuses = fut_stat

    def trigger_historic_updates(
        self,
        se_sys: ExceptionSystem
    ):
        """Trigger all updates which occured before sim start"""
        if self.future_statuses is not None:
            while (
                not self.future_statuses.is_empty() and
                (
                    self.future_statuses.first().before_sim_start or
                    (self.future_statuses.first().arrival_time < 0)
                )
            ):
                self.update_patient(se_sys=se_sys)
            if self.__dict__[cn.URGENCY_CODE] != 'NT':
                self.__dict__[cn.ANY_ACTIVE] = True
            self.__dict__[cn.INIT_URG_CODE] = (
                self.__dict__[cn.URGENCY_CODE]
            )

    def is_initialized(self) -> bool:
        """Whether future statuses have been loaded for the patient"""
        return self.initialized

    def get_meld(self, type, default: float) -> float:
        """Retrieve a MELD score, returning default if None"""
        meld = self.meld_scores.get(type)
        if meld:
            return meld
        return default

    def _print_dict(self, dic) -> None:
        """Method to print a dictionary."""
        print(' '.join([f'{k}: {str(v).ljust(4)}\t' for k, v in dic.items()]))

    def print_biomarkers(self) -> None:
        """Print biomarkers"""
        self._print_dict(self.biomarkers)

    def print_melds(self) -> None:
        """Print MELD scores"""
        self._print_dict(self.meld_scores)

    def trigger_all_status_update(
            self, se_sys: ExceptionSystem,
            verbose: bool = False
            ) -> None:
        """Function to trigger all status updates for a patient"""
        if self.future_statuses is not None:
            while not self.future_statuses.is_empty():
                if verbose:
                    print(self.future_statuses.first())
                self.update_patient(se_sys=se_sys)
                if verbose:
                    self.print_biomarkers()
                    self.print_melds()
                    print(f'Nat. match-MELD: {self.meld_nat_match}')

    def schedule_death(self, fail_date: datetime) -> None:
        """Schedule a death update"""
        death_event = StatusUpdate(
            type_status=cn.URG,
            arrival_time=(
                fail_date - self.__dict__[cn.LISTING_DATE]
            ) / timedelta(days=1),
            biomarkers={},
            status_action=1,
            status_detail='',
            status_value=cn.D,
            sim_start_time=(
                self.__dict__[cn.LISTING_DATE] - self.sim_set.SIM_START_DATE
            ) / timedelta(days=1)
        )
        self.future_statuses.add(
            death_event
        )
        self.future_statuses.truncate_after(
            truncate_time=(
                fail_date - self.__dict__[cn.LISTING_DATE]
            ) / timedelta(days=1)
        )

    def get_acceptance_prob(self) -> float:
        """Simulate an acceptance probability"""
        prob = self.rng_acceptance.random()
        self.__dict__[cn.DRAWN_PROB] = round_to_decimals(prob, 3)
        return prob

    def return_dict_with_melds(self) -> Dict[str, Any]:
        result = self.__dict__
        result.update(
            {
                cn.MELD_LAB: self.meld_scores[self.sim_set.LAB_MELD],
                self.sim_set.LAB_MELD: self.meld_scores[self.sim_set.LAB_MELD],
                cn.MELD_NAT_MATCH: self.meld_nat_match,
                cn.MELD_INT_MATCH: self.meld_int_match,
                cn.DIAL_BIWEEKLY: self.biomarkers[cn.DIAL_BIWEEKLY]
            }
        )
        return result

    def __str__(self):
        if self.__dict__[cn.EXIT_STATUS] is not None:
            return(
                f'Patient {self.id_recipient}, exited on '
                f'{self.__dict__[cn.EXIT_DATE]} with status '
                f' {self.__dict__[cn.EXIT_STATUS]} '
                f'(delisting reason: {self.urgency_reason})'
                )

        return(
            f'Patient {self.id_recipient}, listed on '
            f'{self.__dict__[cn.LISTING_DATE].strftime("%d-%m-%Y")} '
            f'at center {self.__dict__[cn.RECIPIENT_CENTER]} with '
            f'current status {self.urgency_code} and '
            f'nat match MELD: {self.meld_nat_match}'
            )

    def __lt__(self, other):
        """Order by match tuple."""
        return self.meld_int_match < other.meld_int_match

    def __repr__(self):
        return f'Patient {self.id_recipient}, listed on ' \
               f'{self.__dict__[cn.LISTING_DATE].strftime("%d-%m-%Y")}: ' \
               f'at center {self.__dict__[cn.RECIPIENT_CENTER]}\n'


class Profile:
    """Class which implements an obligation
    ...

    Attributes   #noqa
    ----------
    debtor: str
        the debtor of the obligation (i.e., the party who received the
         organ and has to return)
    creditor: str
        the creditor of the obligation (i.e., the party who offered the
        organ which needs to be returned)
    obl_insert_date: datetime
        date that the obligation was created

    Methods
    -------
    _check_acceptable(self, don: Donor) -> bool
        Check whether donor is acceptable according to profile.
    """
    __slots__ = [
            'min_age', 'max_age', 'min_weight', 'max_weight', 'hbsag',
            'hcvab', 'hbcab', 'sepsis', 'meningitis', 'malignancy',
            'drug_abuse', 'marginal', 'rescue', 'acceptable_types',
            'euthanasia', 'dcd'
            ]

    def __init__(
        self, min_age: int, max_age: int,
        min_weight: int, max_weight: int,
        hbsag: bool, hcvab: bool, hbcab: bool,
        sepsis: bool, meningitis: bool,
        malignancy: bool, drug_abuse: bool,
        marginal: bool, rescue: bool,
        lls: bool, erl: bool, rl: bool,
        ll: bool, euthanasia: bool,
        dcd: bool
    ) -> None:

        self.min_age = min_age
        self.max_age = max_age
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.hbsag = hbsag
        self.hcvab = hcvab
        self.hbcab = hbcab
        self.sepsis = sepsis
        self.meningitis = meningitis
        self.malignancy = malignancy
        self.drug_abuse = drug_abuse
        self.marginal = marginal
        self.rescue = rescue
        self.acceptable_types = [es.TYPE_OFFER_WLIV]
        self.euthanasia = euthanasia
        self.dcd = dcd
        if lls:
            self.acceptable_types.append(es.TYPE_OFFER_LLS)
        if erl:
            self.acceptable_types.append(es.TYPE_OFFER_ERL)
        if rl:
            self.acceptable_types.append(es.TYPE_OFFER_ERL)
        if ll:
            self.acceptable_types.append(es.TYPE_OFFER_LL)

    def _check_acceptable(self, don: Donor, verbose=False) -> bool:
        """Check whether donor is acceptable to patient."""

        # Profiles are not used for HU patients.
        if don.__dict__[cn.D_DCD] > self.dcd:
            if verbose:
                print('DCD-incompatible')
            return False
        if don.__dict__[cn.D_AGE] < self.min_age:
            if verbose:
                print(f'{don.__dict__[cn.D_AGE]} >= {self.min_age}')
            return False
        if don.__dict__[cn.D_AGE] > self.max_age:
            if verbose:
                print(f'Donor age {don.__dict__[cn.D_AGE]} > {self.max_age}')
            return False
        if don.d_weight < self.min_weight:
            if verbose:
                print(f'Donor weight {don.d_weight} <= {self.min_weight}')
            return False
        if don.d_weight > self.max_weight:
            if verbose:
                print(f'Donor weight {don.d_weight} > {self.max_weight}')
            return False
        if don.hbsag > self.hbsag:
            if verbose:
                print('HBsag incompatible')
            return False
        if don.hcvab > self.hcvab:
            if verbose:
                print('HCVab incompatible')
            return False
        if don.hbcab > self.hbcab:
            if verbose:
                print('HBCab incompatible')
            return False
        if don.sepsis > self.sepsis:
            if verbose:
                print('sepsis incompatible')
            return False
        if don.meningitis > self.meningitis:
            if verbose:
                print('meningitis incompatible')
            return False
        if don.malignancy > self.malignancy:
            if verbose:
                print('malignancy incompatible')
            return False
        if don.marginal > self.marginal:
            if verbose:
                print('marginal incompatible')
            return False
        if don.rescue > self.rescue:
            if verbose:
                print('rescue incompatible')
            return False
        if don.first_offer_type not in self.acceptable_types:
            if verbose:
                print('Offer type incompatible')
            return False
        if don.euthanasia > self.euthanasia:
            if verbose:
                print('Euthanasia incompatible')
            return False
        return True

    def __str__(self):
        return str(
            {slot: getattr(self, slot) for slot in self.__slots__}
        )


class Obligation:
    """Class which implements an obligation
    ...

    Attributes   #noqa
    ----------
    debtor: str
        the debtor of the obligation (i.e., the party who received the
         organ and has to return)
    creditor: str
        the creditor of the obligation (i.e., the party who offered the
        organ which needs to be returned)
    obl_insert_date: datetime
        date that the obligation was created
    start_date: datetime
        obligation start date. May differ from the creation date
        for linkable obligations

    Methods
    -------

    """

    _ids = count(0)

    def __init__(
        self, bloodgroup: str,
        creditor: str, debtor: str,
        obl_insert_date: datetime,
        start_date: Optional[datetime] = None,
        obl_init_reason: Optional[str] = None,
        obl_init_donorid: Optional[str] = None,
        obl_exit_reason: Optional[str] = None,
        obl_exit_donorid: Optional[str] = None,
        obl_end_date: Optional[datetime] = None
    ):
        self.obl_id = next(self._ids)
        self.bloodgroup = bloodgroup
        self.debtor = debtor
        self.creditor = creditor
        assert isinstance(obl_insert_date, datetime), \
            "creation date should be datetime object"
        self.obl_insert_date = obl_insert_date
        if start_date:
            self.obl_start_date = start_date
        else:
            self.obl_start_date = obl_insert_date

        self.obl_init_reason = obl_init_reason
        self.obl_init_donorid = obl_init_donorid
        self.obl_exit_reason = obl_exit_reason
        self.obl_exit_donorid = obl_exit_donorid
        self.obl_end_date = obl_end_date

    @classmethod
    def from_patient(
        cls, patient: Patient, donor: Donor,
        obl_insert_date: datetime
    ):
        """Create an obligation from patient / donor"""
        creditor = (
                donor.donor_country
                if donor.donor_country in es.OBLIGATION_COUNTRIES
                else donor.donor_center
            )
        debtor = (
                patient.__dict__[cn.RECIPIENT_COUNTRY]
                if patient.__dict__[cn.RECIPIENT_COUNTRY]
                in es.OBLIGATION_COUNTRIES
                else patient.__dict__[cn.RECIPIENT_CENTER]
            )
        if debtor == creditor:
            return None
        return cls(
            bloodgroup=donor.d_bloodgroup,
            creditor=creditor,
            debtor=debtor,
            obl_insert_date=obl_insert_date
        )

    @classmethod
    def from_matchrecord(
        cls, match_record: 'simulator.code.AllocationSystem.MatchRecord'
    ):
        """Create an obligation from patient / donor"""
        creditor = (
                match_record.donor.donor_country
                if match_record.donor.donor_country in es.OBLIGATION_COUNTRIES
                else match_record.donor.donor_center
            )
        debtor = (
            match_record.patient.__dict__[cn.RECIPIENT_COUNTRY]
            if match_record.patient.__dict__[cn.RECIPIENT_COUNTRY]
            in es.OBLIGATION_COUNTRIES
            else match_record.patient.__dict__[cn.RECIPIENT_CENTER]
        )
        if debtor == creditor:
            return None

        if match_record.__dict__[cn.PATIENT_IS_HU]:
            obl_reason = cn.HU
        elif match_record.patient.__dict__[cn.R_ACO]:
            obl_reason = cn.ACO
        elif match_record.__dict__[cn.MTCH_OBL]:
            obl_reason = cn.MTCH_OBL
        else:
            obl_reason = 'unknown'
            print(obl_reason)

        return cls(
            bloodgroup=match_record.donor.d_bloodgroup,
            creditor=creditor,
            debtor=debtor,
            obl_insert_date=match_record.__dict__[cn.MATCH_DATE],
            obl_init_reason=obl_reason,
            obl_init_donorid=match_record.donor.__dict__[cn.ID_DONOR]
        )

    def __str__(self):
        return f'{self.bloodgroup}{self.obl_id}, ' \
               f'creation date {self.obl_insert_date.strftime("%d-%m-%Y")}: ' \
               f'start date {self.obl_insert_date.strftime("%d-%m-%Y")}: ' \
               f'{self.debtor} -> {self.creditor}'

    def __repr__(self):
        return f'{self.bloodgroup}{self.obl_id}, ' \
               f'creation date {self.obl_insert_date.strftime("%d-%m-%Y")}: ' \
               f'start date {self.obl_insert_date.strftime("%d-%m-%Y")}: ' \
               f'{self.debtor} -> {self.creditor}'

    def __lt__(self, other):
        """Youngest obligation first (index then corresponds to age)."""
        return self.obl_start_date > other.obl_start_date


class LinkedObligation(Obligation):
    """Linked obligation. Inherits from Obligation."""

    def __init__(
            self, debtor: str, creditor: str, bloodgroup: str,
            obl_insert_date: datetime, obl_start_date: datetime,
            init_donor_id: str
    ) -> None:
        """Linked obligation"""
        self.obl_id = next(Obligation._ids)
        self.bloodgroup = bloodgroup
        self.debtor = debtor
        self.creditor = creditor
        self.obl_insert_date = obl_insert_date
        self.obl_start_date = obl_start_date
        self.obl_init_reason = 'link'
        self.obl_init_donorid = init_donor_id
        self.obl_exit_donorid = None
        self.obl_exit_reason = None
        self.obl_end_date = None


class CountryObligations:
    """ Obligation system per bloodgroup within ELAS
    ...

    Attributes   #noqa
    ----------
    parties: str
        names of the parties with obligations

    initial_obligations: Dict[str, Dict[Tuple[str, str], List[Obligation]]]
        initialization of obligations. For each BG a dictionary with the
        (debtor, creditor)-tuple as key, and a list of obligations as value.

    obligations: Dict[str, Dict[(str, str), List[Obligation]]]
        Nested dictionary with bloodgroup as first key. Then, a dictionary with
        (debtor, creditor) as the key and a list of obligations as value.

    verbose: int
        whether to print messages of updates to obligations. 0 prints nothing,
        1 prints some messages, 2 prints all.

    Methods
    -------
    print_obligations_for_bg(self, bloodgroup: str)
        prints obligations for a given bloodgroup.

    return_obligations_for_debtor(debtor: str)
        returns obligations for the debtor.

    update_with_new_obligation(self, patient: Patient, donor: Donor,
                                   creation_time: datetime):
        adds an obligation from patient country to donor country,
        and links obligations, if necessary.

    add_obligation_from_match_record(self, match_record, creation_time):
        Create obligation directly from match record (helps with metadata)

    return_obligation_ranks(self, bloodgroup: str, debtor_party):
        returns ranks for creditors of obligations.

    _add_obligation_return_linkables(
        patient: Patient, donor: Donor, creation_time: datetime\
            ):
        adds an obligation to the system.

    _retrieve_oldest_debt(bloodgroup, debtor_party)
        retrieves oldest debt of debtor country

    _retrieve_oldest_credit(bloodgroup, creditor_party)
        retrieve oldest credit of creditor country

    _return_generated_obligations()

    """

    def __init__(
            self, parties: List[str],
            initial_obligations:
            Dict[str, Dict[Tuple[str, str], List[Obligation]]],
            verbose: int = 1
            ) -> None:

        self._parties = parties
        self.n_parties = len(parties)
        self.verbose = int(verbose)

        # Check if for each bloodgroup all debtor/creditor keys are included.
        if not all(
            (
                all(p in bg_dict.keys() for p in product(parties, parties)) for
                _, bg_dict in initial_obligations.items()
            )
        ):
            raise ValueError("For each bloodgroup, all creditor/debtor "
                             "combinations should be included")
        else:
            self.obligations = deepcopy(initial_obligations)

        self.generated_obligations = {
            bg: {
                tp: [] for tp in obl.keys()
                }
            for bg, obl in initial_obligations.items()
        }

    def __str__(self) -> str:
        """Print the outstanding obligations per bloodgroup"""
        df_obl = self._current_obl_as_df()
        return df_obl.to_string()

    def print_obligations_for_bg(self, bloodgroup):
        """Print the outstanding obligations per bloodgroup."""
        df_obl = self._current_obl_as_df()
        bg_mask = df_obl.index.get_level_values(0) == bloodgroup
        print(df_obl[bg_mask].to_string())

    def _current_obl_as_df(self) -> pd.DataFrame:
        """Return current obligations as pd.DataFrame"""
        df_obl = pd.DataFrame.from_records(
            [(i, j, len(v)) for i in self.obligations.keys()
             for j, v in self.obligations[i].items()],
            columns=['BG', 'Obl', 'Number of obligations']
        )
        df_obl[['Debtor', 'Creditor']] = pd.DataFrame(df_obl['Obl'].tolist(),
                                                      index=df_obl.index)
        df_obl = df_obl.drop(columns='Obl')
        df_obl = df_obl.pivot(
            index=['BG', 'Debtor'],
            columns='Creditor',
            values='Number of obligations'
        )
        return df_obl

    def _add_obligation_return_linkables(self, obl: Obligation,
                                         creation_time: datetime) -> \
            Tuple[Optional[List[Obligation]], Optional[LinkedObligation]]:
        """Create the new obligation and return whether it results
        in linkables"""

        if self.verbose > 0:
            print(f'Adding obligation from {obl.debtor} to {obl.creditor} at '
                  f'{obl.obl_insert_date.strftime("%d-%m-%Y")} '
                  f'with start date {obl.obl_start_date.strftime("%d-%m-%Y")} '
                  f'(obl_id: {obl.bloodgroup}{obl.obl_id})')

        # Append the new obligation to existing obligations
        self.obligations[obl.bloodgroup][(obl.debtor, obl.creditor)].\
            append(obl)

        # Check if the obligation introduces a cycle or path. If so,
        # return linkable obligations and potentially a new linked obligation
        linkable_obls, linked_obl = self._find_linkable_obligations(obl)

        return linkable_obls, linked_obl

    def update_with_new_obligation(
            self, patient: Patient, donor: Donor,
            creation_time: datetime, track: bool = False
            ) -> None:
        """Add a new obligation to the system, and update based on linking
         obligations"""

        if not isinstance(patient, Patient):
            raise TypeError(f'patient must be Patient, not a {type(patient)}')
        if not isinstance(donor, Donor):
            raise TypeError(f'donor must be Donor, not a {type(donor)}')

        obl = Obligation.from_patient(
            patient=patient,
            donor=donor,
            obl_insert_date=creation_time
        )

        if obl:
            linkable_obligations, linked_obl = \
                self._add_obligation_return_linkables(
                    obl, creation_time
                )

            # If there are linkable obligations (a cycle or path),
            # remove linkables and add linked obligation (if applicable)
            if linkable_obligations is not None:
                if self.verbose > 0:
                    print('Removing linkable obligations:')
                    print(linkable_obligations)

                for el in linkable_obligations:
                    self.obligations[
                        el.bloodgroup
                        ][(el.debtor, el.creditor)].remove(el)

                if linked_obl is not None:
                    if self.verbose > 0:
                        print('Adding linked obligations:')
                        print(linked_obl)
                    self.obligations[
                        linked_obl.bloodgroup
                        ][
                            (linked_obl.debtor, linked_obl.creditor)
                        ].append(linked_obl)

    def add_obligation_from_matchrecord(
        self, match_record: 'simulator.code.AllocationSystem.MatchRecord',
        track: bool = False
    ) -> None:
        """Add a new obligation to the system, and update based on linking
         obligations"""

        if not isinstance(match_record.patient, Patient):
            raise TypeError(
                f'patient must be Patient, not a {type(match_record.patient)}'
                )
        if not isinstance(match_record.donor, Donor):
            raise TypeError(
                f'donor must be Donor, not a {type(match_record.donor)}'
                )

        new_obl = Obligation.from_matchrecord(
            match_record
        )

        if new_obl:
            linkable_obligations, linked_obl = \
                self._add_obligation_return_linkables(
                    new_obl,
                    creation_time=match_record.__dict__[cn.MATCH_DATE]
                )
            if track:
                self.generated_obligations[new_obl.bloodgroup][
                    (new_obl.debtor, new_obl.creditor)
                    ].append(new_obl)

            # If there is a linked obligation, we're dealing with a path.
            # Add the linked obligation to all obligations, and remove
            # any linkables
            if linked_obl:
                if track:
                    self.generated_obligations[linked_obl.bloodgroup][
                        (linked_obl.debtor, linked_obl.creditor)
                        ].append(linked_obl)
                if linkable_obligations:
                    if self.verbose > 0:
                        print('Removing linkable obligations (path):')
                        print(linkable_obligations)
                    for el in linkable_obligations:
                        self.obligations[
                            el.bloodgroup
                            ][(el.debtor, el.creditor)].remove(el)
                        el.obl_exit_reason = 'link'
                        el.obl_end_date = new_obl.obl_insert_date
                    linkable_obligations[0].obl_exit_donorid = (
                        linkable_obligations[1].obl_init_donorid
                    )
                    linkable_obligations[1].obl_exit_donorid = (
                        linkable_obligations[0].obl_init_donorid
                    )

                if self.verbose > 0:
                    print(
                        f'Adding obligation from {linked_obl.debtor}'
                        f' to {linked_obl.creditor} at '
                        f'{linked_obl.obl_insert_date.strftime("%d-%m-%Y")} '
                        f'with start date '
                        f'{linked_obl.obl_start_date.strftime("%d-%m-%Y")} '
                        f'(obl_id: {linked_obl.bloodgroup}{linked_obl.obl_id})'
                    )
                self.obligations[
                    linked_obl.bloodgroup
                ][
                    (linked_obl.debtor, linked_obl.creditor)
                ].append(linked_obl)

            # If there are linkable obligations, but no linked obligation
            # we're dealing with a cycle.
            elif linkable_obligations is not None:
                if self.verbose > 0:
                    print('Removing linkable obligations (cycle):')
                    print(linkable_obligations)
                for el in linkable_obligations:
                    self.obligations[
                        el.bloodgroup
                        ][(el.debtor, el.creditor)].remove(el)
                    el.obl_end_date = new_obl.obl_insert_date
                linkable_obligations[0].obl_exit_reason = (
                    linkable_obligations[1].obl_init_reason
                )
                linkable_obligations[0].obl_exit_donorid = (
                    linkable_obligations[1].obl_init_donorid
                )
                linkable_obligations[1].obl_exit_reason = (
                    linkable_obligations[0].obl_init_reason
                )
                linkable_obligations[1].obl_exit_donorid = (
                    linkable_obligations[0].obl_init_donorid
                )

    def return_obligation_ranks(
            self, bloodgroup: str, debtor_party: str,
            count_all: bool = True
            ) -> Dict[str, int]:
        """Returns obligation ranks (int) for debtor country (str)."""

        assert debtor_party in self.parties, \
            f'{debtor_party} is not a valid debtor.'

        all_debts = self._retrieve_oldest_debts_per_creditor(
            bloodgroup=bloodgroup,
            debtor_party=debtor_party
        )

        if count_all:
            obligation_ranks = dict.fromkeys(self.parties, 0)
        else:
            obligation_ranks = defaultdict(int)
        for obl_rank, debt in enumerate(
            sorted(d for d in all_debts if d is not None)
        ):
            obligation_ranks[debt.creditor] = obl_rank+1

        return obligation_ranks

    def _find_linkable_obligations(self, new_obl: Obligation) -> \
            Tuple[Optional[List[Obligation]], Optional[LinkedObligation]]:
        """Find linkables a new obligation introduces.

        If new obligation introduces a cycle, return list of the cycle and
         no new linked obligation
        If new obligation introduces a path, return list of the path new link.
        """

        # Check if it introduces a cycle. If so, return the oldest obligation.
        if len(self.obligations[new_obl.bloodgroup]
               [(new_obl.creditor, new_obl.debtor)]) > 0:
            oldest_obl = min(
                    self.obligations[new_obl.bloodgroup]
                    [(new_obl.creditor, new_obl.debtor)],
                    key=lambda obl: obl.obl_start_date
                )
            return [oldest_obl, new_obl], None

        # Retrieve the oldest debt to the new creditor
        oldest_debt, _ = self._retrieve_oldest_debt(
            bloodgroup=new_obl.bloodgroup,
            debtor_party=new_obl.creditor
        )

        # Retrieve the oldest credit the new debtor has.
        oldest_credit, _ = self._retrieve_oldest_credit(
            bloodgroup=new_obl.bloodgroup,
            creditor_party=new_obl.debtor
        )

        # If the obligation becomes linkable, resolve it immediately.
        if oldest_credit is None and oldest_debt is None:
            return None, None
        else:
            linkable_obligations = [
                x for x in [oldest_credit, new_obl, oldest_debt]
                if x is not None
            ]

            linked_obl = LinkedObligation(
                bloodgroup=new_obl.bloodgroup,
                debtor=linkable_obligations[0].debtor,
                creditor=linkable_obligations[-1].creditor,
                obl_start_date=linkable_obligations[-1].obl_start_date,
                obl_insert_date=new_obl.obl_insert_date,
                init_donor_id=linkable_obligations[0].obl_init_donorid
            )

            return linkable_obligations, linked_obl

    def _retrieve_all_debts(self, bloodgroup: str, debtor_party: str) \
            -> Generator[Tuple[List[Obligation], str], None, None]:
        """Retrieve all debts the debtor country has.

        Returns a tuple with first element list of obligations, second element
        the creditor.
        """

        assert debtor_party in self.parties, \
            f'{debtor_party} is not a valid debtor.'

        all_debts = (
            (v, pair[1]) for pair, v in self.obligations[bloodgroup].items()
            if pair[0] == debtor_party
        )

        return all_debts

    def _retrieve_oldest_debts_per_creditor(
            self, bloodgroup: str, debtor_party: str
            ) -> Generator[Obligation | None, None, None]:
        """Retrieve oldest debt per creditor."""

        assert debtor_party in self.parties, \
            f'{debtor_party} is not a valid debtor.'

        # Retrieve all debts for selected debtor country. Second element
        # is creditor, first element is debtor.
        all_debts = self._retrieve_all_debts(
            bloodgroup=bloodgroup,
            debtor_party=debtor_party
            )

        # If multiple, filter to the oldest debt per creditor country.
        oldest_debts = (
            min(debts, key=lambda obl: obl.obl_start_date)
            if len(debts) > 0 else None
            for debts, country in all_debts
        )

        return oldest_debts

    def _retrieve_oldest_debt(self, bloodgroup: str, debtor_party: str) \
            -> Tuple[Optional[Obligation], Optional[Tuple[str, str]]]:
        """Retrieve which is the oldest obligation debtor country
         has within bloodgroup."""

        oldest_debts = self._retrieve_oldest_debts_per_creditor(
            bloodgroup=bloodgroup,
            debtor_party=debtor_party
        )

        # Retrieve oldest obligation
        oldest_obl = min(
            [o for o in oldest_debts if o is not None],
            key=lambda obl: obl.obl_start_date,
            default=None
        )

        if oldest_obl is None:
            return None, None
        else:
            return oldest_obl, (debtor_party, oldest_obl.creditor)

    def _retrieve_oldest_credit(self, bloodgroup: str, creditor_party: str) \
            -> Tuple[Optional[Obligation], Optional[Tuple[str, str]]]:
        """Retrieve which is the oldest credit of creditor country within
        selected bloodgroup."""

        # Retrieve all debts for selected debtor country.
        all_credits = (
            (v, k[0]) for k, v in self.obligations[bloodgroup].items()
            if k[1] == creditor_party
        )

        # If multiple, filter to the oldest debt per creditor country.
        oldest_credits = (
            min(creds, key=lambda obl: obl.obl_start_date)
            if len(creds) > 0
            else None
            for creds, country in all_credits
        )

        # Retrieve oldest obligation
        oldest_obl: Optional[Obligation] = min(
            [o for o in oldest_credits if o is not None],
            key=lambda obl: obl.obl_start_date,
            default=None
        )

        if oldest_obl is None:
            return None, None
        else:
            return oldest_obl, (oldest_obl.debtor, creditor_party)

    def _return_generated_obligations(self) -> pd.DataFrame:
        """Returns overview of generated obligations per blood type"""
        rcrds = list()
        for bg, obl_set in self.generated_obligations.items():
            for _, generated_obligations in obl_set.items():
                for obl in generated_obligations:
                    rcrds.append(
                        obl.__dict__
                    )
        df = pd.DataFrame.from_records(rcrds).sort_values(
            by='obl_id'
        )
        return df

    @property
    def parties(self) -> List[str]:
        """Parties involved in the Exception System"""
        return self._parties
