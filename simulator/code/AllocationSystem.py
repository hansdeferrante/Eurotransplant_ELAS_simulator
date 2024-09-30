#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

@author: H.C. de Ferrante
"""

from datetime import timedelta, datetime
from typing import List, Any, Dict, Optional, Type, Tuple, Union, Generator
from itertools import count
import pandas as pd
import numpy as np

from simulator.code.utils import round_to_decimals, round_to_int
from simulator.code.entities import Patient, Donor, \
    CountryObligations
import simulator.magic_values.column_names as cn
import simulator.magic_values.column_groups as cg
import simulator.magic_values.elass_settings as es
import simulator.magic_values.magic_values_rules as mgr

from simulator.magic_values.rules import (
    FL_CENTER_CODES, DICT_CENTERS_TO_REGIONS,
    CENTER_OFFER_GROUPS
    )
from simulator.magic_values.elass_settings import (
        CNTR_OBLIGATION_CNTRIES
    )


class MatchRecord:
    """Class which implements an MatchList
    ...

    Attributes   #noqa
    ----------
    patient: Patient
        Patient
    donor: Donor
        Donor info
    match_date: datetime
        Date that match list is generated.

    Methods
    -------

    """

    _ids = count(0)

    def __init__(
            self, patient: Patient, donor: Donor,
            match_date: datetime,
            type_offer_detailed: int,
            alloc_center: str,
            alloc_region: Optional[str],
            alloc_country: str,
            don_center_offer_group: str,
            obl_ranks: Dict[str, int],
            debtor: str,
            match_time: float,
            dhospital_travel_time_dict: Optional[
                Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, str]]]
            ],
            attr_order_match: Optional[List[str]] = None,
            id_mtr: Optional[int] = None
            ) -> None:

        if attr_order_match is None:
            self.attr_order_match = [cn.ALLOC_SCORE_INT]
        else:
            self.attr_order_match = attr_order_match

        self.id_mr = next(self._ids)
        self.id_mtr = id_mtr
        self.__dict__[cn.MATCH_DATE] = match_date
        self.match_time = match_time
        self.patient = patient
        self.donor = donor
        self.__dict__[cn.OFFERED_TO] = 'recipient'

        # Copy over selected attributes from patient and donor
        self.__dict__.update(
            {
                **patient.needed_match_info,
                **donor.needed_match_info
            }
        )

        self.__dict__[cn.TYPE_OFFER_DETAILED] = type_offer_detailed
        self.__dict__[cn.TYPE_OFFER] = (
            cn.WLIV if type_offer_detailed == es.TYPE_OFFER_WLIV
            else cn.SPLIT
        )

        # Add match list geography.
        self.__dict__[cn.D_ALLOC_CENTER] = alloc_center
        self.__dict__[cn.D_ALLOC_REGION] = alloc_region
        self.__dict__[cn.D_ALLOC_COUNTRY] = alloc_country
        self.__dict__[cn.DON_CENTER_OFFER_GROUP] = (
            don_center_offer_group
        )

        # Add travel time information
        if dhospital_travel_time_dict:
            self.travel_time_dict = dhospital_travel_time_dict

        # Add obligation info
        self.__dict__[cn.OBL_DEBTOR] = debtor
        self._add_obligations(
            obl_ranks,
            rec_country=self.__dict__[cn.RECIPIENT_COUNTRY],
            rec_center=self.__dict__[cn.RECIPIENT_CENTER]
            )

        # Add match age
        self.__dict__[cn.R_MATCH_AGE] = round_to_decimals(
            (match_date - patient.__dict__[cn.R_DOB]) /
            es.DAYS_PER_YEAR,
            2
        )

        # Add international match meld (for naive sorting)
        self.__dict__[cn.ALLOC_SCORE_INT] = patient.meld_int_match

        self._match_tuple = None

    @property
    def match_tuple(self):
        if not self._match_tuple:
            self._match_tuple = self.return_match_tuple()
        return self._match_tuple

    def _add_obligations(
            self,
            obl_ranks: Dict[str, int],
            rec_country: str,
            rec_center: str
            ):
        """Add obligation info"""
        if not obl_ranks:
            self.__dict__[cn.N_OPEN_OBL] = 0
            self.__dict__[cn.ANY_OBL] = False
            return
        if rec_country in CNTR_OBLIGATION_CNTRIES:
            if rec_center in es.AUSTRIAN_CENTER_GROUPS:
                obl_creditor = es.AUSTRIAN_CENTER_GROUPS[
                        rec_center
                ]
            else:
                obl_creditor = rec_center
        else:
            obl_creditor = rec_country
        self.__dict__[cn.OBL_CREDITOR] = obl_creditor
        if obl_creditor in obl_ranks:
            n_open_obl = obl_ranks[obl_creditor]
            self.__dict__[cn.N_OPEN_OBL] = n_open_obl
            self.__dict__[cn.ANY_OBL] = n_open_obl > 0
        else:
            self.__dict__[cn.N_OPEN_OBL] = 0
            self.__dict__[cn.ANY_OBL] = False

    def return_match_info(
        self, cols: Optional[Tuple[str, ...]] = None
    ) -> Dict[str, Any]:
        """Return relevant match information"""
        if cols is None:
            cols = es.MATCH_INFO_COLS
        result_dict = {}

        for key in cols:
            if key in self.__dict__:
                result_dict[key] = self.__dict__[key]
            elif key in self.donor.__dict__:
                result_dict[key] = self.donor.__dict__[key]
            elif key in self.patient.__dict__:
                result_dict[key] = self.patient.__dict__[key]
            elif key in self.patient.meld_scores:
                result_dict[key] = self.patient.meld_scores[key]
            elif key in self.patient.biomarkers:
                result_dict[key] = self.patient.biomarkers[key]

        return result_dict

    def return_match_tuple(self):
        """Return a match tuple"""
        return tuple(
                self.__dict__[attr] for attr in self.attr_order_match
        )

    def add_patient_rank(self, rnk: int) -> None:
        """
        Add patient rank in current sorted list,
        and add it as tie-breaker.
        """
        self.__dict__[cn.PATIENT_RANK] = int(rnk+1)
        self.attr_order_match += (cn.PATIENT_RANK, )

    def _calculate_truncated_dri(self) -> None:
        lp = 0
        for var, map in es.ET_DRI_DICT.items():
            value = self.__dict__[var]
            for contrib, cond in map.items():
                if cond(value):
                    lp += contrib
                    break
            else:
                raise ValueError(
                    f'{value} does not yield a contrib for {var}'
                )
        self.__dict__[cn.D_ET_DRI_TRUNC] = round_to_decimals(
            np.exp(lp),
            2
        )

    def set_acceptance(self, reason: str):
        if reason in cg.ACCEPTANCE_CODES:
            self.__dict__[cn.ACCEPTANCE_REASON] = reason
            if reason == cn.T1 or reason == cn.T3:
                self.__dict__[cn.ACCEPTED] = 1
            else:
                self.__dict__[cn.ACCEPTED] = 0
        else:
            raise ValueError(
                f'{reason} is not a valid acceptance reason.'
            )

    def __repr__(self):
        return (
            f'{self.__dict__[cn.OFFERED_TO]} offer to '
            f'{self.__dict__[cn.ID_RECIPIENT] } '
            f'({self.__dict__[cn.RECIPIENT_CENTER]}) '
            f'from {self.__dict__[cn.D_ALLOC_CENTER] }'
        )

    def __str__(self):
        """Match record"""
        return(
            f'{self.__dict__[cn.OFFERED_TO]} offer to '
            f'{self.__dict__[cn.ID_RECIPIENT] } '
            f'({self.__dict__[cn.RECIPIENT_CENTER]}) '
            f'from {self.__dict__[cn.D_ALLOC_CENTER] }'
        )

    def __lt__(self, other):
        """Order by match tuple."""
        return self.match_tuple > other.match_tuple


class MatchList:
    """Class which implements an MatchList
    ...

    Attributes   #noqa
    ----------
    donor: Donor
        Donor that is on offer
    match_date: datetime
        Date that match list is generated.
    match_list: List[Union[MatchRecordCurrentELAS, CenterOffer]]
        Match list, consisting of list of match records

    Methods
    -------

    """

    _ids = count(0)

    def __init__(
            self, patients: Generator[Patient, None, None], donor: Donor,
            match_date: datetime,
            obl: CountryObligations,
            type_offer: int = 4,
            alloc_center: Optional[str] = None,
            record_class: Type[MatchRecord] = MatchRecord,
            sort: bool = True,
            sim_start_date: Optional[datetime] = None,
            travel_time_dict: Optional[
                Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, str]]]
            ] = None,
            attr_order_match: Optional[List[str]] = None
            ) -> None:

        if sim_start_date is None:
            sim_start_date = es.INTRODUCTION_MELD

        self.__dict__[cn.MATCH_DATE] = match_date
        self.match_time = (
            (match_date-sim_start_date) /
            timedelta(days=1)
        )

        self.id_mtr = next(self._ids)
        self.donor = donor
        if not isinstance(match_date, datetime):
            raise TypeError(
                f'match_date must be datetime,'
                f'not a {type(match_date)}'
                )

        alloc_center = (
            alloc_center if alloc_center is not None
            else donor.donor_center
        )
        don_center_offer_group = CENTER_OFFER_GROUPS.get(
            alloc_center,
            alloc_center
        )
        alloc_region = DICT_CENTERS_TO_REGIONS.get(
            alloc_center,
            None
            )
        alloc_country = FL_CENTER_CODES[alloc_center[0]]
        self.__dict__[cn.D_ALLOC_COUNTRY] = alloc_country

        if (alloc_country == mgr.GERMANY and alloc_region is None):
            raise Exception(
                'No region defined for German center: {alloc_center}'
            )

        debtor = (
            alloc_center if alloc_country in CNTR_OBLIGATION_CNTRIES
            else alloc_country
        )

        if debtor in es.OBLIGATION_PARTIES:
            curr_obl_ranks = obl.return_obligation_ranks(
                        bloodgroup=donor.d_bloodgroup,
                        debtor_party=debtor,
                        count_all=False
                    )
        else:
            curr_obl_ranks = {}

        if travel_time_dict:
            dhospital_travel_time_dict = travel_time_dict[
                donor.__dict__[cn.D_HOSPITAL]
            ]
        else:
            dhospital_travel_time_dict = None

        self.match_list = [
                record_class(
                    patient=pat, donor=donor,
                    match_date=match_date,
                    type_offer_detailed=type_offer,
                    alloc_center=alloc_center,
                    alloc_region=alloc_region,
                    alloc_country=alloc_country,
                    don_center_offer_group=don_center_offer_group,
                    debtor=debtor,
                    obl_ranks=curr_obl_ranks,
                    match_time=self.match_time,
                    dhospital_travel_time_dict=dhospital_travel_time_dict,
                    id_mtr=self.id_mtr,
                    attr_order_match=attr_order_match
                ) for pat in patients
            ]

        if sort:
            self.sorted = True
            self.match_list.sort()
        else:
            self.sorted = False

        # Type offer
        self.__dict__[cn.TYPE_OFFER_DETAILED] = type_offer

    def is_empty(self) -> bool:
        """Check if event is empty"""
        return len(self.match_list) == 0

    def return_match_list(
            self
    ) -> List[MatchRecord]:
        if self.sorted:
            return [m for m in self.match_list]
        else:
            raise Exception("Cannot return a match list that is not sorted!")

    def return_match_info(
        self
    ) -> List[Dict[str, Any]]:
        """Return match lists"""
        return [
            matchr.return_match_info() for matchr in self.match_list
            ]

    def return_match_df(self) -> Optional[pd.DataFrame]:
        """Print match list as DataFrame"""
        if self.match_list is not None:
            return pd.DataFrame.from_records(
                [mr.return_match_info() for mr in self.match_list],
                columns=es.MATCH_INFO_COLS
            )

    def print_match_list(self) -> None:
        """Print match list as DataFrame"""
        print(self.return_match_df())

    def __str__(self) -> str:
        string = ''
        for evnt in sorted(self.match_list):
            string += str(evnt) + '\n'
        return string

    def __repr__(self):
        """Print the match list"""
        string = ''
        for evnt in sorted(self.match_list):
            string += str(evnt) + '\n'
        return string

    def __len__(self):
        return len(self.match_list)


class CenterOffer:
    """Class which implements a center offer
    ...

    Attributes   #noqa
    ----------
    eligible_patients: List[MatchRecord]
        List of patients on the MatchList
    rank: int
        Rank of the center
    offer_center: str
        Center which makes the offer
    center: str
        Center to which the offer is made

    Methods
    -------

    """

    def __init__(
        self,
        type_center_offer,
        records: (
            'List[Union[MatchRecord, CurrentELAS.MatchRecordCurrentELAS]]'
        ),
        attr_order_match: Optional[List[str]] = None
    ) -> None:
        self.eligible_patients = records
        self.n_profile_eligible = sum(
            r.patient.profile._check_acceptable(r.donor)
            if r.patient.profile else False
            for r in records
        )
        self.__dict__[cn.PROFILE_COMPATIBLE] = int(
            self.n_profile_eligible > 0
        )
        self.__dict__[cn.N_OFFERS] = len(records)
        self.__dict__[cn.OFFERED_TO] = type_center_offer
        self.rank = min(r.__dict__[cn.PATIENT_RANK] for r in records)

        if attr_order_match is None:
            self.attr_order_match = records[0].attr_order_match

        # Rank items based on the first patient
        self.return_match_tuple = records[0].return_match_tuple
        for key in self.attr_order_match:
            self.__dict__[key] = records[0].__dict__[key]

        for key in es.CENTER_OFFER_INHERIT_COLS:
            if key in records[0].__dict__:
                self.__dict__[key] = records[0].__dict__[key]

        # Sort eligible patients
        self.eligible_patients.sort()

        # Fixed attributes to copy over from first match record.
        self.donor = self.eligible_patients[0].donor

        self._match_tuple = None

    def __str__(self):
        return(
            f'{self.__dict__[cn.OFFERED_TO]} to '
            f'{self.__dict__[cn.REC_CENTER_OFFER_GROUP] } '
            f'from {self.__dict__[cn.D_ALLOC_CENTER] } with '
            f'{len(self.eligible_patients)} eligible patients.'
        )

    def __repr__(self):
        return(
            f'{self.__dict__[cn.OFFERED_TO]} to '
            f'{self.__dict__[cn.REC_CENTER_OFFER_GROUP] } '
            f'from {self.__dict__[cn.D_ALLOC_CENTER] } with '
            f'{len(self.eligible_patients)} eligible patients.'
        )

    def _add_co_rank(self, rnk: int) -> None:
        """Add center offer ranks, if it is a center offer."""
        self.__dict__[cn.CENTER_RANK] = int(rnk+1)

    def _initialize_acceptance_information(self) -> None:
        # Initialize travel times
        # Set travel time dictionary
        if self.eligible_patients[0].travel_time_dict:
            self.__dict__.update(
                self.eligible_patients[0].travel_time_dict[
                    self.eligible_patients[0].__dict__[cn.RECIPIENT_CENTER]
                ]
            )

        if hasattr(
            self.eligible_patients[0],
            '_initialize_acceptance_information'
        ):
            for pat in self.eligible_patients:
                pat._initialize_acceptance_information()
        for attr in cg.CENTER_OFFER_ACCEPTANCE_COLS:
            self.__dict__[attr] = self.eligible_patients[0].__dict__[attr]

    @property
    def match_tuple(self):
        if not self._match_tuple:
            self._match_tuple = self.return_match_tuple()
        return self._match_tuple

    def return_match_tuple(self):
        """Return a match tuple"""
        return tuple(
                self.__dict__[attr] for attr in self.attr_order_match
            )

    def return_match_info(
        self, cols: Optional[Tuple[str, ...]] = None
    ) -> Dict[str, Any]:
        """Return relevant match information"""
        if cols is None:
            cols = es.MATCH_INFO_COLS
        result_dict = {}
        for key in cols:
            if (value := self.__dict__.get(key)) is not None:
                result_dict[key] = value
            elif (value := self.donor.__dict__.get(key)) is not None:
                result_dict[key] = value
        return result_dict

    def set_acceptance(self, reason: str):
        if reason in cg.ACCEPTANCE_CODES:
            self.__dict__[cn.ACCEPTANCE_REASON] = reason
            if reason == cn.T1 or reason == cn.T3:
                self.__dict__[cn.ACCEPTED] = 1
            else:
                self.__dict__[cn.ACCEPTED] = 0
        else:
            raise ValueError(
                f'{reason} is not a valid acceptance reason.'
            )

    def _determine_rescue_priority(self):
        mr = self.eligible_patients[0]
        if hasattr(mr, '_determine_rescue_priority'):
            mr._determine_rescue_priority()
            self.__dict__[cn.RESCUE_PRIORITY] = (
                mr.__dict__[cn.RESCUE_PRIORITY]
            )
        else:
            raise Exception(
                "Could not determine rescue priority for center offer"
            )

    def __lt__(self, other):
        """Order by match tuple."""
        return self.match_tuple > other.match_tuple
