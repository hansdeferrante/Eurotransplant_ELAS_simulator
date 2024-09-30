from typing import List, Callable, Any, Tuple, Dict, Optional, Union, Generator
from math import isnan
import pandas as pd
import numpy as np

import simulator.magic_values.column_names as cn
import simulator.magic_values.column_groups as cg
import simulator.magic_values.elass_settings as es
import simulator.magic_values.magic_values_rules as mgr
from simulator.code.PostTransplantPredictor import PostTransplantPredictor
from simulator.code.AllocationSystem import MatchList, MatchRecord, CenterOffer
from simulator.code.entities import Patient
from simulator.code.utils import RuleTuple, round_to_decimals
from simulator.magic_values.rules import (
    BG_COMPATIBILITY_DEFAULTS,
    BG_PRIORITY_RULES,
    CENTER_OFFER_GROUPS,
    DEFAULT_BG_PRIORITY,
    MAIN_RANK_TIERS,
    MAIN_TIER_REVERSAL_DICTS,
    BLOOD_GROUP_COMPATIBILITY_DICT,
    RECIPIENT_ELIGIBILITY_TABLES, DEFAULT_BG_TAB_COLLE,
    BG_COMPATIBILITY_TAB, DEFAULT_RANK_TIER,
    FIRST_LAYER_TIERS,
    FOURTH_MRL, RULES_LOCAL_CENTER_OFFER,
    RULES_INT_CENTER_OFFER,
    CENTER_OFFER_TIERS,
    OBL_TIER_APPLICABLE,
    check_rec_low_weight,
    check_ELAS_ped_don,
    check_ELAS_ped_rec,
    DEFAULT_MELD_ETCOMP_THRESH
    )


class MatchRecordCurrentELAS(MatchRecord):
    """Class which implements an match record for current ELAS
    ...

    Attributes   #noqa
    ----------
    patient: Patient
        Patient
    donor: Donor
        Donor info
    match_date: datetime
        Date that match list is generated.
    """
    check_d_pediatric: Callable = check_ELAS_ped_don
    check_r_pediatric: Callable = check_ELAS_ped_rec
    check_r_lowweight: Callable = check_rec_low_weight

    et_comp_thresh = None

    def __init__(
            self,
            patient: Patient,
            *args,
            **kwargs
    ) -> None:

        # Construct general match records for the patient.
        super(MatchRecordCurrentELAS, self).__init__(
            patient=patient, *args, **kwargs
            )

        # Construct columns for allocation scores
        self.__dict__[cn.MELD_NAT_MATCH] = patient.meld_nat_match
        self.__dict__[cn.MELD_INT_MATCH] = patient.meld_int_match

        # Construct whether MELD is greater than threshold for
        # extra prioritization
        self.__dict__[cn.MELD_GE_ETTHRESH] = (
            self.__dict__[cn.MELD_NAT_MATCH] >=
            MatchRecordCurrentELAS.et_comp_thresh
        )

        # Determine whether patient is pediatric.
        self._determine_pediatric()
        self._determine_match_criterium()

        # For local Croatian/Hungarian offers record match criterium
        # as national. # These are also "national" for the center
        # offer model.
        if (
            self.__dict__[cn.ALLOCATION_LOC] and
            self.__dict__[cn.RECIPIENT_COUNTRY] in {mgr.CROATIA, mgr.HUNGARY}
        ):
            self.__dict__[cn.MATCH_CRITERIUM] = cn.NAT

        # Add BG information
        self._add_bg_information()

        # Add match rank information only if patient is BG compatible.
        if self.__dict__[cn.BG_COMP]:

            # Determine BG priority.
            self._check_zipped_rules(
                rule_dict=BG_PRIORITY_RULES,
                default=DEFAULT_BG_PRIORITY,
                attr_name=cn.BG_PRIORITY
            )

            # Determine national & international allocation scores
            self.__dict__[cn.ALLOC_SCORE_NAT] = patient.return_nat_alloc_score(
                        don=self.donor
                    )
            self.__dict__[cn.ALLOC_SCORE_INT] = patient.return_int_alloc_score(
                don=self.donor
            )

            self._add_match_ranks(
                d_alloc_country=self.__dict__[cn.D_ALLOC_COUNTRY],
                type_offer=self.__dict__[cn.TYPE_OFFER]
            )
            # Add whether a local center offer ought to be made,
            #  and determine ranks.
            self._add_center_offer_elig()
            # Clean the match rank (i.e. reverse the letters)
            self._clean_match_rank()

            self.initialized_acceptance = False
            self.initialized_posttxp = False

            # Donor and recipient info needed for acceptance &
            # simulation results
            if (
                value := patient.meld_scores[patient.sim_set.LAB_MELD]
            ) is not None:
                self.__dict__[cn.MELD_LAB] = value
            else:
                self.__dict__[cn.MELD_LAB] = patient.sim_set.DOWNMARKED_MELD
        else:
            self.__dict__[cn.REC_CENTER_OFFER_GROUP] = CENTER_OFFER_GROUPS.get(
                patient.__dict__[cn.RECIPIENT_CENTER],
                patient.__dict__[cn.RECIPIENT_CENTER]
            )
            self.__dict__[cn.DON_CENTER_OFFER_GROUP] = CENTER_OFFER_GROUPS.get(
                self.__dict__[cn.D_ALLOC_CENTER],
                self.__dict__[cn.D_ALLOC_CENTER]
            )

        self._match_tuple = None

    def _initialize_acceptance_information(self) -> None:

        # Initialize travel times
        if self.travel_time_dict:
            self.__dict__.update(
                self.travel_time_dict[
                    self.__dict__[cn.RECIPIENT_CENTER]
                ]
            )

        # Copy over patient and donor information, needed for acceptance.
        self.__dict__.update(self.donor.offer_inherit_cols)
        self.__dict__.update(self.patient.offer_inherit_cols)

        # Information copied over manually
        self.__dict__[cn.D_MALIGNANCY] = self.donor.malignancy
        self.__dict__[cn.D_DRUG_ABUSE] = self.donor.drug_abuse
        self.__dict__[cn.D_MARGINAL] = self.donor.marginal
        self.__dict__[cn.LOCAL_MATCH_CROATIA] = (
            self.__dict__[cn.RECIPIENT_COUNTRY] == mgr.CROATIA and
            self.__dict__[cn.GEOGRAPHY_MATCH] == 'L'
        )
        self.ratio_weight = self.donor.d_weight / max(self.patient.r_weight, 1)
        self.delta_height = self.donor.d_height - self.patient.r_height

        # Donor and recipient info needed for post-transplant survival
        if self.__dict__[cn.TYPE_OFFER_DETAILED] == es.TYPE_OFFER_WLIV:
            self.__dict__[cn.DELTA_WEIGHT_NOSPLIT_MA30] = min(
                max(
                    (
                        self.donor.__dict__[cn.D_WEIGHT] -
                        self.__dict__[cn.R_WEIGHT]
                    ),
                    -30
                ),
                30
            )
        else:
            self.__dict__[cn.DELTA_WEIGHT_NOSPLIT_MA30] = 0

        self.__dict__[cn.TX_BLOODGROUP_MATCH] = int(
            self.__dict__[cn.R_BLOODGROUP] == self.__dict__[cn.D_BLOODGROUP]
        )

        # Determine match abroad (part of acceptance)
        self._determine_match_abroad()

        if (
            self.__dict__[cn.URGENCY_CODE] == 'HU' or
            self.__dict__[cn.R_ACO]
        ):
            if self.__dict__[cn.R_ACO] == 1:
                if self.__dict__[cn.MATCH_ABROAD] == 'A':
                    self.__dict__[cn.INTERREGIONAL_ACO] = 1
                elif (
                    self.__dict__[cn.MATCH_ABROAD] == 'H' and
                    self.__dict__[cn.D_ALLOC_COUNTRY] == mgr.GERMANY
                ):
                    self.__dict__[cn.INTERREGIONAL_ACO] = 1
                else:
                    self.__dict__[cn.INTERREGIONAL_ACO] = 0
            else:
                self.__dict__[cn.INTERREGIONAL_ACO] = 0

        if self.patient.profile:
            self.__dict__[cn.PROFILE_COMPATIBLE] = int(
                self.patient.profile._check_acceptable(
                    self.donor
                )
            )
        elif cn.PROFILE_COMPATIBLE not in self.__dict__.keys():
            self.__dict__[cn.PROFILE_COMPATIBLE] = 0

    def _initialize_posttxp_information(
        self, ptp: PostTransplantPredictor
    ) -> None:

        # Copy over biomarkers
        for col in cg.BIOMARKERS:
            self.__dict__[col] = self.patient.biomarkers[col]
        self.__dict__[cn.CREA_NODIAL] = (
            1 if self.__dict__[cn.DIAL_BIWEEKLY] > 0
            else self.__dict__[cn.CREA]
        )

        # Date relative to 2014
        self.__dict__[cn.YEAR_TXP_RT2014] = (
            self.__dict__[cn.MATCH_DATE].year - 2014
        )

        # Time since previous transplant
        if self.patient.__dict__[cn.TIME_SINCE_PREV_TXP] is None:
            self.__dict__[cn.TIME_SINCE_PREV_TXP_CAT] = 'none'
        elif self.patient.__dict__[cn.TIME_SINCE_PREV_TXP] < 14:
            self.__dict__[cn.TIME_SINCE_PREV_TXP_CAT] = '2weeks'
        elif self.patient.__dict__[cn.TIME_SINCE_PREV_TXP] < 90:
            self.__dict__[cn.TIME_SINCE_PREV_TXP_CAT] = '2weeks90days'
        elif self.patient.__dict__[cn.TIME_SINCE_PREV_TXP] < 365:
            self.__dict__[cn.TIME_SINCE_PREV_TXP_CAT] = '90days1year'
        elif self.patient.__dict__[cn.TIME_SINCE_PREV_TXP] < 3*365:
            self.__dict__[cn.TIME_SINCE_PREV_TXP_CAT] = '1to3years'
        else:
            self.__dict__[cn.TIME_SINCE_PREV_TXP_CAT] = 'over3years'

        self._calculate_posttxp_survival(ptp=ptp)

    def _calculate_posttxp_survival(
        self, ptp: PostTransplantPredictor
    ) -> None:
        for window in es.WINDOWS_TRANSPLANT_PROBS:
            self.__dict__[f'{es.PREFIX_SURVIVAL}_{window}'] = (
                round_to_decimals(
                    ptp.calculate_survival(
                        offer=self,
                        time=window,
                        split=int(
                            self.__dict__[cn.TYPE_TRANSPLANTED] !=
                            es.TYPE_OFFER_WLIV
                        )
                    ),
                    3
                )
            )

    def _determine_pediatric(
            self
            ):
        """Determine whether donor/patient are pediatric.
        """

        pat_dict = self.patient.__dict__

        if pat_dict['_pediatric'] or pat_dict['_pediatric'] is None:
            self.__dict__[cn.R_PED] = self.patient.check_pediatric(
                ped_fun=MatchRecordCurrentELAS.check_r_pediatric,
                match_age=self.__dict__[cn.R_MATCH_AGE]
            )
        else:
            self.__dict__[cn.R_PED] = pat_dict['_pediatric']

        if pat_dict[cn.R_LOWWEIGHT] is None:
            self.__dict__[cn.R_LOWWEIGHT] = self.patient.check_lowweight(
                lowweight_fun=MatchRecordCurrentELAS.check_r_lowweight
            )
        else:
            self.__dict__[cn.R_LOWWEIGHT] = pat_dict[cn.R_LOWWEIGHT]

        if self.donor.__dict__[cn.D_PED] is None:
            self.__dict__[cn.D_PED] = self.donor.check_pediatric(
                ped_fun=MatchRecordCurrentELAS.check_d_pediatric
            )
        else:
            self.__dict__[cn.D_PED] = self.donor.__dict__[cn.D_PED]

    def _determine_match_criterium(self):
        """Determine match criterium for allocation"""
        self_dict = self.__dict__
        pat_dict = self.patient.__dict__

        if (
            pat_dict[cn.RECIPIENT_COUNTRY] !=
            self_dict[cn.D_ALLOC_COUNTRY]
        ):
            self_dict[cn.MATCH_CRITERIUM] = cn.INT
            self_dict[cn.GEOGRAPHY_MATCH] = cn.A
            self_dict[cn.ALLOCATION_LOC] = False
            self_dict[cn.ALLOCATION_REG] = False
            self_dict[cn.ALLOCATION_NAT] = False
            self_dict[cn.ALLOCATION_INT] = True
        else:
            if (
                pat_dict[cn.RECIPIENT_CENTER] ==
                self_dict[cn.D_ALLOC_CENTER]
            ):
                self_dict[cn.MATCH_CRITERIUM] = cn.LOC
                self_dict[cn.ALLOCATION_LOC] = True
                if self_dict[cn.D_ALLOC_COUNTRY] == mgr.GERMANY:
                    self_dict[cn.ALLOCATION_REG] = True
                else:
                    self_dict[cn.ALLOCATION_REG] = False
                self_dict[cn.ALLOCATION_NAT] = True
                self_dict[cn.ALLOCATION_INT] = False
            elif (
                pat_dict[cn.RECIPIENT_REGION] ==
                self_dict[cn.D_ALLOC_REGION]
            ):
                self_dict[cn.MATCH_CRITERIUM] = cn.REG
                self_dict[cn.ALLOCATION_LOC] = False
                self_dict[cn.ALLOCATION_REG] = True
                self_dict[cn.ALLOCATION_NAT] = True
                self_dict[cn.ALLOCATION_INT] = False
            elif (
                pat_dict[cn.RECIPIENT_COUNTRY] ==
                self_dict[cn.D_ALLOC_COUNTRY]
            ):
                self_dict[cn.MATCH_CRITERIUM] = cn.NAT
                self_dict[cn.ALLOCATION_LOC] = False
                self_dict[cn.ALLOCATION_REG] = False
                self_dict[cn.ALLOCATION_NAT] = True
                self_dict[cn.ALLOCATION_INT] = False
            if (
                pat_dict[cn.RECIPIENT_CENTER] ==
                self_dict[cn.D_PROC_CENTER]
            ):
                self_dict[cn.GEOGRAPHY_MATCH] = cn.L
            elif self_dict[cn.ALLOCATION_REG]:
                self_dict[cn.GEOGRAPHY_MATCH] = cn.R
            elif self_dict[cn.ALLOCATION_NAT]:
                self_dict[cn.GEOGRAPHY_MATCH] = cn.H

    def _determine_rescue_priority(self):
        """Determine which centers have local priority in rescue"""
        if (
            self.__dict__[cn.ALLOCATION_REG] and
            self.__dict__[cn.RECIPIENT_COUNTRY] == mgr.GERMANY
        ):
            self.__dict__[cn.RESCUE_PRIORITY] = 1
        elif (
            self.__dict__[cn.RECIPIENT_COUNTRY] == mgr.BELGIUM
            and self.__dict__[cn.ALLOCATION_LOC]
        ):
            self.__dict__[cn.RESCUE_PRIORITY] = 1
        else:
            self.__dict__[cn.RESCUE_PRIORITY] = 0

    def _determine_match_abroad(self):
        """Determine match abroad"""

        if self.__dict__[cn.GEOGRAPHY_MATCH] == 'L':
            if self.__dict__[cn.RECIPIENT_COUNTRY] == mgr.GERMANY:
                self.__dict__[cn.MATCH_ABROAD] = 'R'
            else:
                self.__dict__[cn.MATCH_ABROAD] = 'H'
        elif self.__dict__[cn.GEOGRAPHY_MATCH] == 'R':
            if self.__dict__[cn.RECIPIENT_COUNTRY] != mgr.GERMANY:
                self.__dict__[cn.MATCH_ABROAD] = 'H'
            else:
                self.__dict__[cn.MATCH_ABROAD] = 'R'
        else:
            self.__dict__[cn.MATCH_ABROAD] = self.__dict__[cn.GEOGRAPHY_MATCH]

    def _add_bg_information(self):
        """Add which blood group table applies to the patient
        """

        d_bg = self.__dict__[cn.D_BLOODGROUP]
        r_bg = self.__dict__[cn.R_BLOODGROUP]

        # (1) retrieve which blood group table to consult
        bg_tab = self._check_zipped_rules(
            attr_name=cn.BG_TAB_COL,
            rule_dict=RECIPIENT_ELIGIBILITY_TABLES,
            default=DEFAULT_BG_TAB_COLLE,
            value_required=True
        )

        # (2) retrieve BG compatibility defs. to maintain
        bg_rule = self._check_zipped_rules(
            attr_name=cn.BG_RULE_COL,
            rule_dict=BG_COMPATIBILITY_TAB[bg_tab],
            default=BG_COMPATIBILITY_DEFAULTS[bg_tab],
            value_required=True
        )

        # Also add bg compatibility columns. Necessary to construct BG
        # priority groups.
        for comp_def, comp_dict in BLOOD_GROUP_COMPATIBILITY_DICT.items():
            if d_bg == r_bg:
                self.__dict__[cn.BG_IDENTICAL] = True
                self.__dict__[comp_def] = True
            else:
                self.__dict__[cn.BG_IDENTICAL] = False
                self.__dict__[comp_def] = (
                    r_bg in comp_dict[d_bg]
                )

        # (3) Add whether patient is BG compatible according to rules
        self.__dict__[cn.BG_COMP] = r_bg in (
            BLOOD_GROUP_COMPATIBILITY_DICT[bg_rule][d_bg]
        )

    def _add_center_offer_elig(self):
        """Initialize center offer (eligibility) information"""

        # Determine whether the tier is eligible for center offers
        center_offer_eligible = False
        for key, conditions in CENTER_OFFER_TIERS[
                self.__dict__[cn.TYPE_OFFER]
        ]:
            for attr, value in conditions:
                if self.__dict__[attr] != value:
                    break
            else:
                center_offer_eligible = key
                break

        # Initialize to False.
        if center_offer_eligible:
            # Add local center offers.
            if (
                    self.__dict__[cn.DON_CENTER_OFFER_GROUP] ==
                    self.__dict__[cn.REC_CENTER_OFFER_GROUP]
            ):
                for type_co, co_dict in RULES_LOCAL_CENTER_OFFER.items():
                    if (self.__dict__[cn.D_ALLOC_COUNTRY] in co_dict):
                        self._check_zipped_rules(
                            attr_name=type_co,
                            rule_dict=co_dict[
                                self.__dict__[cn.D_ALLOC_COUNTRY]
                                ],
                            default=False
                        )
                    else:
                        self.__dict__[type_co] = False
            else:
                for k in RULES_LOCAL_CENTER_OFFER.keys():
                    self.__dict__[k] = False

            # Add non-local center offers
            for type_co, co_dict in RULES_INT_CENTER_OFFER.items():
                self._check_zipped_rules(
                        attr_name=type_co,
                        rule_dict=co_dict,
                        default=False
                    )
        else:
            for k in RULES_LOCAL_CENTER_OFFER.keys():
                self.__dict__[k] = False
            for k in RULES_INT_CENTER_OFFER.keys():
                self.__dict__[k] = False

    def _add_match_ranks(self, d_alloc_country: str, type_offer: str) -> None:

        # Determine match rank tier (A-G)
        mtch_tier = self._check_zipped_rules(
            attr_name=cn.MTCH_TIER,
            rule_dict=MAIN_RANK_TIERS[type_offer],
            default=DEFAULT_RANK_TIER[type_offer],
            value_required=True
        )

        # Determine match layer (1-26)
        self._check_zipped_rules(
            attr_name=cn.MTCH_LAYER,
            rule_dict=(
                FIRST_LAYER_TIERS[type_offer]
                [mtch_tier]
                [d_alloc_country]
            ),
            default=0
        )

        # Determine match layer for obligations
        obl_applicable = self._check_zipped_rules(
            attr_name=cn.OBL_APPLICABLE,
            rule_dict=OBL_TIER_APPLICABLE[d_alloc_country],
            default=False,
            value_required=True
        )
        mtch_obl = (
            self.__dict__[cn.N_OPEN_OBL] if
            obl_applicable else
            0
        )
        self.__dict__[cn.ANY_OBL] = int(mtch_obl > 0)
        self.__dict__[cn.MTCH_OBL] = mtch_obl

        # Add layer for allocation score.
        if mtch_tier in {cn.TIER_A, cn.TIER_B}:
            self.__dict__[cn.MTCH_LAYER_MELD] = int(0)
        elif mtch_obl > 0:
            self.__dict__[cn.MTCH_LAYER_MELD] = \
                self.__dict__[cn.ALLOC_SCORE_NAT]
        elif self.__dict__[cn.ALLOCATION_INT]:
            self.__dict__[cn.MTCH_LAYER_MELD] = \
                self.__dict__[cn.ALLOC_SCORE_INT]
        else:
            self.__dict__[cn.MTCH_LAYER_MELD] = \
                self.__dict__[cn.ALLOC_SCORE_NAT]

        # Determine fourth MRL (local in Germany only)
        for conditions in FOURTH_MRL:
            for attr, value in conditions:
                if self.__dict__[attr] != value:
                    break
            else:
                self.__dict__[cn.MTCH_LAYER_REG] = int(1)
                break
        else:
            self.__dict__[cn.MTCH_LAYER_REG] = int(0)

        # Determine waiting time (tie-breaker)
        if self.__dict__[cn.ALLOCATION_NAT]:
            self.__dict__[cn.MTCH_LAYER_WT] = (
                self.patient.get_accrued_waiting_time(
                    type_meld=cn.MELD_NAT_MATCH,
                    current_cal_time=self.match_time
                )
            )
        else:
            self.__dict__[cn.MTCH_LAYER_WT] = (
                self.patient.get_accrued_waiting_time(
                    type_meld=cn.MELD_INT_MATCH,
                    current_cal_time=self.match_time
                )
            )

        self.__dict__[cn.MTCH_LAYER_WT] = int(
            self.__dict__[cn.MTCH_LAYER_WT]
        ) if not isnan(self.__dict__[cn.MTCH_LAYER_WT]) else \
            self.__dict__[cn.MTCH_LAYER_WT]

        # Add tie-breaker date.
        if self.__dict__[cn.MTCH_TIER] == cn.TIER_A:
            self.__dict__[cn.MTCH_DATE_TIEBREAK] = self.patient.hu_since
        elif self.__dict__[cn.MTCH_TIER] == cn.TIER_B:
            self.__dict__[cn.MTCH_DATE_TIEBREAK] = (
                self.patient.__dict__[cn.ACO_SINCE]
            )
        else:
            self.__dict__[cn.MTCH_DATE_TIEBREAK] = (
                self.patient.__dict__[cn.LISTING_DATE]
            )

    def _clean_match_rank(self):
        """Reverse the letters in the match rank"""
        self.__dict__[cn.MTCH_TIER] = MAIN_TIER_REVERSAL_DICTS[
            self.__dict__[cn.TYPE_OFFER]
        ][
            self.__dict__[cn.MTCH_TIER]
        ]

    def _check_zipped_rules(
            self, attr_name, rule_dict: RuleTuple,
            default=None, value_required: Optional[bool] = False
    ) -> Optional[Any]:
        obj_dict = self.__dict__
        for key, conditions in rule_dict:
            for attr, value in conditions:
                if obj_dict[attr] != value:
                    break
            else:
                obj_dict[attr_name] = key
                if value_required:
                    return key
                else:
                    break
        else:
            obj_dict[attr_name] = default
            if value_required:
                return default

    def _count_zipped_rules(
            self,
            attr_name,
            rule_dict: RuleTuple,
            counter: Dict[str, Dict[str, Dict[Tuple[str, Any], int]]]
            ) -> Dict[str, Dict[str, Dict[Tuple[str, Any], int]]]:
        """
        Checking whether the object fulfills the rules in the rule_dict.
        """
        counter[attr_name]['total_count'][('mr', 1)] += 1

        for key, conditions in rule_dict:
            for attr, value in conditions:
                if self.__dict__[attr] != value:
                    break
                else:
                    counter[attr_name][key][(attr, value)] += 1
        return counter

    @property
    def match_tuple(self):
        if not self._match_tuple:
            self._match_tuple = self.return_match_tuple()
        return self._match_tuple

    def __lt__(self, other):
        """Youngest obligation first (index then corresponds to age)."""
        return self.match_tuple > other.match_tuple


class MatchListCurrentELAS(MatchList):
    """Class implementing a match list for the Current ELAS system.
    ...

    Attributes   #noqa
    ----------
    match_list: List[Union[MatchRecordCurrentELAS, CenterOffer]]
        Match list

    Methods
    -------

    """
    def __init__(
            self,
            sort: bool = False,
            record_class=MatchRecordCurrentELAS,
            construct_center_offers: bool = True,
            aggregate_obligations: bool = False,
            et_comp_thresh: Optional[int] = None,
            *args,
            **kwargs
    ) -> None:
        record_class.et_comp_thresh = (
            et_comp_thresh if et_comp_thresh else DEFAULT_MELD_ETCOMP_THRESH
        )
        super(MatchListCurrentELAS, self).__init__(
            sort=sort,
            record_class=record_class,
            attr_order_match=cg.DEFAULT_ATTR_ORDER,
            *args,
            **kwargs
            )

        # Remove blood group incompatible donors from the match list,
        # and add patient ranks.
        self.match_list = [
                mtch for mtch in self.match_list
                if mtch.__dict__[cn.BG_COMP]
            ]
        self.match_list.sort()
        self.sorted = True

        for rank, match_record in enumerate(self.match_list):
            if isinstance(match_record, MatchRecordCurrentELAS):
                match_record.add_patient_rank(rank)

        # Replace patients that are in a center offer by the center offer.
        if construct_center_offers:
            self.match_list: List[
                Union[MatchRecordCurrentELAS, CenterOffer]
                ] = self.match_list
            self._add_center_offers()

        if aggregate_obligations:
            self._aggregate_obligation_offers()

        if construct_center_offers | aggregate_obligations:
            self.match_list.sort()
            for rank, match_record in enumerate(self.match_list):
                match_record.__dict__[cn.RANK] = int(rank+1)
                if isinstance(match_record, CenterOffer):
                    match_record._add_co_rank(rank)

    def _add_center_offers(
            self,
            co_cols: Tuple[str, ...] = (
                cg.PED_CENTER_OFFERS + cg.ADULT_CENTER_OFFERS
            )
            ):
        """Retrieve center ranks"""
        if self.match_list is not None:
            center_offers = {k: {} for k in co_cols}
            for co_col in co_cols:
                for match_record in self.match_list:
                    if match_record.__dict__[co_col]:
                        if (
                            match_record.__dict__[cn.REC_CENTER_OFFER_GROUP] in
                            center_offers[co_col]
                        ):
                            center_offers[co_col][
                                match_record.__dict__[
                                    cn.REC_CENTER_OFFER_GROUP
                                    ]
                                ].append(match_record)
                        else:
                            center_offers[
                                co_col
                                ][
                                    match_record.__dict__[
                                        cn.REC_CENTER_OFFER_GROUP
                                        ]
                                    ] = [match_record]

            for type_offer, co_dict in center_offers.items():
                for _, records in co_dict.items():
                    if len(records) > 0:
                        # Add center offer to the match list.
                        self.match_list.append(
                            CenterOffer(
                                type_center_offer=type_offer,
                                records=records
                            )
                        )
                        # Remove patients from the match list.
                        for rcrd in records:
                            if rcrd in self.match_list:
                                self.match_list.remove(
                                    rcrd
                                )

    def _initialize_rescue_priorities(self) -> None:
        for mr in self.match_list:
            mr._determine_rescue_priority()

    def _aggregate_obligation_offers(
            self
            ):
        """Aggregate obligation offers into a center offer."""
        if self.match_list is not None:
            obl_offers = {}
            for match_record in self.match_list:
                if not isinstance(match_record, CenterOffer):
                    if (
                        match_record.__dict__[cn.MTCH_OBL] > 0 &
                        match_record.__dict__[cn.OBL_APPLICABLE]
                    ):
                        if (
                            match_record.__dict__[cn.REC_CENTER_OFFER_GROUP] in
                            obl_offers
                        ):
                            obl_offers[
                                match_record.__dict__[
                                    cn.REC_CENTER_OFFER_GROUP
                                ]
                            ].append(match_record)
                        else:
                            obl_offers[
                                match_record.__dict__[
                                    cn.REC_CENTER_OFFER_GROUP
                                ]
                            ] = [match_record]

            for _, records in obl_offers.items():
                if len(records) > 0:
                    # Add center offer to the match list.
                    self.match_list.append(
                        CenterOffer(
                            type_center_offer='obl',
                            records=records
                        )
                    )
                    # Remove patients from the match list.
                    for rcrd in records:
                        if rcrd in self.match_list:
                            self.match_list.remove(
                                rcrd
                            )

    def return_match_list(
            self
    ) -> List[Union[MatchRecord, MatchRecordCurrentELAS, CenterOffer]]:
        return [m for m in self.match_list]

    def return_patient_ids(
        self
    ) -> List[int]:
        """Return matching patients"""
        patients = []
        for matchr in self.return_match_list():
            if isinstance(matchr, MatchRecordCurrentELAS):
                patients.append(matchr.patient.id_recipient)
            elif isinstance(matchr, CenterOffer):
                patients += [
                    pat.patient.id_recipient
                    for pat in matchr.eligible_patients
                ]
        return patients

    def return_match_info(
        self
    ) -> List[Dict[str, Any]]:
        """Return match lists"""
        return [
            matchr.return_match_info() for matchr in self.match_list
            ]

    def return_match_df(
            self,
            reorder: bool = True,
            collapse_mrl: bool = True
            ) -> Optional[pd.DataFrame]:
        """Print match list as DataFrame"""
        if self.match_list is not None:
            matchl = pd.DataFrame.from_records(
                [matchr.return_match_info() for matchr in self.match_list],
                columns=es.MATCH_INFO_COLS
            )
            matchl[cn.ID_DONOR] = matchl[cn.ID_DONOR].astype('Int64')
            matchl[cn.ID_RECIPIENT] = matchl[cn.ID_RECIPIENT].astype('Int64')

            if collapse_mrl:
                matchl[cn.MTCH_CODE] = (
                    matchl.loc[:, [col for col in cg.MTCH_COLS]].apply(
                        lambda row: "".join(
                            np.char.ljust(
                                row.values.astype(str),
                                width=3, fillchar='_'
                                )
                            ), axis=1
                            )
                )

            return matchl
