#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13

@author: H.C. de Ferrante
"""

from typing import Optional, Tuple, Dict, Union, List, Callable, Any, Generator
from math import isnan
from statistics import mean
from operator import attrgetter, and_
from collections import defaultdict
from functools import reduce
import numpy as np
import pandas as pd
from simulator.code.utils import round_to_decimals, round_to_int

import simulator.magic_values.column_names as cn
import simulator.magic_values.elass_settings as es
import simulator.magic_values.magic_values_rules as mgr
from simulator.code.functions import construct_piecewise_term
from simulator.code.current_elas.CurrentELAS import \
    MatchListCurrentELAS, MatchRecordCurrentELAS
from simulator.code.AllocationSystem import CenterOffer, \
    MatchRecord
from simulator.magic_values.rules import OBL_TIERS
from simulator.code.entities import Donor, Patient
from simulator.code.read_input_files import (
    read_rescue_probs, read_rescue_baseline_hazards
)

import numpy as np


def boolkey_to_str(x):
    if x is None:
        return None
    return str(int(x)) if isinstance(x, bool) else str(x)


def logit(prob: float) -> float:
    """Calculates logit for probability"""
    return np.log(prob) - np.log(1 - prob)


def inv_logit(logit_: float) -> float:
    """Calculates probability for logit"""
    return np.exp(logit_) / (1 + np.exp(logit_))


class AcceptanceModule:
    """ Class to implement an acceptance module,
        based on logistic regression.

    Attributes   #noqa
    ----------

    Methods
    -------
    check_profile_compatibility(
            patient: Patient, donor: Donor
        ) -> bool:
        Check whether patient profile is compatible with donor

    """
    def __init__(
        self,
        seed,
        patient_acc_policy: str,
        center_acc_policy: str,
        generate_splits: bool = True,
        dict_paths_coefs: Dict[str, str] = es.ACCEPTANCE_PATHS,
        path_coef_splits: str = es.PATH_COEFFICIENTS_SPLIT,
        separate_huaco_model: bool = True,
        separate_ped_model: bool = True,
        simulate_rescue: bool = False,
        path_rescue_probs: Optional[str] = None,
        center_offer_patient_method: str = 'prop',
        verbose: Optional[int] = None,
        simulate_random_effects: bool = True,
        simulate_joint_random_effects: Optional[bool] = False,
        re_variance_components: Optional[Dict[str, Dict[str, float]]] = None
    ):
        # Initialize random number generators
        rng = np.random.default_rng(seed=seed)
        seeds = rng.choice(999999, size=10, replace=False)
        self.rng_split = np.random.default_rng(seed=seeds[0])
        self.rng_center = np.random.default_rng(seed=seeds[1])
        self.rng_patient_per_center = np.random.default_rng(seed=seeds[2])
        self.rng_rescue = np.random.default_rng(seed=seeds[3])
        self.rng_random_eff = np.random.default_rng(seed=seeds[4])

        self.verbose = verbose if verbose else 0

        # Set patient acceptance policy.
        if patient_acc_policy.lower() == 'LR'.lower():
            self.determine_patient_acceptance = self._patient_accept_lr
        elif patient_acc_policy.lower() == 'Always'.lower():
            self.determine_patient_acceptance = self._patient_accept_always
        else:
            raise ValueError(
                f'Patient acceptance policy should be one of '
                f'{", ".join(es.PATIENT_ACCEPTANCE_POLICIES)}, '
                f'not {patient_acc_policy}'
            )

        # Set center acceptance policy.
        if center_acc_policy.lower() == 'LR'.lower():
            self.determine_center_acceptance = self._center_accept_lr
        elif center_acc_policy.lower() == 'Always'.lower():
            self.determine_center_acceptance = self._center_accept_always
        else:
            raise ValueError(
                f'Patient acceptance policy should be one of '
                f'{", ".join(es.CENTER_ACCEPTANCE_POLICIES)}, '
                f'not {center_acc_policy}'
            )

        # Set acceptance policy of patient within a center offer.
        if patient_acc_policy.lower() == 'LR'.lower():
            assert center_offer_patient_method in \
                ('prop', 'highest', 'prop_sq'), \
                f'Patient per center offer selection should be' \
                f'"prop" or "highest", not ' \
                f'{center_offer_patient_method}'
            if center_offer_patient_method == 'prop':
                self.determine_accepting_patient = (
                    self._prop_prob_acceptance
                )
            elif center_offer_patient_method == 'prop_sq':
                self.determine_accepting_patient = (
                    self._prop_sq_prob_acceptance
                )
            elif center_offer_patient_method == 'highest':
                self.determine_accepting_patient = (
                    self._highest_prob_acceptance
                )
        elif patient_acc_policy.lower() == 'Always'.lower():
            self.determine_accepting_patient = self._first_accepting_patient
        else:
            raise ValueError(
                f'Patient acceptance policy should be one of '
                f'{", ".join(es.PATIENT_ACCEPTANCE_POLICIES)}, '
                f'not {patient_acc_policy}'
            )

        # Simulate rescue
        self.simulate_rescue = simulate_rescue
        if simulate_rescue:
            if path_rescue_probs is None:
                path_rescue_probs = es.PATH_RESCUE_PROBABILITIES
                path_basehaz_rescueprobs = es.PATH_RESCUE_COX_BH
                path_coefs_rescueprobs = es.PATH_RESCUE_COX_COEFS

            self.rescue_bh, self.rescue_bh_stratavars = (
                read_rescue_baseline_hazards(path_basehaz_rescueprobs)
            )
            dict_paths_coefs.update(
                {'coxph_rescue': path_coefs_rescueprobs}
            )

            self.rescue_init_probs = read_rescue_probs(path_rescue_probs)

        # Initialize coefficients for the logistic regression
        if (
            (patient_acc_policy == 'LR') |
            (center_acc_policy == 'LR') |
            generate_splits
        ):
            self._initialize_lr_coefs(
                dict_paths_coefs=dict_paths_coefs,
                path_coef_split=path_coef_splits
            )

        # Save whether to use separate models for HU/ACO and regular
        if separate_huaco_model:
            if separate_ped_model:
                self.calculate_prob_patient_accept = (
                    self._calc_prob_accept_pedad_huaco
                )
            else:
                self.calculate_prob_patient_accept = (
                    self._calc_prob_accept_huaco
                )
        else:
            if separate_ped_model:
                print(
                    "Cannot simulate separate pediatric "
                    "model without separate HU/ACO model."
                )
                exit()
            self.calculate_prob_patient_accept = self._calc_prob_accept

        self.simulate_random_effects = simulate_random_effects
        self.simulate_joint_random_effects = simulate_joint_random_effects
        if self.simulate_random_effects:

            if re_variance_components is None:
                raise Exception(
                    f'Cannot simulate random effects, '
                    f'without variance components'
                )
            else:
                self.re_varcomps = re_variance_components

            self.joint_re_vars = set(
                reduce(
                    and_,
                    (sd.keys() for _, sd in self.re_varcomps.items())
                    )
                )
            if len(self.joint_re_vars) > 0:
                self.joint_res = {
                    joint_re: mean(
                        sd[joint_re]
                        for sd in self.re_varcomps.values()
                    )
                    for joint_re in self.joint_re_vars
                }
            else:
                self.joint_res = None

            self.realizations_random_effects = {
                k: defaultdict(dict)
                for k in self.re_varcomps.keys()
                if self.joint_res is None or k not in self.joint_res
            }
            self.realization_joint_random_effects = {
                k: defaultdict(dict)
                for k in self.joint_res
            }

    def _calculate_lp(
            self, item: Dict,
            which: str, verbose: Optional[int] = None,
            realization_intercept: Optional[float] = None
    ):
        # Realization of random intercept
        if realization_intercept:
            lp = realization_intercept
        else:
            lp = 0

        for key, fe_dict in self.fixed_effects[which].items():
            slogit_b4 = lp
            var2 = None
            sel_coefs = None
            if isinstance(fe_dict, dict):
                if es.REFERENCE in fe_dict:
                    var_slope = 0
                    for variable, subgroup in fe_dict.items():
                        if variable == es.REFERENCE:
                            var_slope += subgroup
                        else:
                            var_slope += fe_dict[variable].get(
                                str(item[variable]),
                                0
                            )
                    lp += var_slope * item[key]
                else:
                    # If it is a regular item, it is not a slope.
                    # Simply add the matching coefficient.
                    sel_coefs = fe_dict.get(
                        boolkey_to_str(item[key]),
                        None
                    )
                    if isinstance(sel_coefs, dict):
                        for var2, dict2 in sel_coefs.items():
                            if var2 == es.REFERENCE:
                                lp += dict2
                            else:
                                lp += dict2.get(
                                    boolkey_to_str(item[var2]),
                                    0
                                )
                                if (
                                    (verbose > 1) and
                                    dict2.get(boolkey_to_str(item[var2]), 0)
                                 ) != 0:
                                    print(
                                        f'{key}-{item[key]}:'
                                        f'{var2}-{item[var2]}: ',
                                        dict2.get(
                                            str(boolkey_to_str(item[var2])),
                                            0
                                        )
                                    )
                    elif sel_coefs is None:
                        newkey = (
                            str(int(item[key]))
                            if isinstance(item[key], bool)
                            else str(item[key])
                        )
                        if self.reference_levels[which][key] is None:
                            fe_dict[newkey] = 0
                            self.reference_levels[which][key] = newkey
                        else:
                            raise Exception(
                                f'Multiple reference levels for {key} '
                                f'(model: {which}):\n'
                                f'\t{self.reference_levels[which][key]}'
                                f'and {newkey}\n are both assumed reference'
                                f'levels.\n Existing keys are:'
                                f'\n{self.fixed_effects[which][key]}'
                            )
                    else:
                        lp += sel_coefs

            elif key == 'intercept':
                lp += fe_dict
            else:
                lp += item[key] * fe_dict

            if (
                (slogit_b4 != lp) & (verbose > 1) &
                (not isinstance(sel_coefs, dict))
            ):
                if key in item:
                    if var2:
                        print(
                            f'{key}-{item[key]}:'
                            f'{var2}-{item[var2]}: '
                            f'{lp-slogit_b4}'
                        )
                    else:
                        print(
                            f'{key}-{item[key]}: '
                            f'{lp-slogit_b4}'
                        )
                else:
                    print(f'{key}: {lp-slogit_b4}')

        for orig_var, fe_dict in (
            self.continuous_transformations[which].items()
        ):
            for coef_to_get, trafo in fe_dict.items():
                if (value := item[orig_var]) is not None:
                    contr = (
                        trafo(value) *
                        self.continuous_effects[which][orig_var][coef_to_get]
                    )
                    lp += contr

                    if (contr != 0) & (verbose > 1):
                        print(f'{coef_to_get}-{value}: {contr}')
                else:
                    print(f'{orig_var} yields None for {item}')

        return(lp)

    def _generate_rescue_eventcurve(
            self, donor: Donor,
            verbose: Optional[int] = 0) -> Tuple[
        np.ndarray,
        np.ndarray
    ]:
        if verbose:
            print('****** generate rescue event curve')
        lp = self._calculate_lp(
            item=donor.__dict__,
            which='coxph_rescue',
            realization_intercept=0,
            verbose=verbose
        )

        cbh = self.rescue_bh
        if self.rescue_bh_stratavars is not None:
            for strata_var in self.rescue_bh_stratavars:
                # print(donor.__dict__[strata_var])
                cbh = cbh[donor.__dict__[strata_var]]
        else:
            cbh = cbh[np.nan]

        ind_cbh = cbh[cn.CBH_RESCUE] * np.exp(lp)
        return cbh[cn.N_OFFERS_TILL_RESCUE], 1-np.exp(-ind_cbh)

    def generate_offers_to_rescue(
            self, donor: Donor,
            r_prob: Optional[float] = None,
            stratum: Optional[str] = None,
            verbose: Optional[int] = 0
    ) -> int:
        """ Sample the number of rejections made at triggering rescue/
            extended allocation from the empirical distribution per country
        """

        n_offers, event_probs = self._generate_rescue_eventcurve(
            donor=donor,
            verbose=verbose
        )

        if r_prob is None:
            r_prob = self.rng_rescue.random()

        if all(r_prob < event_probs):
            return int(0)
        elif any(r_prob < event_probs):
            which_n_offers = np.argmax(
                event_probs > r_prob
            )-1
            kth_offer = n_offers[which_n_offers]
        else:
            which_n_offers = len(n_offers)
            kth_offer = n_offers[which_n_offers-1]

        if verbose:
            print(f'{kth_offer} due to prob {r_prob}')

        return int(kth_offer)

    def predict_rescue_prob(
            self, donor: Donor, kth_offer: int,
            verbose: Optional[int] = 0) -> float:
        n_offers, event_probs = self._generate_rescue_eventcurve(
            donor=donor,
            verbose=verbose
        )

        if all(n_offers < kth_offer):
            which_prob = len(n_offers)-1
        else:
            which_prob = np.argmax(
                n_offers > kth_offer
            ) - 1

        return(event_probs[which_prob])

    def check_profile_compatibility(
            self, patient: Patient, donor: Donor
            ) -> bool:
        """Check whether profile is compatible with donor"""
        if patient.__dict__[cn.URGENCY_CODE] == cn.HU:
            return True
        if patient.profile is not None:
            return patient.profile._check_acceptable(donor)
        return False

    def return_transplanted_organ(
            self, match_record: Union[MatchRecord, MatchRecordCurrentELAS]
    ) -> int:
        """Return which organ is transplanted (based on offer and split)"""
        if match_record.__dict__[cn.TYPE_OFFER_DETAILED] != es.TYPE_OFFER_WLIV:
            return match_record.__dict__[cn.TYPE_OFFER_DETAILED]
        else:
            prob_split = self.calculate_prob_split(
                match_record
            )
            if prob_split > self.rng_split.random():
                if (
                    match_record.__dict__[cn.R_MATCH_AGE] <
                    es.AGE_BELOW_CLASSIC_SPLIT
                ):
                    return es.TYPE_OFFER_LLS
                else:
                    if (
                        match_record.__dict__[cn.R_WEIGHT] >=
                        es.WEIGHT_ABOVE_FULL_SPLIT
                    ):
                        return es.TYPE_OFFER_RL
                    else:
                        return es.TYPE_OFFER_ERL
            else:
                return es.TYPE_OFFER_WLIV

    def _patient_accept_always(
        self, match_record: Union[MatchRecord, MatchRecordCurrentELAS]
    ) -> float:
        """Policy to always accept the offer, if profile permits."""
        if match_record.__dict__[cn.MTCH_TIER] in es.IGNORE_PROFILE_TIERS:
            match_record.set_acceptance(
                reason=cn.T3 if match_record.donor.rescue else cn.T1
            )
            return True
        if match_record.patient.profile is not None:
            if self.check_profile_compatibility(
                match_record.patient,
                match_record.donor
            ):
                match_record.set_acceptance(
                    reason=cn.T3 if match_record.donor.rescue else cn.T1
                )
                return True
        match_record.set_acceptance(reason=cn.FP)
        return False

    def _patient_accept_lr(
        self, match_record: Union[MatchRecord, MatchRecordCurrentELAS]
    ) -> float:
        """Check acceptance with LR if profile acceptable or not checked."""
        if (
            match_record.__dict__[cn.MTCH_TIER] in es.IGNORE_PROFILE_TIERS or
            self.check_profile_compatibility(
                match_record.patient,
                match_record.donor
            )
        ):
            # Assume that HU/ACO candidates always reject split offers.
            if (
                match_record.__dict__[cn.TYPE_OFFER_DETAILED] != 4 and
                match_record.__dict__[cn.MTCH_TIER] in es.TIERS_HUACO_SPLIT
            ):
                match_record.set_acceptance(
                    reason=cn.RR
                )
                return False
            # Assume that rescue donors are not accepted in obligations
            if (
                match_record.donor.rescue and match_record.__dict__[cn.ANY_OBL]
            ):
                match_record.set_acceptance(
                    reason=cn.RR
                )
                return False
            if (
                self.calculate_prob_patient_accept(
                    offer=match_record,
                    verbose=0
                    ) >= match_record.patient.get_acceptance_prob()
            ):
                match_record.set_acceptance(
                    reason=cn.T3 if match_record.donor.rescue else cn.T1
                )
                return True
            else:
                match_record.set_acceptance(
                    reason=cn.RR
                )
        else:
            match_record.set_acceptance(
                reason=cn.FP
            )
        return False

    def _center_accept_always(
        self, center_offer: Union[CenterOffer, MatchRecord],
        verbose: Optional[int] = None
    ) -> bool:
        """Always accept center offer. """
        if isinstance(center_offer, CenterOffer):
            if len(center_offer.eligible_patients) > 0:
                center_offer.set_acceptance(
                    reason=cn.T3 if center_offer.donor.rescue else cn.T1
                )
                return True
            else:
                center_offer.set_acceptance(
                    reason=cn.FP
                )
                return False
        else:
            center_offer.set_acceptance(
                reason=cn.T3 if center_offer.donor.rescue else cn.T1
            )
            return True

    def _center_accept_lr(
        self, center_offer: Union[CenterOffer, MatchRecord],
        verbose: Optional[int] = None
    ) -> bool:
        """Check whether the center accepts."""
        # Assume that rescue donors are not accepted in obligations.
        # Note that centers still routinely accept if none of their
        # patients meet filtered offer criteria.
        if (
            center_offer.donor.rescue and center_offer.__dict__[cn.ANY_OBL]
        ):
            center_offer.set_acceptance(
                reason=cn.CR
            )
            return False
        center_offer.__dict__[cn.DRAWN_PROB_C] = self.rng_center.random()
        if (
            self.calculate_center_offer_accept(
                    offer=center_offer,
                    verbose=verbose
                ) >= center_offer.__dict__[cn.DRAWN_PROB_C]
        ):
            center_offer.set_acceptance(
                reason=cn.T3 if center_offer.donor.rescue else cn.T1
            )
            return True
        if (
            isinstance(center_offer, CenterOffer) and
            center_offer.n_profile_eligible == 0
        ):
            center_offer.set_acceptance(
                reason=cn.FP
            )
        else:
            center_offer.set_acceptance(
                reason=cn.CR
            )
        return False

    def _calc_prob_accept_pedad_huaco(
        self, offer: MatchRecord, verbose: Optional[int] = None
    ):
        """Calculate probability acceptance with separate
        HU/ACO and adult/ped models"""
        if verbose is None:
            verbose = self.verbose
        if verbose > 1:
            print('******* Pediatric HU/ACO')

        if offer.patient.r_aco or offer.__dict__[cn.PATIENT_IS_HU]:
            if offer.__dict__[cn.R_MATCH_AGE] < 18:
                selected_model = 'rd_ped_huaco'
            else:
                selected_model = 'rd_adult_huaco'
        else:
            if offer.__dict__[cn.R_MATCH_AGE] < 18:
                selected_model = 'rd_ped_reg'
            else:
                selected_model = 'rd_adult_reg'

        if (
            self.simulate_random_effects and
            selected_model in self.re_varcomps.keys()
        ):
            realization_intercept = self.simulate_random_intercept(
                offerdict=offer.__dict__,
                selected_model=selected_model
            )
        else:
            realization_intercept = 0

        return self._calculate_logit(
                offer=offer,
                which=selected_model,
                verbose=verbose,
                realization_intercept=realization_intercept
            )

    def _calc_prob_accept_huaco(
        self, offer: MatchRecord, verbose: Optional[int] = None
    ):
        """Calculate probability acceptance with separate HU/ACO models"""
        if verbose is None:
            verbose = self.verbose
        if verbose > 1:
            print('******* Patient-driven HU/ACO')
        if offer.patient.r_aco or offer.__dict__[cn.PATIENT_IS_HU]:
            selected_model = 'rd_huaco'
        else:
            selected_model = 'rd_reg'

        return self._calculate_logit(
                offer=offer,
                which=selected_model,
                verbose=verbose
        )

    def _calc_pcd_select(
        self, offer: MatchRecord, verbose: Optional[int] = None
    ):
        """Calculate probability acceptance with separate HU/ACO models"""
        if verbose is None:
            verbose = self.verbose
        if verbose > 1:
            print('******* Patient selection for center offers')
        return self._calculate_logit(
                offer=offer,
                which='pcd_ped' if offer.__dict__[cn.R_PED] else 'pcd_adult',
                verbose=verbose
        )

    def _calc_prob_accept(
        self, offer: MatchRecord, verbose: Optional[int] = None
    ):
        """Calculate probability acceptance with separate HU/ACO models"""
        if verbose is None:
            verbose = self.verbose
        if verbose > 1:
            print('******* Probability acceptance')
        return self._calculate_logit(
                offer=offer,
                which='rd',
                verbose=verbose
        )

    def calculate_center_offer_accept(
            self, offer: Union[CenterOffer, MatchRecord],
            verbose: Optional[int] = 0
    ) -> float:
        """Calculate probability center accepts"""
        if verbose and verbose > 1:
            print('******* Center offer')
        return self._calculate_logit(
                offer=offer,
                which='cd',
                verbose=verbose
        )

    def calculate_prob_split(
            self, offer: MatchRecord, verbose: Optional[int] = None
    ) -> float:
        """Calculate probability center accepts"""

        if offer.__dict__[cn.RECIPIENT_CENTER] in es.CENTERS_WHICH_SPLIT:
            return self._calculate_logit(
                    offer=offer,
                    which='sp',
                    verbose=verbose
                )
        else:
            return 0

    def simulate_random_intercept(
            self, offerdict: Dict[str, Any],
            selected_model: str
    ) -> float:
        # If simulating joint random effects, take from joint res only.
        realization_intercept = 0
        for var, sd in self.re_varcomps[selected_model].items():
            if var in self.joint_re_vars and self.joint_res is not None:

                if (
                    re := self.realization_joint_random_effects[var].get(
                        offerdict[var]
                    )
                ) is not None:
                    realization_intercept += re
                else:
                    re = self.rng_random_eff.normal(
                        loc=0,
                        scale=self.joint_res[var]
                    )
                    self.realization_joint_random_effects[
                        var
                    ][offerdict[var]] = re
                    realization_intercept += re
            elif (
                re := self.realizations_random_effects[selected_model][
                    var
                    ].get(offerdict[var])) is not None:
                realization_intercept += re
            else:
                re = self.rng_random_eff.normal(
                    loc=0,
                    scale=sd
                )
                self.realizations_random_effects[selected_model][
                    var
                ][offerdict[var]] = re
                realization_intercept += re
        return realization_intercept

    def _initialize_random_effects(self, re_levels: Dict[str, Any]) -> None:

        for selected_model, varcomps in self.re_varcomps.items():
            for var, sd in varcomps.items():
                if not (
                    self.simulate_joint_random_effects and
                    var in self.joint_re_vars
                ):
                    if (lvls := re_levels.get(var)):
                        d = dict(
                            zip(
                                lvls,
                                self.rng_random_eff.normal(
                                    loc=0,
                                    scale=sd,
                                    size=len(lvls)
                                )
                            )
                        )
                        self.realizations_random_effects[
                            selected_model
                            ][var].update(d)
                    else:
                        print(f'Random effects for {var} not initialized')

        if self.joint_res:
            for var, sd in self.joint_res.items():
                if (lvls := re_levels.get(var)):
                    d = dict(
                        zip(
                            lvls,
                            self.rng_random_eff.normal(
                                loc=0,
                                scale=sd,
                                size=len(lvls)
                            )
                        )
                    )
                    self.realization_joint_random_effects[var].update(d)
                else:
                    print(f'Random effects for {var} not initialized')

        return

    def _calculate_logit(
            self, offer: Union[MatchRecord, CenterOffer],
            which: str, verbose: Optional[int] = None,
            realization_intercept: Optional[float] = None
    ) -> float:
        """Calculate probability patient accepts"""
        if verbose is None:
            verbose = self.verbose

        slogit = self._calculate_lp(
            item=offer.__dict__,
            which=which,
            verbose=verbose,
            realization_intercept=realization_intercept
        )

        if which == 'cd':
            offer.__dict__[cn.PROB_ACCEPT_C] = (
                round_to_decimals(inv_logit(slogit), 3)
            )
        elif which != 'sp':
            offer.__dict__[cn.PROB_ACCEPT_P] = (
                round_to_decimals(inv_logit(slogit), 3)
            )
            if verbose:
                print(f'{which}: {round_to_decimals(inv_logit(slogit), 3)}')
                print(inv_logit(slogit))
        return inv_logit(slogit)

    def simulate_liver_allocation(
        self, match_list: MatchListCurrentELAS
    ) -> Optional[MatchRecord]:
        """ Iterate over a list of match records. If we model rescue alloc,
            we simulate when rescue will be triggered and terminate
            recipient-driven allocation. In that case, we simulate further
            allocation with the donor identified as a rescue donor, and
            prioritize locally in Belgium / regionally in Germany.
        """
        # When not simulating rescue, offer to all patients on the match list
        if not self.simulate_rescue:
            acc_matchrecord, _ = self._find_accepting_matchrecord(
                match_list.return_match_list()
            )
            return acc_matchrecord
        else:
            if (
                match_list.donor.__dict__[cn.D_DCD] and
                match_list.__dict__[cn.D_ALLOC_COUNTRY] != mgr.NETHERLANDS
            ):
                offers_till_rescue = 9999
            else:
                # Draw number of offers until rescue is initiated
                offers_till_rescue = self.generate_offers_to_rescue(
                    match_list.donor
                )

            # Allocate until `offers_till_rescue` rescue offers have been made.
            # This returns the match record for the accepting patient
            # (`acc_matchrecord`) if the graft was accepted, and a list of
            # centers willing to accept the organ (determined at the
            # center level)
            acc_matchrecord, center_willingness = (
                self._find_accepting_matchrecord(
                    match_list.return_match_list(),
                    max_offers=offers_till_rescue,
                    max_offers_per_center=5
                )
            )

            # If rescue was triggered before an acceptance, continue
            # allocation with prioritization for candidates with
            # rescue priority (local in BE, regional in DE).
            if not acc_matchrecord:
                match_list._initialize_rescue_priorities()
                match_list.donor.rescue = True

                # If rescue is triggered, prioritize remaining waitlist
                # on rescue priority (i.e. local in Belgium, regional in DE)
                acc_matchrecord, _ = self._find_accepting_matchrecord(
                    list(
                        sorted(
                            match_list.return_match_list(),
                            key=attrgetter(cn.RESCUE_PRIORITY),
                            reverse=True
                        )
                    ),
                    center_willing_to_accept=center_willingness
                )

            return acc_matchrecord

    def _find_accepting_matchrecord(
            self,
            match_records_list: List[
                Union[MatchRecord, MatchRecordCurrentELAS, CenterOffer]
            ],
            max_offers: Optional[int] = 9999,
            max_offers_per_center: int = 9999,
            center_willing_to_accept: Optional[Dict[str, bool]] = None
    ) -> Tuple[Optional[MatchRecord], Dict[str, bool]]:
        """ Iterate over all match records in the match list, and simulate
            if the match object accepts the graft offer.

        Parameters
        ------------
        match_records_list: List[
            Union[MatchRecord, MatchRecordCurrentELAS, CenterOffer]
            ]
            match list to iterate over, and make offers to
        max_offers: int
            maximum number of offers that will be made
            (rescue allocation is triggered after it)
        max_offers_per_center: int
            maximum number of offers per center that counts towards
            triggering rescue allocation. Recipients will still be
            offered the graft.
        center_willing_to_accept: Optional[Dict[str, bool]]
            Dictionary of center names with booleans whether they are
            willing to consider the graft or not.
        """
        count_rejections_total = 0
        n_rejections_per_center = defaultdict(int)
        if center_willing_to_accept is None:
            center_willing_to_accept = {}

        # Make offers to match objects. We count an offer as made if
        # (i) it is center-driven offer,
        # (ii) it is the first offer to a center in non-HU/ACO RD allocation,
        # (iii) it is an offer to a filtered match list candidate
        for match_object in match_records_list:
            # If more than max number of offers are made, break
            # to initiate rescue.
            if max_offers and count_rejections_total >= max_offers:
                break

            # Skip patients who already rejected the offer
            if match_object.__dict__.get(cn.ACCEPTANCE_REASON, None):
                continue
            elif hasattr(match_object, '_initialize_acceptance_information'):
                match_object._initialize_acceptance_information()

            if isinstance(match_object, CenterOffer):
                # Check center acceptance
                if self.determine_center_acceptance(match_object, verbose=0):
                    acc_matchrecord, _ = self.determine_accepting_patient(
                        match_object
                    )

                    # Check if there is an accepting patient (i.e.
                    # not all are filtered out for the center offer).
                    if acc_matchrecord:
                        acc_matchrecord.set_acceptance(
                            reason=(
                                cn.T3 if acc_matchrecord.donor.rescue
                                else cn.T1
                            )
                        )
                        # Copy over center rank from CenterOffer object.
                        acc_matchrecord.__dict__[cn.RANK] = (
                            match_object.__dict__.get(cn.RANK)
                        )
                        return acc_matchrecord, center_willing_to_accept
                else:
                    # Record rejection
                    if match_object.n_profile_eligible == 0:
                        match_object.set_acceptance(
                            reason=cn.FP
                        )
                    else:
                        count_rejections_total += 1
            else:
                # Recipient-driven allocation
                # 1) In case of MELD-based allocation (non-HU/ACO), simulate
                #   whether the center would be willing to accept the graft
                if not (
                    match_object.__dict__[cn.RECIPIENT_CENTER] in
                    center_willing_to_accept
                ) and not match_object.__dict__[cn.MTCH_TIER] in OBL_TIERS:
                    if self.check_profile_compatibility(
                        match_object.patient,
                        match_object.donor
                    ):
                        if self.determine_center_acceptance(
                            match_object, verbose=0
                        ):
                            center_willing_to_accept[
                                match_object.__dict__[cn.RECIPIENT_CENTER]
                            ] = True
                        else:
                            center_willing_to_accept[
                                match_object.__dict__[cn.RECIPIENT_CENTER]
                            ] = False
                            count_rejections_total += 1
                    else:
                        match_object.set_acceptance(
                            reason=cn.FP
                        )

                # 2) If center finds patient acceptable or candidate
                #       is in HU/ACO tiers, make a patient-driven offer
                if center_willing_to_accept.get(
                    match_object.__dict__[cn.RECIPIENT_CENTER],
                    False
                ) or match_object.__dict__[cn.MTCH_TIER] in OBL_TIERS:
                    # Make offer if the center has received fewer than
                    # `max_offers_per_center` offers, or if rescue was
                    # triggered
                    if (
                        n_rejections_per_center[
                            match_object.__dict__[cn.RECIPIENT_CENTER]
                        ] < max_offers_per_center
                    ) or match_object.donor.rescue:
                        if self.determine_patient_acceptance(match_object):
                            return match_object, center_willing_to_accept
                        elif (
                            match_object.__dict__[
                                cn.ACCEPTANCE_REASON
                                ] != cn.FP
                        ):
                            n_rejections_per_center[
                                match_object.__dict__[cn.RECIPIENT_CENTER]
                                ] += 1
                            count_rejections_total += 1
                else:
                    # Else record center rejection (CR)
                    match_object.set_acceptance(
                        reason=cn.CR
                    )

        return None, center_willing_to_accept

    def _highest_prob_acceptance(
            self, center_offer: CenterOffer
    ) -> Tuple[Optional[MatchRecord], float]:
        """Return patient with highest acceptance probability as accepting."""
        max_prob = 0
        offer_to_return = None
        for pat_offer in center_offer.eligible_patients:
            if (
                self.check_profile_compatibility(
                    pat_offer.patient,
                    pat_offer.donor
                )
            ):
                prob = self._calc_pcd_select(
                    offer=pat_offer
                )
                if prob > max_prob:
                    offer_to_return = pat_offer
                    max_prob = prob

        offer_to_return.__dict__[cn.DRAWN_PROB_C] = (
            center_offer.__dict__[cn.DRAWN_PROB_C]
        )

        return offer_to_return, max_prob

    def _prop_prob_acceptance(
        self, center_offer: CenterOffer,
        weight_fun: Optional[Callable]
    ) -> Tuple[Optional[MatchRecord], float]:
        """
            Return patient with probability proportional to acceptance prob.
            Note that centers can also accept center offers even if there is
            no donor profile eligible recipient. In this case, we add a small
            probability to determine the accepting candidate.
            """
        probs = np.full(
            len(center_offer.eligible_patients),
            fill_value=1e-8
            )
        for (i, pat_offer) in enumerate(center_offer.eligible_patients):
            if (
                pat_offer.patient.profile is not None and
                pat_offer.patient.profile._check_acceptable(pat_offer.donor)
            ):
                probs[i] = self._calc_pcd_select(
                        offer=pat_offer
                    )

        if weight_fun is None:
            sampling_probs = probs / sum(probs)
        else:
            sampling_probs = weight_fun(probs) / sum(weight_fun(probs))

        if sum(probs) > 0:
            index_to_return = self.rng_patient_per_center.choice(
                a=len(center_offer.eligible_patients),
                p=sampling_probs
            )
            to_return = center_offer.eligible_patients[index_to_return]
            to_return.__dict__[cn.OFFERED_TO] = (
                center_offer.__dict__[cn.OFFERED_TO]
            )

            return (
                to_return,
                probs[index_to_return]
            )
        else:
            return None, 1

    def _prop_sq_prob_acceptance(
        self, center_offer: CenterOffer
    ) -> Tuple[Optional[MatchRecord], float]:
        """
            Wrapper for _prop_prob_acceptance where we weigh patients by
            the squared probability of acceptance, as predicted by LR.
        """
        return self._prop_prob_acceptance(
            center_offer,
            weight_fun=es.square
        )

    def _first_accepting_patient(
            self, center_offer: CenterOffer
    ) -> Tuple[Optional[MatchRecord], float]:
        """Return first accepting patient as accepting."""
        for pat_offer in center_offer.eligible_patients:
            if pat_offer.patient.profile is not None:
                if self.check_profile_compatibility(
                    pat_offer.patient,
                    pat_offer.donor
                ):
                    if pat_offer:
                        pat_offer.set_acceptance(
                            reason=cn.T3 if pat_offer.donor.rescue else cn.T1
                        )
                    return pat_offer, 1

            pat_offer.set_acceptance(
                reason=cn.FP
            )
        return None, 1

    def fe_coefs_to_dict(self, level: pd.Series, coef: pd.Series):
        """Process coefficients to dictionary"""
        fe_dict = {}
        for lev, val in zip(level, coef):
            if ':' in str(lev):
                l1, lv2 = lev.split(':')
                var2, l2 = lv2.split('-')

                if l1 in fe_dict:

                    if var2 in fe_dict[l1]:
                        fe_dict[l1][var2].update(
                            {l2: val}
                        )
                    else:
                        fe_dict[l1].update(
                            {var2: {l2: val}}
                        )
                else:
                    fe_dict.update(
                        {l1: {var2: {l2: val}}}
                    )
            else:
                fe_dict.update(
                    {lev: val}
                )
        return fe_dict

    def _initialize_lr_coefs(
        self,
        dict_paths_coefs: Dict[str, str],
        path_coef_split: str
    ):
        """Initialize the logistic regression coefficients for
            recipient (rd) and center-driven (cd) allocation
        """

        self.__dict__['sp'] = pd.read_csv(path_coef_split, dtype='str')
        self.__dict__['sp']['coef'] = self.__dict__['sp']['coef'].astype(float)

        for k, v in dict_paths_coefs.items():
            self.__dict__[k] = pd.read_csv(v, dtype='str')
            self.__dict__[k]['coef'] = self.__dict__[k]['coef'].astype(float)

        coef_keys = list(dict_paths_coefs.keys()) + ['sp']

        # Create dictionary for fixed effects
        self.fixed_effects = {}
        self.reference_levels = {}
        for dict_name in coef_keys:
            self.fixed_effects[dict_name] = (
                self.__dict__[dict_name].loc[
                    ~ self.__dict__[dict_name].variable_transformed.notna()
                ].groupby('variable').
                apply(
                    lambda x: self.fe_coefs_to_dict(x['level'], x['coef'])
                ).to_dict()
            )
            self.reference_levels[dict_name] = defaultdict(lambda: None)

        for _, td in self.fixed_effects.items():
            for k, v in td.items():
                if (
                    isinstance(list(v.keys())[0], float) and
                    isnan(list(v.keys())[0])
                ):
                    td[k] = list(v.values())[0]

        self.continuous_transformations = {}
        self.continuous_effects = {}
        for dict_name in coef_keys:
            self.continuous_transformations[dict_name] = (
                self.__dict__[dict_name].loc[
                    self.__dict__[dict_name]
                    .variable_transformed.notna()
                ].groupby('variable').
                apply(
                    lambda x: {
                        k: construct_piecewise_term(v)
                        for k, v in zip(
                            x['variable_transformed'],
                            x['variable_transformation']
                        )
                    }
                ).to_dict()
            )

            self.continuous_effects[dict_name] = (
                self.__dict__[dict_name].loc[
                    self.__dict__[dict_name]
                    .variable_transformed.notna()
                ].groupby('variable').
                apply(
                    lambda x: dict(zip(x['variable_transformed'], x['coef']))
                ).to_dict()
            )
