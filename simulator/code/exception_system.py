#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

@author: H.C. de Ferrante
"""

from typing import Dict, Optional

import pandas as pd
import numpy as np

from simulator.code.utils import DotDict, round_to_decimals, round_to_int
import simulator.magic_values.elass_settings as es
from simulator.code.read_input_files import read_se_rules
import simulator.magic_values.column_names as cn
from simulator.code.ScoringFunction import clamp


class InvertBonuses:
    """Calculate the approximate match MELD, if based on a bonus SE"""

    def __init__(self, eqs: Dict[int, float]) -> None:
        dvs = list(eqs.values())
        self.scores = dict()
        for score in range(6, 40):
            key_gt = next(
                i for i, x in enumerate(
                    list(eqs.values())
                    ) if x > score
            )
            self.scores[score] = (dvs[key_gt-1] + dvs[key_gt])/2


class ExceptionScore:
    """Class which implements a MELD exception
    ...

    Attributes   #noqa
    ----------
    id_se: int
        identifier for exception score
    dis: str
        disease type
    type_e: str
        name of exception
    eq_init: int
        initial equivalent
    eq_upgrade: int
        upgrade
    eq_upgrade_schedule: int
        number of days after which upgrade
    max_age_upgrade: Optional[int]
        age after which score is no longer upgraded
    eq_max: int
        Maximum equivalent score
    current_eq: Optional[int]
        Current equivalence
    adds_bonus: bool
        Whether it adds a bonus
    adds_bonus_amount: int
        percentage bonus SE awards.
    allowed_eqs_by: int
        seq(from, to, by) by argument from initial to max.
    allowed_eqs_offset: int
        number by which to decrease from argument to retrieve allowed values.
    replacement_se: Optional[str]
        Optional replacement SE
    use_real: bool
        whether to copy over real exceptions, or upgrade exceptions
    nse_delay: float
        number of days to delay receival of (N)SE with.

    Methods
    -------
    set_equivalent():
        Set mortality equivalent
    set_to_initial_equivalent():
        Set to initial mortality equivalent
    upgrade_equivalent():
        Upgrade from current equivalent with upgrade
    next_update_in():
        Receive time difference to next update
    calculate_bonus_meld(lab_meld):
        Calculate SE MELD score

    """

    def __init__(self, id_se: str, dis: str, type_e: str, e_country: str,
                 eq_init: float, eq_upgrade: float, eq_upgrade_schedule: int,
                 eq_max: float, eq_bonus: bool, max_age_upgrade: int,
                 allowed_eqs_by: int, allowed_eqs_offset: float,
                 replacement_se: str,
                 sim_set: DotDict,
                 current_eq: Optional[int] = None,
                 nse_delay: float = 0) -> None:

        self.__dict__[cn.ID_SE] = id_se

        assert type_e in (cn.SE, cn.NSE, cn.PED), \
            'Type exception must be SE, NSE, or PED.'
        self.type_e = type_e

        assert e_country in es.ET_COUNTRIES, \
            'Exception country must be an ET country'
        self.e_country = e_country

        self.dis = dis
        self.eq_init = eq_init
        self.eq_upgrade = eq_upgrade
        self.eq_max = eq_max
        self.max_age_upgrade = max_age_upgrade

        self.upgr_time = eq_upgrade_schedule
        self.current_eq = current_eq if current_eq else eq_init
        self.adds_bonus = eq_bonus
        self.allowed_eqs_by = allowed_eqs_by
        self.allowed_eqs_offset = allowed_eqs_offset
        self.replacement_se = replacement_se
        if self.adds_bonus:
            self.adds_bonus_amount = eq_init
        else:
            self.adds_bonus_amount = 0
        self.retain_historic_se = sim_set.__dict__.get(
            'RETAIN_HISTORIC_SE', True
        )
        if self.allowed_eqs_by == 0:
            self.allowed_eqs = set([self.eq_init, self.eq_max])
        else:
            self.allowed_eqs = set(
                list(
                    np.arange(
                        self.eq_init - self.allowed_eqs_offset,
                        self.eq_max,
                        self.allowed_eqs_by
                    )
                ) + [self.eq_max]
            )

        # Whether to copy over values from real SE updates,
        # (i) during simulation, and (ii) during simulation initialization
        self.use_real_se = sim_set.USE_REAL_SE
        self.real_se_before_simstart = sim_set.__dict__.get(
            'REAL_SE_BEFORE_SIMSTART',
            True
        )

        self.nse_delay = nse_delay

    def set_equivalent(self, eqv: float) -> None:
        """Set mortality equivalent to given equivalent"""

        # Ensure that upgrade is valid
        eqv = min(eqv, self.eq_max)
        if self.eq_upgrade != 0:
            assert self.use_real_se or eqv in self.allowed_eqs, \
                    (
                        f'{eqv}% is not a valid mortality equivalent'
                        f' for {self.__dict__[cn.ID_SE]}: {self.dis}'
                    )

        self.current_eq = eqv

    def set_to_initial_equivalent(self) -> None:
        """Set initial equivalent"""
        self.current_eq = self.eq_init

    def upgrade_equivalent(self) -> None:
        """Upgrade current equivalent"""
        assert self.current_eq is not None, \
            'Cannot upgrade an unitialized exception status'
        self.current_eq = min(
            self.current_eq + self.eq_upgrade,
            self.eq_max
            )

    def next_update_in(self) -> int:
        """Generate future update time."""
        return self.upgr_time

    def __str__(self) -> str:
        if self.current_eq is not None:
            return(
                f'{self.current_eq}%-{self.type_e} mortality eqv.'
                f' for {self.dis} in {self.e_country}'
            )
        return f'{self.type_e} for {self.dis} in {self.e_country}'

    def __repr__(self) -> str:
        if self.current_eq is not None:
            return(
                f'{self.current_eq}%-{self.type_e} mortality eqv.'
                f'for {self.dis} in {self.e_country}'
                )
        else:
            return f'{self.type_e} for {self.dis} in {self.e_country}'


class ExceptionSystem:
    """Class which implements the MELD exception scores
    ...

    Attributes   #noqa
    ----------
    exceptions: Dict[str, Dict[str, ExceptionScore]]
        standard exception scores

    Methods
    -------

    """

    def __init__(self, sim_set) -> None:

        path_ses = sim_set.PATH_SE_SETTINGS

        sel_score = (
            sim_set.LAB_MELD if
            sim_set.LAB_MELD != cn.SCORE_LABHISTORIC
            else cn.SCORE_UNOSMELD
        )

        self.meld_limits = sim_set.SIMULATION_SCORES[
            sel_score
        ].get('score_limits')

        # Read in pd.DataFrame with the standard exceptions.
        self.exceptions: Dict[str, Dict[str, ExceptionScore]] = {
            'SE': {},
            'NSE': {},
            'PED': {}
        }

        # Load in standard exceptions
        df_se = read_se_rules(path_ses, add_replacing_bonus_ses=True)
        df_se[cn.MAX_AGE_UPGRADE] = df_se[cn.MAX_AGE_UPGRADE].fillna(9999)
        for index, row in df_se.iterrows():
            self.exceptions[str(row[cn.TYPE_E])][str(index)] = ExceptionScore(
                id_se=str(index),
                type_e=row[cn.TYPE_E],
                dis=row[cn.DISEASE_SE],
                e_country=row[cn.SE_COUNTRY],
                eq_init=row[cn.INITIAL_EQUIVALENT],
                eq_upgrade=row[cn.SE_UPGRADE],
                eq_upgrade_schedule=row[cn.SE_UPGRADE_INTERVAL],
                max_age_upgrade=row[cn.MAX_AGE_UPGRADE],
                eq_max=row[cn.MAX_EQUIVALENT],
                eq_bonus=row[cn.LAB_MELD_BONUS],
                allowed_eqs_by=row[cn.ALLOWED_EQUIVALENTS_BY],
                allowed_eqs_offset=row[cn.ALLOWED_EQS_OFFSET],
                replacement_se=row[cn.REPLACEMENT_SE],
                sim_set=sim_set,
                nse_delay=float(row.get(cn.NSE_DELAY, 0))
            )

        # Initialize how to calculate mortality equivalents
        self.exc_meld_slope = sim_set.EXC_SLOPE
        self.exc_meld10_eq = sim_set.EXC_MELD10_EQUIVALENT
        self.exc_max = sim_set.EXC_MAX

    def get_equivalent_meld(self, mrt: float) -> float:
        """Calculate equivalent MELD score from mortalty equivalent."""
        if mrt >= 100:
            return self.exc_max
        else:
            eq = round_to_decimals(
                np.log(np.log(1-mrt/100) / np.log(self.exc_meld10_eq)) /
                self.exc_meld_slope + 10,
                2
            )
            return eq

    def get_mrt_equivalent(self, meld: float):
        if meld >= self.exc_max:
            return 100
        else:
            return 100 - round_to_int(
                self.exc_meld10_eq**(
                    np.exp(self.exc_meld_slope*(meld - 10))
                ) * 100
            )

    def calculate_bonus_meld(
            self, lab_meld: float, bonus_amount: float
    ) -> float:
        """Approximate the bonus MELD."""
        meld_eq = self.get_mrt_equivalent(lab_meld)
        return round_to_decimals(
            clamp(
                self.get_equivalent_meld(
                    meld_eq + float(bonus_amount)
                ),
                lims=self.meld_limits
            ),
            2
        )

    def __str__(self) -> str:
        """Print out exception system."""
        return ', '.join([
                f'{len(v)} exception scores for {k}'
                for k, v in self.exceptions.items()
            ])

    def __repr__(self) -> str:
        """Print out exception system."""
        return ', '.join([
            f'{len(v)} exception scores for {k}'
            for k, v in self.exceptions.items()
            ])
