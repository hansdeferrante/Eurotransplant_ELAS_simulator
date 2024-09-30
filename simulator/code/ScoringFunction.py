#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19th

@author: H.C. de Ferrante
"""

from typing import Tuple, Union, Optional, TYPE_CHECKING
import simulator.magic_values.elass_settings as es
import simulator.magic_values.column_names as cn
from simulator.code.utils import round_to_decimals, round_to_int
from simulator.code.functions import construct_alloc_fun, construct_minus_fun
from math import isnan
import numpy as np

if TYPE_CHECKING:
    from simulator.code import entities
    from simulator.code import AllocationSystem


def clamp(x: float, lims: Tuple[float, float], default_lim: int = 0) -> float:
    """Force number between limits. Do not return a number."""
    if isnan(x):
        return(lims[default_lim])
    return max(min(lims[1], x), lims[0])


BIOMARKER_INTERACTIONS = {
    'revsodiumlncrea': {cn.SODIUM: es.revNa, cn.CREA: np.log}
}


class MELDScoringFunction:
    """Class which implements MELD scoring functions
    ...

    Attributes   #noqa
    ----------
    coef: dict[str, float]
        coefficients to use to calculate score
    intercept: float
        intercept for calculating score
    trafos: dict[str, str]
        transformations to apply to biomarkers
    caps: dict[str, Tuple[float, float]]
        caps to apply to the biomarkers
    limits: Tuple[float, float]
        caps to apply to final score
    round: bool
        whether to round scores to nearest integer

    Methods
    -------
    calc_score(biomarkers) -> float
    """

    def __init__(
            self,
            coef: dict[str, float],
            intercept: float,
            trafos: dict[str, str],
            caps: dict[str, Tuple[float, float]],
            limits: Tuple[float, float],
            rnd: bool,
            clamp_defaults: Optional[dict[str, int]] = None
    ) -> None:
        self.intercept = intercept
        self.coef = coef
        self.trafos = trafos
        self.caps = caps

        if clamp_defaults:
            self.clamp_defaults = clamp_defaults
        else:
            # Whether to clamp a biomarker to the
            # lower (0) or upper (1) limit
            self.clamp_defaults = {
                cn.ALBU: 1,
                cn.SODIUM: 1,
                cn.CREA: 0,
                cn.BILI: 0,
                cn.INR: 0
            }

        for k in BIOMARKER_INTERACTIONS.keys():
            if k not in self.caps:
                self.caps[k] = (-9999, 9999)
        self.limits = limits
        self.round = rnd

    def calc_score(
            self,
            biomarkers: dict[str, float]
            ) -> float:
        """Calculate the score"""

        score = self.intercept

        # Replace creatinine by upper cap for patients on dialysis
        if biomarkers[cn.DIAL_BIWEEKLY]:
            biomarkers[
                cn.CREA
            ] = self.caps[cn.CREA][1]

        # Calculate score
        for k, v in self.coef.items():
            if k in BIOMARKER_INTERACTIONS:
                if biomarkers['sodium'] > 138.6:
                    biomarkers[k] = 0
                else:
                    biomarkers[k] = 1
                    for bm, tr in BIOMARKER_INTERACTIONS[k].items():
                        biomarkers[k] = tr(
                            clamp(
                                biomarkers[bm], self.caps[bm],
                                self.clamp_defaults[bm]
                            )
                        ) * biomarkers[k]

            score += es.TRAFOS[self.trafos[k]](
                clamp(
                    biomarkers[k], self.caps[k],
                    self.clamp_defaults.get(k, 0)
                )
            ) * v

        if self.round:
            return round_to_int(clamp(score, self.limits))
        return clamp(score, self.limits)

    def __str__(self):
        fcoefs = [
            f'{round_to_decimals(v, p=3)}*{self.trafos[k]}({k})'
            for k, v in self.coef.items()
            ]
        if self.intercept != 0:
            return ' + '.join([str(self.intercept)] + fcoefs)
        else:
            return ' + '.join(fcoefs)


class AllocationScore:
    """Class which implements an allocation score
    ...

    Attributes   #noqa
    ----------
    coef: dict[str, float]
        coefficients to use to calculate score
    intercept: float
        intercept for calculating score
    limits: Tuple[float, float]
        caps to apply to final score
    round: bool
        whether to round scores to nearest integer

    Methods
    -------
    calc_score(x_dict) -> float
    """

    def __init__(
            self,
            coef: dict[str, float],
            intercept: float,
            limits: Tuple[float, float],
            lab_meld: str,
            rnd: bool
    ) -> None:
        self.intercept = intercept
        self.coef = {k: v for k, v in coef.items() if v != 0}
        self.coef[lab_meld] = self.coef.pop('labmeld')
        self.limits = limits
        self.round = rnd
        self.nonzero_coefs = [k for k, v in self.coef.items() if v != 0]
        self.needed_variables = list(set([
            i.split('-')[0] for i in self.nonzero_coefs
        ]))
        self.raw_variables = list(
            set(
                sum(
                    (var.split('_minus_') for var in self.needed_variables),
                    []
                )
            )
        )
        self.trafos = {
            i: {
                'var': i.split('-')[0],
                'trafo': construct_alloc_fun(
                    i.split('-')[1]
                )
            }
            for i in self.nonzero_coefs if '-' in i
        }
        self.var_construction_funs = {
            var: construct_minus_fun(var)
            for var in self.needed_variables
            if 'minus' in var
        }

    def calc_score(
            self,
            x_dict: dict[str, float],
            verbose: int = 0
            ) -> Union[float, int]:
        """Calculate the score"""

        score = self.intercept

        if self.var_construction_funs:
            x_dict.update(
                {
                    var: fun(x_dict)
                    for var, fun
                    in self.var_construction_funs.items()}
            )

        if self.trafos:
            for term, trafo in self.trafos.items():
                x_dict[term] = trafo['trafo'](x_dict.get(trafo['var']))
        if verbose:
            print('***** Calculation score in class AllocationScore:')
        for k, v in self.coef.items():
            if verbose:
                print(f'key {k}, coef: {v}, obs value: {x_dict[k]}')
            score += x_dict[k] * v
        if self.round:
            if verbose:
                print(
                    f'Final score: '
                    f'{round_to_int(clamp(score, self.limits))}'
                )
            return round_to_int(clamp(score, self.limits))
        if verbose:
            print(f'Final score: {clamp(score, self.limits)}')
        return clamp(score, self.limits)

    def __str__(self):
        fcoefs = [
            f'{round_to_decimals(v, p=3)}*{k}'
            for k, v in self.coef.items()
            if v != 0
            ]
        if self.intercept != 0:
            return ' + '.join([str(self.intercept)] + fcoefs)
        else:
            return ' + '.join(fcoefs)
