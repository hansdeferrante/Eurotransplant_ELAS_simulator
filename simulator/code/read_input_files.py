#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:33:44 2022

Scripts to read in input files.

@author: H.C. de Ferrante
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from functools import reduce
from numpy import arange, ndarray, nan, array
import pandas as pd
import warnings
import yaml
from collections import defaultdict
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

from simulator.code.utils import DotDict, round_to_decimals, round_to_int
from simulator.code.ScoringFunction import MELDScoringFunction, \
    AllocationScore
import simulator.magic_values.inputfile_settings as dtypes
import simulator.magic_values.column_names as cn
import simulator.magic_values.elass_settings as es
from simulator.magic_values.column_groups import SE_RULE_COLS, \
    MATCHLIST_COLS
from simulator.magic_values.elass_settings import \
    ET_COUNTRIES

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def read_datetimes_dayfirst(x):
    return pd.to_datetime(x, dayfirst=True)


def _read_with_datetime_cols(
        input_path: str,
        dtps: Dict[str, Any],
        casecols: bool = False,
        usecols: Optional[List[str]] = None,
        datecols: Optional[List[str]] = None,
        **kwargs
) -> pd.DataFrame:
    """Read in pd.DataFrame with datecols as datetime"""

    if usecols is None:
        usecols = list(dtps.keys())
    if casecols:
        usecols = [c.upper() for c in usecols]
        dtps = {k.upper(): v for k, v in dtps.items()}
        if datecols:
            datecols = [c.upper() for c in datecols]

    data_ = pd.read_csv(
        input_path,
        dtype=dtps,
        usecols=lambda x: x in usecols,
        **kwargs
    )
    if datecols:
        for col in datecols:
            col_format = (
                "%Y-%m-%d"
                if all(data_[col].str.len() == 10)
                else "%Y-%m-%d %H:%M:%S"
            )
            data_[col] = pd.to_datetime(
                data_[col],
                format=col_format
            )

    assert isinstance(data_, pd.DataFrame), \
        f'Expected DataFrame, not {type(data_)}'

    # Read in as standard datetime object, not pd.Timestamp
    if datecols:
        for date_col in datecols:
            data_[date_col] = pd.Series(
                array(data_[date_col].dt.to_pydatetime()),
                dtype='object'
            )

    if casecols:
        data_.columns = data_.columns.str.lower()

    return data_


def read_se_rules(
        input_path: str,
        add_replacing_bonus_ses: bool = True,
        replacing_ses: Optional[List[str]] = es.REPLACING_BONUS_SES
        ) -> pd.DataFrame:
    """Read in the rules for standard exceptions."""

    d_se = _read_with_datetime_cols(
        input_path=input_path,
        dtps=dtypes.DTYPE_SE_RULES,
        usecols=list(dtypes.DTYPE_SE_RULES.keys()),
        casecols=True
    )
    d_se.set_index(cn.ID_SE, inplace=True)

    # Assert column names are correct
    for col in SE_RULE_COLS:
        assert col in d_se.columns, \
            f'{col} should be column for {input_path}'

    # Assert countries are in the ET countries.
    assert all(d_se[cn.SE_COUNTRY].isin(ET_COUNTRIES)), \
        'Non-ET countries are included in the ET rules.'

    if add_replacing_bonus_ses:
        d_psc = d_se.loc[
             replacing_ses, :
             ]
        d_psc.index = d_psc.index.astype(int) + es.SE_OFFSET
        d_psc.index = d_psc.index.astype(str)
        d_psc[cn.LAB_MELD_BONUS] = 1
        d_psc[cn.DISEASE_SE] = d_psc[cn.DISEASE_SE] + ' (Bonus)'

        d_se = pd.concat(
            [d_se, d_psc]
        )

    return d_se


def read_offerlist(input_path: str, usecols=None, **kwargs) -> pd.DataFrame:
    """Read in the MatchList."""

    if usecols is None:
        datecols = [x.upper() for x in [cn.MATCH_DATE, cn.R_DOB]]
        usecols = list(dtypes.DTYPE_OFFERLIST.keys())
    else:
        datecols = [
            x.upper() for x in
            set(usecols).intersection(
                set([x.upper() for x in [cn.MATCH_DATE, cn.R_DOB]])
                )
            ]

    d_o = _read_with_datetime_cols(
        input_path=input_path,
        dtps=dtypes.DTYPE_OFFERLIST,
        casecols=True,
        usecols=usecols,
        datecols=datecols,
        **kwargs
    )

    d_o = d_o.drop_duplicates()
    d_o.columns = d_o.columns.str.lower()

    d_o[cn.D_DCD] = d_o[cn.D_DCD].fillna(0)

    # Assert column names are correct
    if usecols is None:
        for col in MATCHLIST_COLS:
            assert col in d_o.columns, \
                f'{col} should be column for {input_path}'

    return d_o


def read_patients(
    input_path: str,
    datecols: Optional[List[str]] = None,
    usecols: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """"Read in patient file."""

    if datecols is None:
        datecols = [cn.TIME_REGISTRATION, cn.R_DOB]
    if usecols is None:
        usecols = list(dtypes.DTYPE_PATIENTLIST.keys())

    data_ = _read_with_datetime_cols(
        input_path=input_path,
        dtps=dtypes.DTYPE_PATIENTLIST,
        casecols=False,
        usecols=usecols,
        datecols=datecols,
        **kwargs
    )
    assert isinstance(data_, pd.DataFrame), \
        f'Expected DataFrame, not {type(data_)}'

    # Sort recipients by id & time of registration.
    data_: pd.DataFrame = data_.sort_values(
        by=[cn.ID_RECIPIENT, cn.TIME_REGISTRATION]
    )
    data_.reset_index(inplace=True)

    # Add re-transplantation information
    idx = data_[cn.TIME_SINCE_PREV_TXP].notna()

    data_.loc[idx, cn.PREV_TX_DATE] = (
        data_.loc[idx, cn.TIME_REGISTRATION] -
        data_.loc[idx, cn.TIME_SINCE_PREV_TXP].apply(
            lambda x: timedelta(days=x)
        )
    )

    return data_


def read_obligations(
    input_path: str,
    datecols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Read in obligations"""

    if datecols is None:
        datecols = [cn.OBL_INSERT_DATE, cn.OBL_START_DATE, cn.OBL_END_DATE]

    data_: pd.DataFrame = _read_with_datetime_cols(
        input_path=input_path,
        dtps=dtypes.DTYPE_OBLIGATIONS,
        casecols=True,
        datecols=datecols
    )

    return data_


def read_rescue_probs(
    input_path: str
) -> Dict[str, ndarray]:
    """Read in rescue probabilities from Kaplan-Meier"""
    rescue_probs = pd.read_csv(
        input_path,
        delimiter=','
    )
    rescue_probs = {
        n: x.iloc[:, 0:2] for n, x in rescue_probs.groupby('strata')
        }
    rescue_probs = {
        k: {
            cn.N_OFFERS_TILL_RESCUE: v['offers_before_rescue'].to_numpy(),
            cn.PROB_TILL_RESCUE: v['prob'].to_numpy()
        }
        for k, v in rescue_probs.items()
    }

    return rescue_probs


def read_rescue_baseline_hazards(
    input_path: str
) -> Tuple[Dict[str, ndarray], Tuple[str]]:
    """Read in rescue parameters for Cox PH model"""
    rescue_probs = pd.read_csv(
        input_path,
        delimiter=','
    )
    if 'strata' in rescue_probs.columns:
        rescue_probs = {
            n: x.iloc[:, 0:2] for n, x in rescue_probs.groupby('strata')
            }
        rescue_prob_dict = defaultdict(dict)
        strata_vars = None
        for k, v in rescue_probs.items():
            if ',' in k:
                k1, k2 = k.split(',')
                bh_var1, bh_var1_level = k1.split('=')
                bh_var2, bh_var2_level = k2.split('=')
                rescue_prob_dict[bh_var1_level].update(
                    {
                        bh_var2_level: {
                            cn.N_OFFERS_TILL_RESCUE:
                            v['offers_before_rescue'].to_numpy(),
                            cn.CBH_RESCUE: v['hazard'].to_numpy()
                        }
                    }
                )
                strata_vars = (bh_var1.strip(), bh_var2.strip())
            else:
                bh_var, bh_level = k.split('=')

                rescue_prob_dict[bh_level] = {
                    cn.N_OFFERS_TILL_RESCUE:
                    v['offers_before_rescue'].to_numpy(),
                    cn.CBH_RESCUE: v['hazard'].to_numpy()
                }
                strata_vars = (bh_var, )

        return rescue_prob_dict, strata_vars
    else:
        rescue_probs = {
            nan: {
                cn.N_OFFERS_TILL_RESCUE:
                rescue_probs.loc[:, 'offers_before_rescue'].to_numpy(),
                cn.CBH_RESCUE: rescue_probs.loc[:, 'hazard'].to_numpy()
            }
        }
    return rescue_probs, None


def read_donors(
        input_path: str,
        datecols: Optional[List[str]] = None,
        usecols: Optional[List[str]] = None,
        **kwargs
) -> pd.DataFrame:
    """"Read in patient file."""

    if usecols is None:
        usecols = list(dtypes.DTYPE_DONORLIST.keys())
    if datecols is None:
        datecols = [cn.D_DATE]

    data_: pd.DataFrame = _read_with_datetime_cols(
        input_path=input_path,
        dtps=dtypes.DTYPE_DONORLIST,
        casecols=False,
        usecols=usecols,
        datecols=datecols,
        **kwargs
    )

    # Remove donors not from ET
    data_ = data_[data_[cn.D_COUNTRY].isin(es.ET_COUNTRIES)]

    data_ = data_.fillna(
        value=dtypes.DTYPE_DONOR_FILL_NAS
    )

    return data_


def read_acos(input_path: str, **kwargs) -> pd.DataFrame:
    """"Read in patient file."""

    data_ = pd.read_csv(
        filepath_or_buffer=input_path,
        dtype=dtypes.DTYPE_ACOS,
        date_parser=read_datetimes_dayfirst,
        usecols=list(dtypes.DTYPE_ACOS.keys()),
        **kwargs
    )
    assert isinstance(data_, pd.DataFrame), \
        f'Expected DataFrame, not {type(data_)}'
    data_.columns = data_.columns.str.lower()

    return data_


def read_diags(input_path: str, **kwargs) -> pd.DataFrame:
    """"Read in patient file."""

    data_ = pd.read_csv(
        filepath_or_buffer=input_path,
        dtype=dtypes.DTYPE_DIAGS,
        date_parser=read_datetimes_dayfirst,
        usecols=list(dtypes.DTYPE_DIAGS.keys()),
        **kwargs
    )
    assert isinstance(data_, pd.DataFrame), \
        f'Expected DataFrame, not {type(data_)}'
    data_.columns = data_.columns.str.lower()

    return data_


def read_profiles(input_path: str, **kwargs) -> pd.DataFrame:
    """"Read in patient file."""

    data_ = pd.read_csv(
        input_path,
        dtype=dtypes.DTYPE_PROFILES,
        date_parser=read_datetimes_dayfirst,
        usecols=list(dtypes.DTYPE_PROFILES.keys()),
        **kwargs
    )
    assert isinstance(data_, pd.DataFrame), \
        f'Expected DataFrame, not {type(data_)}'
    data_.columns = data_.columns.str.lower()

    return data_


def read_status_updates(
        input_path: str, sim_set: DotDict,
        start_date_col: Optional[str] = None,
        end_date_col: Optional[str] = None,
        fix_bonus_ses: bool = True,
        **kwargs) -> pd.DataFrame:
    """Read in status updates."""

    d_s = pd.read_csv(
        input_path,
        dtype=dtypes.DTYPE_STATUSUPDATES,
        usecols=list(dtypes.DTYPE_STATUSUPDATES.keys()),
        parse_dates=[cn.LISTING_DATE],
        **kwargs
    )
    assert isinstance(d_s, pd.DataFrame), \
        f'Expected DataFrame, not {type(d_s)}'
    d_s.columns = d_s.columns.str.lower()

    end_date: datetime = (
        sim_set.SIM_END_DATE if not end_date_col
        else sim_set.__dict__[end_date_col]
    )
    d_s = d_s.loc[
        d_s[cn.LISTING_DATE] <= end_date,
        :
    ]

    # Impute status details forward
    d_s.loc[
        d_s[cn.TYPE_UPDATE].isin(es.STATUSES_EXCEPTION),
        cn.STATUS_DETAIL
    ] = d_s.loc[
         d_s[cn.TYPE_UPDATE].isin(es.STATUSES_EXCEPTION)
    ].groupby(cn.ID_REGISTRATION)[cn.STATUS_DETAIL].ffill()

    # Set variable detail to removal reason for patients removed.
    d_s.loc[
        d_s[cn.STATUS_VALUE] == 'R',
        cn.STATUS_DETAIL
    ] = d_s.loc[
        d_s[cn.STATUS_VALUE] == 'R',
        cn.REMOVAL_REASON
    ]

    # If we simulate all exceptions, we remove repeat statuses
    if not sim_set['USE_REAL_SE']:
        d_s.loc[
            d_s[cn.TYPE_UPDATE].isin(es.STATUSES_EXCEPTION),
            'repeat_se'
        ] = d_s.loc[
                 d_s[cn.TYPE_UPDATE].isin(es.STATUSES_EXCEPTION)
            ].groupby(
                [cn.ID_REGISTRATION, cn.TYPE_UPDATE, cn.STATUS_DETAIL]
                )[cn.STATUS_ACTION].diff().abs().fillna(1) != 1

        d_s = d_s.loc[
            ~ d_s.repeat_se.fillna(False)
        ].drop(
            ['repeat_se'],
            axis=1
        )

    if not sim_set.USE_REAL_PED:
        d_s.loc[
            d_s[cn.TYPE_UPDATE] == cn.PED,
            'repeat_ped'
        ] = d_s.loc[
                 d_s[cn.TYPE_UPDATE] == cn.PED
            ].groupby(
                [cn.ID_REGISTRATION, cn.TYPE_UPDATE]
                )[cn.STATUS_ACTION].diff().abs().fillna(1) != 1

        d_s = d_s.loc[
            ~ d_s.repeat_ped.fillna(False)
        ].drop(
            ['repeat_ped'],
            axis=1
        )

    # Add the SE offset to missing SEs (these apply a bonus).
    if fix_bonus_ses:
        rules_se = read_se_rules(
            sim_set.PATH_SE_SETTINGS,
            add_replacing_bonus_ses=False
            )
        bonus_ses = [
            str(x) for x in rules_se.loc[
                    rules_se[cn.LAB_MELD_BONUS] == 1
                    ].index]
        missing_bonuses = (d_s[cn.TYPE_UPDATE] == cn.SE) & \
            (d_s[cn.STATUS_ACTION] != 0) & \
            (d_s[cn.STATUS_VALUE].isna()) & \
            (~ d_s[cn.STATUS_DETAIL].isin(bonus_ses))

        d_s.loc[missing_bonuses, cn.STATUS_DETAIL] = d_s.loc[
            missing_bonuses, cn.STATUS_DETAIL
        ].apply(lambda x: str(int(x) + es.SE_OFFSET))

    return d_s


def read_sim_settings(
        ss_path: str,
        date_settings: Optional[List[str]] = None
) -> DotDict:
    """Read in simulation settings"""
    with open(ss_path, "r", encoding='utf-8') as file:
        sim_set: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

    if date_settings is None:
        date_settings = [
            'SIM_END_DATE', 'SIM_START_DATE',
            'LOAD_RETXS_TO', 'LOAD_RETXS_FROM'
        ]

    sim_set['SCORES'] = {
        k: MELDScoringFunction(
            coef=v[cn.COEF],
            trafos=v[cn.TRAFOS],
            intercept=v[cn.INTERCEPT],
            caps=v[cn.CAPS],
            limits=v[cn.SCORE_LIMITS],
            rnd=v[cn.SCORE_ROUND]
        ) for k, v in sim_set['SIMULATION_SCORES'].items()
    }

    sim_set['ALLOC_SCORE'] = AllocationScore(
        coef=sim_set['ALLOCATION_SCORE'][cn.COEF],
        intercept=sim_set['ALLOCATION_SCORE'][cn.INTERCEPT],
        limits=sim_set['ALLOCATION_SCORE'][cn.SCORE_LIMITS],
        rnd=sim_set['ALLOCATION_SCORE'][cn.SCORE_ROUND],
        lab_meld=sim_set['LAB_MELD']
    )

    sim_set['MELD_GRID'] = arange(
        start=int(min(sim_set['MIN_MELD_SCORE'], sim_set['DOWNMARKED_MELD'])),
        stop=int(sim_set['MAX_MELD_SCORE'])+1,
        step=1
    )

    sim_set['MELD_GRID_DICT'] = {
            meld: i for i, meld in enumerate(sim_set['MELD_GRID'])
    }

    sim_set['DOWNMARKED_MELD'] = max(
        sim_set['DOWNMARKED_MELD'],
        sim_set['MIN_MELD_SCORE']
    )

    # Fix dates
    min_time = datetime.min.time()
    for k in date_settings:
        sim_set[k] = datetime.combine(
            sim_set[k], min_time
        )

    return DotDict(sim_set)


def read_travel_times(
    path_drive_dist: str = es.PATH_DRIVING_DISTANCE,
    path_drive_time: str = es.PATH_DRIVING_TIMES,
    path_plane_time: str = es.PATH_FLIGHT_TIMES
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, str]]]:
    """Read in expected travelling times between centers"""

    travel_info = []
    for fn in [path_drive_dist, path_drive_time, path_plane_time]:
        travel_info.append(
            pd.read_csv(
                fn,
                dtype={cn.FROM_HOSPITAL: object}
            )
        )

    d_t = reduce(
        lambda d_1, d_2:
        pd.merge(d_1, d_2, on=[cn.FROM_HOSPITAL, cn.TO_CENTER]),
        travel_info
    )

    # Select mode of transportation, based on max dist & time.
    d_t.loc[:, cn.TRAVEL_MODE] = cn.PLANE
    d_t.loc[
        (
            (d_t.loc[:, cn.DRIVING_TIME] < es.MAX_DRIVE_TIME) |
            (d_t.loc[:, cn.DRIVING_DISTANCE] < es.MAX_DRIVE_KM)
        ),
        cn.TRAVEL_MODE
    ] = cn.DRIVE

    # Select the right travel time.
    d_t.loc[:, cn.TRAVEL_TIME] = d_t.loc[:, cn.DRIVING_TIME]
    d_t.loc[
        d_t.loc[:, cn.TRAVEL_MODE] == 'plane',
        cn.TRAVEL_TIME
    ] = d_t.loc[:, cn.TOTAL_TIME_BYPLANE]

    travel_dict = (
        d_t.groupby(cn.FROM_HOSPITAL)[
            [cn.TO_CENTER, cn.TRAVEL_MODE, cn.TRAVEL_TIME]
        ].apply(lambda x: x.set_index(cn.TO_CENTER).to_dict(orient='index'))
        .to_dict()
    )

    return(travel_dict)
