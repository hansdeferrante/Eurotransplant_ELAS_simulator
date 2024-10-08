#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:33:44 2022

@author: H.C. de Ferrante
"""

from functools import reduce
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
import numpy as np
import os
import pandas as pd
import typing
import gzip
import shutil

from simulator.code.utils import DotDict, round_to_decimals, round_to_int
import simulator.magic_values.elass_settings as es
import simulator.magic_values.column_names as cn
from simulator.magic_values.inputfile_settings import DEFAULT_DMY_HMS_FORMAT

if typing.TYPE_CHECKING:
    from simulator.code import entities
    from simulator.code import AllocationSystem


class SimResults:
    """Class which implements a heapqueue

    Attributes   #noqa
    ----------
    transplantations: List[Dict[str, Any]]
        List of transplantations that occured
    exits: List[Dict[str, Any]]
        List of patient exits
    discards: List[Dict[str, Any]]
        List of discarded donors (not accepted by any)
    path_txp: str
        path for transplantations
    path_exits: str
        path for waitlist exits
    path_dsc: str
        path for discarded organs
    path_ml: str
        path for matchlists.
    ...
    """
    def __init__(
            self,
            cols_to_save_exit: Tuple[str, ...],
            cols_to_save_discard: Tuple[str, ...],
            cols_to_save_patients: Tuple[str, ...],
            sim_set: DotDict
    ):
        self.cols_to_save_exit = set(cols_to_save_exit)
        self.cols_to_save_discard = set(cols_to_save_discard)
        self.cols_to_save_patients = set(cols_to_save_patients)
        self.transplantations = []
        self.posttransplant = []
        self.exits = []
        self.discards = []
        self.match_lists = []

        self.path_txp: str = (
            sim_set.RESULTS_FOLDER + sim_set.PATH_TRANSPLANTATIONS
        )
        self.path_exits: str = sim_set.RESULTS_FOLDER + sim_set.PATH_EXITS
        self.path_dsc: str = sim_set.RESULTS_FOLDER + sim_set.PATH_DISCARDS
        self.path_pat: str = (
            sim_set.RESULTS_FOLDER + sim_set.PATH_FINAL_PATIENT_STATUS
        )
        self.path_ml: str = (
            sim_set.RESULTS_FOLDER + sim_set.PATH_MATCH_LISTS
        )

        self.path_obligations: str = (
            sim_set.RESULTS_FOLDER +
            f'obls_k{sim_set.SEED}_s{sim_set.SEED}.csv'
        )

        self.save_ml: bool = sim_set.SAVE_MATCH_LISTS

    def save_transplantation(
        self,
        pat: 'entities.Patient',
        matchr: 'AllocationSystem.MatchRecord'
    ) -> None:
        """Save a transplantation to the transplantation list."""

        if cn.D_ET_DRI_TRUNC in self.cols_to_save_exit:
            matchr._calculate_truncated_dri()

        # Combined dictionary
        result_dict = matchr.return_match_info(
            cols=self.cols_to_save_exit
        )
        result_dict[cn.TIME_WAITED] = round_to_decimals(
            (
                pat.__dict__[cn.EXIT_DATE] - pat.__dict__[cn.LISTING_DATE]
            ) / timedelta(days=1),
            2
        )
        if pat.__dict__[cn.E_SINCE]:
            result_dict[cn.TIME_IN_E] = round_to_decimals(
                result_dict[cn.TIME_WAITED] - pat.__dict__[cn.E_SINCE],
                2
            )
        else:
            result_dict[cn.TIME_IN_E] = np.nan
        result_dict.update(
            {
                'meld_' + k.lower(): v
                for k, v in pat.meld_scores.items()
            }
        )
        self.transplantations.append(
            result_dict
        )

    def save_posttransplant(
        self,
        date_failure: Optional[datetime],
        date_relist: Optional[datetime],
        cens_date: datetime,
        rereg_id: Optional[int],
        matchr_: 'AllocationSystem.MatchRecord'
    ) -> None:
        """Save relevant post-transplant information"""

        result_dict = {
            cn.ID_DONOR: matchr_.donor.__dict__[cn.ID_DONOR],
            cn.ID_RECIPIENT: matchr_.patient.__dict__[cn.ID_RECIPIENT],
            cn.ID_REREGISTRATION: rereg_id,
            cn.TYPE_TRANSPLANTED: matchr_.__dict__[cn.TYPE_TRANSPLANTED],
            cn.PATIENT_FAILURE_DATE: (
                date_failure if (
                    date_failure and
                    date_failure < cens_date and
                    date_relist is None
                ) else None
            ),
            cn.PATIENT_RELISTING_DATE: (
                date_relist if date_relist and date_relist < cens_date
                else None
            ),
            cn.PATIENT_RELISTING: (
                1 if date_relist and date_relist < cens_date
                else 0 if date_relist
                else 0
            )
        }

        self.posttransplant.append(result_dict)

    def save_exit(self, pat: 'entities.Patient') -> None:
        """Save an exit into the exiting list."""
        result_dict = {
                k: v for k, v in pat.__dict__.items()
                if k in self.cols_to_save_exit
        }

        result_dict.update(
            {
                'meld_' + k.lower(): v
                for k, v in pat.meld_scores.items()
            }
        )

        result_dict.update(
            {
                k: v
                for k, v in pat.biomarkers.items()
                if k in self.cols_to_save_exit
            }
        )

        self.exits.append(
            result_dict
        )

    def save_discard(
            self,
            matchl_: 'AllocationSystem.MatchList'
    ) -> None:
        """Save an exit into the exiting list."""

        # Count number of center offers & any obl.
        n_center_offers = 0
        n_obl = 0
        profile_turndowns = 0
        for offer in matchl_.return_match_list():
            n_obl = n_obl + offer.__dict__.get(cn.MTCH_OBL, 0)
            if type(offer).__name__ == 'CenterOffer':
                n_center_offers += 1
            else:
                if (
                    offer.patient.profile is None or
                    not offer.patient.profile._check_acceptable(
                        offer.donor
                        )
                ):
                    profile_turndowns += 1

        result_dict = reduce(
            lambda a, b: {**a, **b},
            [o.__dict__ for o in [matchl_, matchl_.donor] if o is not None]
            )

        dict_to_return = {
            k: v for k, v in result_dict.items()
            if k in self.cols_to_save_discard
        }

        # Add relevant discard information.
        dict_to_return[cn.N_OFFERS] = len(matchl_.return_match_list())
        dict_to_return[cn.N_OPEN_OBL] = n_obl
        dict_to_return[cn.N_PROFILE_TURNDOWNS] = profile_turndowns
        dict_to_return[cn.N_CENTER_OFFERS] = n_center_offers

        self.discards.append(
            dict_to_return
        )

    def save_match_list(
        self,
        matchl_: 'AllocationSystem.MatchList',
        write_every_n: int = int(100e3)
    ) -> None:
        """Save match list information"""
        self.match_lists.extend(
            matchl_.return_match_info()
        )
        if len(self.match_lists) >= write_every_n:
            self.match_lists_to_file()
            self.match_lists.clear()

    def return_exits(self) -> pd.DataFrame:
        """Return exits (RM) as a pd.DataFrame"""
        data_ = pd.DataFrame.from_records(
            self.exits,
            columns=self.cols_to_save_exit
        )
        cols = [
            col for col in self.cols_to_save_exit if col in data_.columns
        ]
        data_ = data_.loc[:, cols]
        data_.dropna(how='all', axis=1, inplace=True)
        return data_

    def return_discards(self) -> pd.DataFrame:
        """Return discards as a pd.DataFrame"""
        data_ = pd.DataFrame.from_records(
            self.discards,
            columns=self.cols_to_save_discard
        )
        cols = [
            col for col in self.cols_to_save_discard if col in data_.columns
        ]
        data_ = data_.loc[:, cols]
        data_.dropna(how='all', axis=1, inplace=True)
        return data_

    def return_patient_info(
        self,
        patients: Dict[int, 'entities.Patient']
    ) -> pd.DataFrame:
        """Return patient info"""
        data_ = pd.DataFrame.from_records(
            [p.return_dict_with_melds() for p in patients.values()],
            columns=self.cols_to_save_patients
        )
        cols = [
            col for col in self.cols_to_save_patients if col in data_.columns
        ]
        data_.dropna(how='all', axis=1, inplace=True)
        data_ = data_.loc[:, cols]
        data_.loc[:, cn.R_AGE_LISTING] = (
            data_.loc[:, cn.TIME_REGISTRATION] - data_.loc[:, cn.R_DOB]
        ).apply(lambda dt: round_to_decimals(dt / timedelta(days=365), 1))
        return data_

    def return_all_matchlists(self) -> pd.DataFrame:
        """Return all match lists"""
        data_ = pd.DataFrame.from_records(
            self.match_lists,
            columns=es.MATCH_INFO_COLS
        )
        data_ = data_.loc[:, list(es.MATCH_INFO_COLS)]
        id_cols = [c for c in data_.columns if c.startswith('id')]
        data_.loc[:, id_cols] = data_.loc[:, id_cols].astype('Int64')
        data_.dropna(how='all', axis=1, inplace=True)
        return data_

    def return_transplantations(
            self,
            patients: Dict[int, 'entities.Patient'],
            cens_date: datetime,
            save_posttxp: bool = True
    ) -> pd.DataFrame:
        """Return transplantation information as DataFrame."""
        data_ = pd.DataFrame.from_records(
            self.transplantations,
            columns=self.cols_to_save_exit.union(
                es.OUTPUT_COLS_EXIT_CONSTRUCTED
            )
        )
        cols = [
            col for col in self.cols_to_save_exit.union(
                es.OUTPUT_COLS_EXIT_CONSTRUCTED
            ) if col in data_.columns
        ]
        data_ = data_.loc[:, cols]
        data_.dropna(how='all', axis=1, inplace=True)

        if save_posttxp:
            # Add post-transplant information
            # (graft failure & patient failure dates)
            d_posttxp = pd.DataFrame.from_records(
                self.posttransplant
            )
            data_ = data_.merge(
                d_posttxp,
                on=[cn.ID_DONOR, cn.ID_RECIPIENT],
                suffixes=(None, '_posttxp'),
                how='left'
            )

            # Add outcomes for re-registrations (whether retransplanted or not)
            d_post_retx = data_.loc[
                :, [cn.ID_REGISTRATION, cn.MATCH_DATE]
            ].dropna().rename(
                columns={
                    cn.ID_REGISTRATION: cn.ID_REREGISTRATION,
                    cn.MATCH_DATE: cn.DATE_RETRANSPLANT
                }
            )
            data_ = data_.merge(
                d_post_retx,
                on=[cn.ID_REREGISTRATION],
                how='left'
            )

            # Add exit statuses for non-retransplanted patients
            d_sim_exits = self.return_exits().loc[
                :, [cn.ID_REGISTRATION, cn.EXIT_DATE, cn.EXIT_STATUS]
            ].rename(
                columns={cn.ID_REGISTRATION: cn.ID_REREGISTRATION}
            )
            data_ = data_.merge(
                d_sim_exits,
                on=[cn.ID_REREGISTRATION],
                suffixes=(None, '_rereg'),
                how='left'
            )

            # Add final urgency code for relisted patients.
            data_.loc[
                data_.loc[:, cn.ID_REREGISTRATION].notna() &
                data_.loc[:, cn.DATE_RETRANSPLANT].isna() &
                data_.loc[:, cn.EXIT_STATUS_REREG].isna(),
                cn.EXIT_STATUS_REREG
            ] = data_.loc[
                data_.loc[:, cn.ID_REREGISTRATION].notna() &
                data_.loc[:, cn.DATE_RETRANSPLANT].isna() &
                data_.loc[:, cn.EXIT_STATUS_REREG].isna(),
                cn.ID_REREGISTRATION
            ].apply(
                lambda id: patients[id].urgency_code
            )

            # Construct time to cens
            data_.loc[
                :, 'cens_date'
            ] = data_.loc[
                    :, [
                        cn.PATIENT_FAILURE_DATE,
                        cn.DATE_RETRANSPLANT,
                        'exit_date_rereg'
                    ]
            ].min(axis=1, skipna=True).fillna(cens_date)

            data_.loc[
                :, cn.TIME_TO_CENS
            ] = (
                data_.loc[:, 'cens_date'] -
                data_.loc[:, cn.MATCH_DATE]
            ).apply(
                lambda td: td / timedelta(days=1)
            )

            # Construct time-to-patient failure columns
            data_.loc[:, cn.PATIENT_FAILURE] = (
                data_.loc[:, cn.PATIENT_FAILURE_DATE].notna() |
                data_.loc[:, cn.EXIT_STATUS_REREG].isin(['D'])
            ).astype(int)
            data_.loc[
                data_.loc[:, cn.EXIT_STATUS_REREG] == 'D',
                cn.PATIENT_FAILURE_DATE
            ] = data_.loc[
                :, 'exit_date_rereg'
            ]
            data_.loc[:, cn.TIME_TO_PATIENT_FAILURE] = (
                data_.loc[:, cn.PATIENT_FAILURE_DATE] -
                data_.loc[:, cn.MATCH_DATE]
            ).apply(lambda td: td / timedelta(days=1))

            # Construct time-to-retransplant and time-to-rereg columns
            data_.loc[:, cn.TIME_TO_RETX] = (
                data_.loc[:, cn.DATE_RETRANSPLANT] -
                data_.loc[:, cn.MATCH_DATE]
            ).apply(lambda td: td / timedelta(days=1))
            data_.loc[:, cn.TIME_TO_REREG] = (
                data_.loc[:, cn.PATIENT_RELISTING_DATE] -
                data_.loc[:, cn.MATCH_DATE]
            ).apply(lambda td: td / timedelta(days=1))

            # Construct retransplant indicator column
            data_.loc[:, cn.RETRANSPLANTED] = (
                data_.loc[:, cn.DATE_RETRANSPLANT].notna()
            ).astype(int)

            # Construct time-to-event column
            data_.loc[:, cn.EVENT] = (
                data_.loc[:, [cn.RETRANSPLANTED, cn.PATIENT_FAILURE]]
            ).max(axis=1).astype(int)
            data_.loc[:, cn.TIME_TO_EVENT] = (
                data_.loc[:, [cn.TIME_TO_RETX, cn.TIME_TO_PATIENT_FAILURE]]
            ).min(axis=1, skipna=True)

            # Fill unknown with censoring time (administrative only)
            data_.loc[:, cn.TIME_TO_EVENT] = np.ceil(
                data_.loc[:, cn.TIME_TO_EVENT].fillna(
                    data_.loc[:, cn.TIME_TO_CENS]
                )*1000
            )/1000
            data_.loc[:, cn.TIME_TO_REREG] = np.ceil(
                data_.loc[:, cn.TIME_TO_REREG].fillna(
                    data_.loc[:, cn.TIME_TO_CENS]
                )*1000
            )/1000
            data_.loc[:, cn.TIME_TO_PATIENT_FAILURE] = np.ceil(
                data_.loc[:, cn.TIME_TO_PATIENT_FAILURE].fillna(
                    data_.loc[:, cn.TIME_TO_CENS]
                )*1000
            )/1000

        return data_

    def transplantations_to_file(
            self,
            patients: Dict[int, 'entities.Patient'],
            cens_date: datetime,
            save_posttxp: bool = True,
            file: Optional[str] = None
    ) -> None:
        """Save transplantations to file"""
        if file is None:
            file = self.path_txp
        data_ = self.return_transplantations(
            patients=patients,
            cens_date=cens_date,
            save_posttxp=save_posttxp
        )
        outdir = os.path.dirname(file)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        data_.to_csv(
            path_or_buf=file,
            index=False,
            date_format=DEFAULT_DMY_HMS_FORMAT
        )

    def exits_to_file(self, file: Optional[str] = None):
        """Save exits to file"""
        if file is None:
            file = self.path_exits
        data_ = self.return_exits()
        data_.rename(
            columns={cn.URGENCY_REASON: cn.REMOVAL_REASON},
            inplace=True
            )
        outdir = os.path.dirname(file)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        data_.to_csv(
            path_or_buf=file,
            index=False,
            date_format=DEFAULT_DMY_HMS_FORMAT
        )

    def discards_to_file(self, file: Optional[str] = None) -> None:
        """Save patient deaths to file"""
        if file is None:
            file = self.path_dsc
        data_ = self.return_discards()
        outdir = os.path.dirname(file)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        data_.to_csv(
            path_or_buf=file,
            index=False,
            date_format=DEFAULT_DMY_HMS_FORMAT
        )

    def patients_to_file(
        self,
        patients: Dict[int, 'entities.Patient'],
        file: Optional[str] = None
    ) -> None:
        """Save patient deaths to file"""
        if file is None:
            file = self.path_pat
        data_ = self.return_patient_info(
            patients=patients
        )
        outdir = os.path.dirname(file)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        data_.to_csv(
            path_or_buf=file,
            index=False,
            date_format=DEFAULT_DMY_HMS_FORMAT
        )

    def match_lists_to_file(
            self,
            file: Optional[str] = None,
            compression: Optional[str] = None
    ) -> None:
        """Save match lists to file"""
        file = file if file else self.path_ml
        if compression:
            if isinstance(file, str) and not file.endswith(compression):
                file += f'.{compression}'
        data_ = self.return_all_matchlists()
        outdir = os.path.dirname(file)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        data_.to_csv(
            path_or_buf=file,
            mode='a',
            compression=compression,
            index=False,
            date_format=DEFAULT_DMY_HMS_FORMAT
        )

    def obls_to_file(
            self,
            obls: 'entities.CountryObligations',
            file: Optional[str] = None,
            compression: Optional[str] = None
    ) -> None:
        """Save match lists to file"""
        file = file if file else self.path_obligations
        if compression:
            if isinstance(file, str) and not file.endswith(compression):
                file += f'.{compression}'
        data_ = obls._return_generated_obligations()
        outdir = os.path.dirname(file)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        data_.to_csv(
            path_or_buf=file,
            mode='a',
            compression=compression,
            index=False,
            date_format=DEFAULT_DMY_HMS_FORMAT
        )

    def gzip_csv_file(
            self,
            file: str
    ) -> None:
        """Save match lists to file"""
        with open(file, 'rb') as f_in:
            with gzip.open(f'{file}.gzip', 'wb', compresslevel=5) as f_out:
                shutil.copyfileobj(f_in, f_out, length=128*1024)
        os.remove(file)

    def save_outcomes_to_file(
        self,
        patients: Dict[int, 'entities.Patient'],
        cens_date: datetime,
        obligations: 'entities.CountryObligations',
        save_posttxp: bool = True
    ) -> None:
        """Save all outcomes to file"""
        self.discards_to_file()
        self.transplantations_to_file(
            patients=patients,
            cens_date=cens_date,
            save_posttxp=save_posttxp
            )
        self.exits_to_file()
        self.obls_to_file(obligations)
        if patients:
            self.patients_to_file(
                patients
            )
        if self.save_ml:
            self.match_lists_to_file()
            self.gzip_csv_file(
                file=self.path_ml
            )
