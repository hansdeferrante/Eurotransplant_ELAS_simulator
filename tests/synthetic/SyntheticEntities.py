import sys
from itertools import product
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from datetime import datetime

sys.path.append('./')

if True:  # noqa: E402
    from simulator.code.entities import CountryObligations
    from simulator.magic_values.column_names import MTCH_OBL, RECIPIENT_COUNTRY
    from datetime import timedelta
    from simulator.code.entities import Patient, Donor
    from simulator.magic_values.elass_settings import (
        OBLIGATION_PARTIES, ALLOWED_BLOODGROUPS,
        ET_COUNTRIES, CNTR_OBLIGATION_CNTRIES
    )
    import simulator.magic_values.column_names as cn
    import simulator.magic_values.column_groups as cg


class SyntheticEntities:
    """This class can be used to generate some synthetic patients.
    """

    def __init__(self, sim_set: Dict[str, object]):
        self.dummy_date = datetime(year=2000, month=1, day=1)
        self.sim_set = sim_set

    def construct_dummy_patient(
            self, pat_id: int, cntry: str, center: str,
            dob: datetime = None, bg: str = 'A'
            ) -> Patient:
        """Construct a dummy patient."""
        if dob is None:
            dob = self.dummy_date
        return Patient(
            id_recipient=pat_id, recipient_country=cntry,
            recipient_center=center, bloodgroup=bg, weight=90,
            listing_date=self.dummy_date, urgency_code='T',
            recipient_region='GW', r_dob=dob,
            lab_meld=30, r_aco=0,
            height=np.nan, sim_set=self.sim_set)

    def construct_dummy_donor(
            self, don_id: int, cntry: str, center: str, weight: int = 90,
            bg: str = 'A'
            ) -> Donor:
        "Construct a dummy donor."
        return Donor(
            id_donor=don_id, donor_country=cntry, donor_center=center,
            weight=weight, bloodgroup=bg, reporting_date=self.dummy_date,
            donor_region='GAH', donor_dcd=False,
            first_offer_type=4
            )

    def construct_obligations_from_tuples(
            self, obligation_tuples: List[Tuple[str, str, str, str, str, int]],
            verbose: int = 0
            ) -> CountryObligations:
        """Construct synthetic country obligations, based on
        input tuples."""

        # Create empty set of initial obligations
        empty_initial_obligations = {bg: {
            (d, c): [] for (d, c) in product(
                OBLIGATION_PARTIES,
                OBLIGATION_PARTIES)
        } for bg in ALLOWED_BLOODGROUPS
        }

        obligations = CountryObligations(
            parties=OBLIGATION_PARTIES,
            initial_obligations=empty_initial_obligations,
            verbose=verbose
        )

        for index, tpl in enumerate(obligation_tuples):
            (d_bg, d_cntry, d_cntr, r_cntry, r_cntr, obl_rank) = tpl

            obligations.update_with_new_obligation(
                patient=self.construct_dummy_patient(
                    index, d_cntry, d_cntr, self.dummy_date
                ),
                donor=self.construct_dummy_donor(
                    index, r_cntry, r_cntr, bg=d_bg
                ),
                creation_time=self.dummy_date+timedelta(index)
            )
        return obligations

    def construct_obltuples_from_ml(
            self, df_obl: pd.DataFrame
            ) -> List[Tuple[str, str, str, str, str, int]]:
        """From a match list, return tuples of obligations."""

        assert len(df_obl[cn.ID_MTR].unique()) == 1, \
            f'Expected a single match list'

        # Construct synthetic obligations from match list.
        ctr_obl = df_obl[cn.RECIPIENT_COUNTRY].isin(CNTR_OBLIGATION_CNTRIES)
        df_obl[cn.OBL_CREDITOR] = np.select(
            condlist=[~ctr_obl, ctr_obl],
            choicelist=df_obl.loc[
                :,
                [cn.RECIPIENT_COUNTRY, cn.RECIPIENT_CENTER]
                ].T.to_numpy()
        )

        # Sort by obligation, descendingly.
        df_obl = (
            df_obl[list(cg.OBL_CODES + (cn.OBL_CREDITOR,))]
            .drop_duplicates()
            .sort_values([cn.OBL_CREDITOR, cn.MTCH_OBL])
            .groupby(cn.OBL_CREDITOR)
            .tail(1)
        ).sort_values(cn.MTCH_OBL)

        return df_obl.loc[
            df_obl[cn.MTCH_OBL].gt(0),
            [col for col in cg.OBL_CODES]
            ].to_numpy()

    @staticmethod
    def construct_obltuples_from_ml_records(
            rcrds: List[Dict[str, object]]
            ) -> List[Tuple[str, str, str, str, str, int]]:
        """From a match list, return tuples of obligations."""

        assert len(list(set([x[cn.ID_MTR] for x in rcrds]))) == 1, \
            'Expected a single match list'

        df_obl = pd.DataFrame(
            [
                {
                    k: v for k, v in rcrd.items()
                    if k in cg.OBL_CODES
                }
                for rcrd in rcrds
            ]
        )
        df_obl[cn.OBL_CREDITOR] = np.select(
            condlist=[
                df_obl[cn.RECIPIENT_COUNTRY].isin(CNTR_OBLIGATION_CNTRIES)
                ],
            choicelist=[df_obl[cn.RECIPIENT_CENTER]],
            default=df_obl[cn.RECIPIENT_COUNTRY]
        )

        # Sort by obligation, descendingly.
        df_obl = (
            df_obl[list(cg.OBL_CODES + (cn.OBL_CREDITOR,))]
            .drop_duplicates()
            .sort_values([cn.OBL_CREDITOR, cn.MTCH_OBL])
            .groupby(cn.OBL_CREDITOR)
            .tail(1)
        ).sort_values(cn.MTCH_OBL)

        return df_obl.loc[
            df_obl.loc[:, cn.MTCH_OBL].gt(0),
            [col for col in cg.OBL_CODES]
        ].to_numpy()
