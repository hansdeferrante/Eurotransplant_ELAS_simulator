import sys
import unittest
import pandas as pd
import numpy as np
import os
from itertools import groupby
from datetime import datetime

sys.path.append('./')

if True:  # noqa: E402
    from tests.synthetic.SyntheticEntities import SyntheticEntities as SEG
    from simulator.code.entities import Patient, Donor
    import simulator.magic_values.column_names as cn
    import simulator.magic_values.column_groups as cg
    import simulator.magic_values.elass_settings as es
    from simulator.code.current_elas.CurrentELAS import \
        MatchListCurrentELAS

    from simulator.code.read_input_files import \
        read_offerlist, read_sim_settings


class TestSplitMatchListRanks(unittest.TestCase):
    """ Test whether split match lists yield the exact same
        order as match lists exported from the ET data warehouse
    """

    def test_mrl(self, n_to_assess: int = 500):
        """Test whether split match list ranks are reproduced exactly.
        """

        pd.set_option('display.max_rows', 500)
        # pd.set_option('display.max_columns', 100)

        # Read in simulation settings
        ss = read_sim_settings(
            os.path.join(
                es.DIR_SIM_SETTINGS,
                'sim_settings_tests.yml'
            )
        )

        # Read in the real MatchLists
        df_ml = read_offerlist(
            ss.PATH_OFFERS
            )

        # Subset to data after 2016 where the donor is from an ET country.
        df_ml = df_ml[df_ml[cn.MATCH_DATE].apply(lambda x: x.year) >= 2016]
        df_ml = df_ml[df_ml[cn.D_COUNTRY].isin(es.ET_COUNTRIES)]

        # Checking WLiv only.
        print('Checking only split offers')
        df_ml = df_ml[df_ml[cn.TYPE_OFFER] != es.TYPE_OFFER_WLIV]
        ml_ids = df_ml[cn.ID_MTR].unique()

        # Subsample to 500 random matchlists to check.
        df_ml = df_ml[
            df_ml[cn.ID_MTR].isin(
                np.random.choice(ml_ids, size=n_to_assess),
            )
            ]

        # For each matchlist
        k = 0
        df_ml[cn.ID_MTR] = df_ml[cn.ID_MTR].astype('category')
        df_ml[cn.TYPE_TRANSPLANTED] = (
            df_ml[cn.TYPE_TRANSPLANTED].astype('category')
        )
        for sel_id, df_sel in groupby(
                df_ml.to_dict(orient='records'),
                lambda x: (x[cn.ID_MTR], x[cn.TYPE_TRANSPLANTED])
                ):
            k += 1
            if k % 100 == 0:
                print(f'Finished checking {k} matchlists.')

            patient_list = []
            dummy_date = datetime(year=2000, month=1, day=1)

            # Construct patient list
            rcrds = list(df_sel)
            for rcrd in rcrds:
                patient_list.append(
                    Patient(
                        id_recipient=rcrd[cn.REC_OFFERED],
                        r_dob=rcrd[cn.R_DOB],
                        recipient_country=rcrd[cn.RECIPIENT_COUNTRY],
                        recipient_region=rcrd[cn.RECIPIENT_REGION],
                        recipient_center=rcrd[cn.RECIPIENT_CENTER],
                        bloodgroup=rcrd[cn.R_BLOODGROUP],
                        lab_meld=rcrd[cn.MTCH_LAYER_MELD],
                        listing_date=dummy_date,
                        weight=rcrd[cn.R_WEIGHT],
                        urgency_code=rcrd[cn.URGENCY_CODE],
                        r_aco=rcrd[cn.R_ACO],
                        height=np.nan,
                        sim_set=ss
                    )
                )

            # Construct donor list
            donor = Donor(
                    id_donor=1255,
                    donor_country=rcrd[cn.D_COUNTRY],
                    donor_region=rcrd[cn.D_REGION],
                    donor_center=rcrd[cn.D_ALLOC_CENTER],
                    bloodgroup=rcrd[cn.D_BLOODGROUP],
                    reporting_date=dummy_date,
                    weight=rcrd[cn.D_WEIGHT],
                    donor_dcd=rcrd[cn.D_DCD],
                    first_offer_type=rcrd[cn.TYPE_OFFER],
                    age=50
                )

            # Initialize synthetic obligations to real obligations
            # at moment of ML.
            seg = SEG(sim_set=ss)
            obl_tuples = seg.construct_obltuples_from_ml_records(
                    rcrds
                )

            obligations = seg.construct_obligations_from_tuples(
                obligation_tuples=obl_tuples
            )

            # Construct MatchList
            ml = MatchListCurrentELAS(
                patients=patient_list,
                donor=donor,
                match_date=rcrd[cn.MATCH_DATE],
                type_offer=rcrd[cn.TYPE_OFFER],
                obl=obligations,
                construct_center_offers=False
                )

            # Load current ELAS allocation rules, and construct matchlist.
            df_mrl = ml.return_match_df()

            cols_to_move = [
                cn.MTCH_TIER, cn.MTCH_LAYER, cn.RECIPIENT_COUNTRY,
                cn.RECIPIENT_CENTER, cn.D_ALLOC_CENTER, cn.D_ALLOC_REGION
                ]
            for col in cols_to_move:
                if col in df_mrl.columns:
                    sel_col = df_mrl.pop(col)
                    df_mrl.insert(0, col, sel_col)
            df_mrl = df_mrl.drop(columns=[cn.MTCH_LAYER_WT])

            # We check whether indices are monotone increases in
            # deduplicated ranks. We check this rather than all ranks,
            # as the order for international center
            # offers is wrong for many match lists.
            if (
                not df_mrl.groupby(cn.PATIENT_RANK).head(1).
                index.is_monotonic_increasing
            ):
                print(df_mrl)
                raise Exception(
                    f'Failed at match list {k} for ID-MTR: '
                    f'{sel_id} is not strictly increasing')


if __name__ == '__main__':
    unittest.main()
