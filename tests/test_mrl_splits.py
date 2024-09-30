import sys
import unittest
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np
import os

sys.path.append('./')

if True:  # noqa: E402
    from tests.synthetic.SyntheticEntities import SyntheticEntities as SEG
    from simulator.code.entities import Patient, Donor
    import simulator.magic_values.column_names as cn
    import simulator.magic_values.column_groups as cg
    import simulator.magic_values.elass_settings as es
    from simulator.magic_values.rules import \
        RECIPIENT_ELIGIBILITY_TABLES, BG_COMPATIBILITY_TAB, \
        BG_PRIORITY_RULES, MAIN_RANK_TIERS, \
        FIRST_LAYER_TIERS
    from simulator.code.current_elas.CurrentELAS import \
        MatchList, MatchListCurrentELAS
    from simulator.code.read_input_files import \
        read_offerlist, read_sim_settings


class TestSplitMatchLists(unittest.TestCase):
    """Test whether current allocation is done correctly for split offers
    """

    def test_mrl(self, n_to_assess: int = 500):
        """ Test whether match codes are correctly generated for split livers.
        """

        # Read in simulation settings
        ss = read_sim_settings(
            os.path.join(
                es.DIR_SIM_SETTINGS,
                'sim_settings_tests.yml'
            )
        )

        # Read in the real MatchLists
        df_ml = read_offerlist(ss.PATH_OFFERS)

        # Subset to data after 2016 where the donor is from an ET country.
        df_ml = df_ml[df_ml[cn.MATCH_DATE].apply(lambda x: x.year) >= 2016]
        df_ml = df_ml[df_ml[cn.D_COUNTRY].isin(es.ET_COUNTRIES)]

        print('Checking only non-WLiv offers')
        df_ml = df_ml[df_ml[cn.TYPE_OFFER] != es.TYPE_OFFER_WLIV]
        ml_ids = df_ml[cn.ID_MTR].unique()

        # Subsample to n_to_assess random matchlists to check.
        sample = True
        if sample:
            np.random.seed(14)
            df_ml = df_ml[
                df_ml[cn.ID_MTR].isin(
                    np.random.choice(ml_ids, size=n_to_assess),
                )
            ]

        # For each matchlist
        k = 0
        for sel_id, df_sel in df_ml.groupby(cn.ID_MTR):
            k += 1
            if k % 100 == 0:
                print(f'Finished {k} matchlists')
            patient_list = []
            dummy_date = datetime(year=2000, month=1, day=1)

            # Construct patient list
            for rcrd in df_sel.to_dict(orient='records'):
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

            donor = Donor(
                    id_donor=1255,
                    donor_country=rcrd[cn.D_COUNTRY],
                    donor_region=rcrd[cn.D_REGION],
                    donor_center=rcrd[cn.D_ALLOC_CENTER],
                    bloodgroup=rcrd[cn.D_BLOODGROUP],
                    reporting_date=dummy_date,
                    weight=rcrd[cn.D_WEIGHT],
                    age=40,
                    donor_dcd=rcrd[cn.D_DCD],
                    first_offer_type=4
                )

            date_match = df_sel[cn.MATCH_DATE].iloc[0]
            alloc_center = df_sel[cn.D_ALLOC_CENTER].iloc[0]

            seg = SEG(sim_set=ss)

            obl_tuples = seg.construct_obltuples_from_ml(df_sel)
            obligations = seg.construct_obligations_from_tuples(
                obligation_tuples=obl_tuples
            )

            ml = MatchListCurrentELAS(
                patients=patient_list,
                donor=donor,
                match_date=date_match,
                type_offer=pd.unique(df_sel[cn.TYPE_OFFER])[0],
                obl=obligations,
                construct_center_offers=False,
                aggregate_obligations=False,
                alloc_center=alloc_center
                )

            for check_col in [cn.MTCH_TIER, cn.MTCH_LAYER]:

                trans_dict = {
                    rcrd.patient.id_recipient:
                    rcrd.__dict__[check_col]
                    for rcrd in ml.match_list
                }

                if (
                    not df_sel[check_col].equals(
                        df_sel[cn.REC_OFFERED].map(trans_dict.get)
                        )
                ):

                    df_mrl = ml.return_match_df(
                        reorder=False,
                        collapse_mrl=False
                    )

                    df_real = df_sel.set_index(cn.REC_OFFERED).sort_index()
                    df_mrl = df_mrl.set_index(
                        cn.ID_RECIPIENT,
                        drop=False
                        ).sort_index()

                    mismatches = (
                        (
                            df_mrl[check_col].reset_index(drop=True) !=
                            df_real[check_col].reset_index(drop=True)
                        ) & df_mrl[check_col].notna()
                    )

                    if mismatches.any():
                        print(f'Sum mismatches: {mismatches.sum()}')
                        df_mrl[cn.ID_MTR] = sel_id

                        df_mrl.insert(
                            0,
                            'real_' + check_col,
                            value=df_real[check_col].tolist()
                        )
                        print(
                            df_mrl.loc[
                                mismatches,
                                cg.TEST_MRL_COLS +
                                [cn.BG_RULE_COL, cn.ANY_OBL] +
                                ['real_'+check_col]].T
                                )

                        if check_col == cn.MTCH_TIER:
                            # Do not throw errors for Belgian non-residents
                            # (match tier A)
                            assert (
                                df_real.loc[
                                    mismatches,
                                    cn.MTCH_TIER
                                    ] == cn.TIER_A).all(), \
                                f'Error in calculation MRL. ' \
                                f'Inspect the output.'
                        elif check_col == cn.MTCH_LAYER:
                            # Do not throw errors if rounded match MELD is 30
                            # (BG-threshold appears based on unrounded
                            # match meld)
                            assert (
                                (df_real.loc[
                                    mismatches,
                                    cn.MTCH_LAYER_MELD] == 30).all() |
                                (df_real.loc[
                                    mismatches,
                                    cn.MTCH_TIER] == 'A').all()
                            ), \
                                f'Error in calculation MRL.' \
                                f' Inspect the output.'


if __name__ == '__main__':
    unittest.main()
