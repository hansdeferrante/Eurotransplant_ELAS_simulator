import sys
import unittest
import pandas as pd
import numpy as np
import os
from datetime import datetime


sys.path.append('./')

if True:  # noqa: E402
    from tests.synthetic.SyntheticEntities import SyntheticEntities as SEG
    from simulator.code.entities import Patient, Donor
    import simulator.magic_values.column_names as cn
    import simulator.magic_values.elass_settings as es
    from simulator.code.current_elas.CurrentELAS import \
        MatchListCurrentELAS
    from simulator.code.read_input_files import \
        read_offerlist, read_sim_settings


class TestBloodgroups(unittest.TestCase):
    """This tests whether the blood group rules are correctly applied
    """

    def test_acceptability_ml(self, n_to_assess: int = 200):
        """Test whether ML have acceptable donor/recipient combinations.
        """

        # Read in simulation settings
        ss = read_sim_settings(
            os.path.join(
                es.DIR_SIM_SETTINGS,
                'sim_settings_tests.yml'
            )
        )

        df_ml = read_offerlist(ss.PATH_OFFERS)

        df_ml = df_ml[df_ml[cn.D_COUNTRY].isin(es.ET_COUNTRIES)]
        df_ml = df_ml[df_ml[cn.MATCH_DATE].apply(lambda x: x.year) >= 2016]

        ml_ids = df_ml[cn.ID_MTR].unique()

        np.random.seed(14)
        df_ml = df_ml[
            df_ml[cn.ID_MTR].isin(
                np.random.choice(ml_ids, size=n_to_assess),
            )
            ]

        for id_mtr, df_ml_sel in df_ml.groupby(cn.ID_MTR):
            patient_list = []
            dummy_date = datetime(year=2000, month=1, day=1)

            for rcrd in df_ml_sel.to_dict('records'):
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
                    donor_dcd=rcrd[cn.D_DCD],
                    first_offer_type=rcrd[cn.TYPE_OFFER],
                    age=50
                )

            date_match = rcrd[cn.MATCH_DATE]
            alloc_center = rcrd[cn.D_ALLOC_CENTER]

            # Re-construct obligations as they appear in the match list.
            seg = SEG(sim_set=ss)
            obl_tuples = seg.construct_obltuples_from_ml(df_ml_sel)
            obligations = seg.construct_obligations_from_tuples(
                obligation_tuples=obl_tuples
            )

            # Construct MatchList.
            ml = MatchListCurrentELAS(
                patients=patient_list,
                donor=donor,
                match_date=date_match,
                type_offer=pd.unique(df_ml_sel[cn.TYPE_OFFER]),
                obl=obligations,
                construct_center_offers=False,
                aggregate_obligations=False
                )

            # Check whether all are BG compatible.
            df_mrl = ml.return_match_df()

            # Assert everyone on the list is BG compatible.
            assert len(patient_list) == df_mrl.shape[0], \
                f'Patients appear on the ML for {id_mtr} ' \
                f'whom are BG-incompatible per ET rules.'


if __name__ == '__main__':
    unittest.main()
