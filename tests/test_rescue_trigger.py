from re import I
import sys
import unittest
import os
from datetime import timedelta, datetime
import pandas as pd
from random import randint

sys.path.append('./')

if True:  # noqa: E402
    from simulator.code.AcceptanceModule import AcceptanceModule
    from simulator.code.entities import Patient, Donor
    import simulator.magic_values.column_names as cn
    import simulator.magic_values.elass_settings as es
    from simulator.code.read_input_files import \
        read_sim_settings, read_travel_times
    from simulator.magic_values.rules import DICT_CENTERS_TO_REGIONS
    from simulator.code.current_elas.CurrentELAS import \
        MatchListCurrentELAS
    from simulator.code.current_elas.CurrentELAS import \
        MatchRecordCurrentELAS
    import simulator.magic_values.magic_values_rules as mgr


class TestCurrentAllocation(unittest.TestCase):
    """Test whether current allocation is done correctly.
    """

    def test_rescue_trigger(self):
        """Test center acceptances
        """

        print(
            'Testing whether acceptance model is'
            ' correct for center/obligations.'
        )

        # Read in simulation settings
        ss = read_sim_settings(
            os.path.join(
                es.DIR_SIM_SETTINGS,
                'sim_settings_tests.yml'
            )
        )
        travel_time_dict = read_travel_times()

        d_test_donors = pd.read_csv('data/test/triggered_rescue.csv')
        acc_module = AcceptanceModule(
            seed=1,
            patient_acc_policy='LR',
            center_acc_policy='LR',
            verbose=0,
            simulate_rescue=True
        )

        dummy_date = datetime(year=2000, month=1, day=1)

        # d_acc_records = d_acc_records.loc[617:,]

        n_bg_incompatible = 0
        for id, rcrd in d_test_donors.iterrows():

            don = Donor(
                    id_donor=1255,
                    donor_country=rcrd[cn.D_COUNTRY],
                    donor_region='Bel_1',
                    donor_center='BANTP',
                    bloodgroup=rcrd[cn.D_BLOODGROUP],
                    reporting_date=dummy_date,
                    weight=rcrd[cn.D_WEIGHT],
                    height=rcrd[cn.D_HEIGHT],
                    donor_dcd=rcrd[cn.D_DCD],
                    first_offer_type=4,
                    age=rcrd['donor_age'],
                    donor_death_cause_group=rcrd[cn.DONOR_DEATH_CAUSE_GROUP],
                    malignancy=rcrd[cn.D_MALIGNANCY],
                    marginal=rcrd[cn.D_MARGINAL],
                    hbcab=rcrd[cn.D_HBCAB],
                    hcvab=rcrd[cn.D_HCVAB],
                    hbsag=rcrd[cn.D_HBSAG],
                    tumor_history=rcrd[cn.D_TUMOR_HISTORY],
                    donor_marginal_free_text=rcrd[cn.D_MARGINAL_FREE_TEXT],
                    drug_abuse=rcrd[cn.D_DRUG_ABUSE],
                    donor_proc_center='BANTP',
                    donor_hospital=81
                )

            pd.set_option('display.max_rows', None)
            phat = acc_module.predict_rescue_prob(
                    don,
                    kth_offer=10,
                    verbose=0
            )

            diff_prob = (phat - (1-rcrd.phat_10))
            assert abs(diff_prob) <= 1e-2, \
                (
                    f'Turndown probability incorrect for test case'
                    f'{id} ({diff_prob})'
                )

            n_offers = acc_module.generate_offers_to_rescue(
                    donor=don,
                    r_prob=phat
            )
            assert n_offers <= 10, \
                (
                    f"At most 10 offer should've been"
                    f"generated for {id} (not {n_offers})"
                )


if __name__ == '__main__':
    unittest.main()
