from re import I
import sys
import unittest
import os
from datetime import timedelta, datetime
from itertools import product
import pandas as pd
import numpy as np

sys.path.append('./')

if True:  # noqa: E402
    from simulator.code.PostTransplantPredictor import PostTransplantPredictor
    from simulator.code.entities import Patient, Donor, CountryObligations
    import simulator.magic_values.column_names as cn
    import simulator.magic_values.column_groups as cg
    import simulator.magic_values.elass_settings as es
    from tests.synthetic.SyntheticEntities import SyntheticEntities as SEG
    from simulator.code.read_input_files import \
        read_sim_settings
    from simulator.code.read_input_files import read_travel_times
    from simulator.code.current_elas.CurrentELAS import \
        MatchListCurrentELAS


class TestPosttransplant(unittest.TestCase):
    """Test whether post-transplant survival is accurately predicted
    """

    def verify_posttxp_surv(
            self,
            urgency_code: str = 'T',
            verbosity: int = 2,
            slack: float = 5e-2
    ) -> None:
        """ Test whether post-transplant survival probabilities are
            correctly predicted.
        """

        # Read in simulation settings
        ss = read_sim_settings(
            os.path.join(
                es.DIR_SIM_SETTINGS,
                'sim_settings_tests.yml'
            )
        )

        dict_travel_time = read_travel_times()

        # Set-up obligations
        empty_initial_obligations = {
            bg: {
                (d, c): [] for (d, c) in
                product(es.OBLIGATION_PARTIES, es.OBLIGATION_PARTIES)
            } for bg in es.ALLOWED_BLOODGROUPS
        }
        obligations = CountryObligations(
            parties=es.OBLIGATION_PARTIES,
            initial_obligations=empty_initial_obligations,
            verbose=True
        )

        d_txps = pd.read_csv(
            es.POSTTXP_SURV_TESTPATHS[urgency_code]
            )
        d_txps['date_transplanted'] = d_txps['date_transplanted'].apply(
            lambda x: datetime.strptime(
                x,
                es.DEFAULT_DATE_TIME
            )
        )
        ptp = PostTransplantPredictor(
            offset_ids_transplants=99999,
            seed=14
        )

        n_bg_incompatible = 0
        for id, rcrd in d_txps.iterrows():

            pat = Patient(
                id_recipient=1,
                r_dob=(
                    rcrd['date_transplanted'] -
                    timedelta(days=rcrd[cn.R_MATCH_AGE]*365.24)
                ),
                recipient_country=rcrd[cn.RECIPIENT_COUNTRY],
                recipient_region=rcrd[cn.RECIPIENT_REGION],
                recipient_center=rcrd[cn.RECIPIENT_CENTER],
                bloodgroup=rcrd[cn.R_BLOODGROUP],
                lab_meld=rcrd[cn.MELD_LAB],
                listing_date=rcrd['date_transplanted'] - timedelta(days=100),
                weight=rcrd[cn.R_WEIGHT],
                urgency_code=urgency_code,
                r_aco=False,
                height=rcrd[cn.R_HEIGHT],
                sex=rcrd[cn.PATIENT_SEX],
                sim_set=ss,
                type_e=rcrd[cn.TYPE_E],
                listed_kidney=False,
                time_since_prev_txp=rcrd[cn.TIME_SINCE_PREV_TXP],
                type_retx=(
                    cn.RETRANSPLANT_DURING_SIM if rcrd[cn.IS_RETRANSPLANT]
                    else cn.NO_RETRANSPLANT
                )
            )
            for biom in cg.BIOMARKERS:
                pat.biomarkers[biom] = rcrd[biom]
            pat.set_dialysis(status=rcrd[cn.DIAL_BIWEEKLY])

            pat.meld_scores[cn.SCORE_UNOSMELD] = rcrd[cn.MELD_LAB]

            don = Donor(
                    id_donor=1255,
                    donor_country=rcrd[cn.D_COUNTRY],
                    donor_region=rcrd[cn.D_REGION],
                    donor_center=rcrd[cn.D_ALLOC_CENTER],
                    donor_hospital=int(rcrd[cn.D_HOSPITAL]),
                    bloodgroup=rcrd[cn.D_BLOODGROUP],
                    reporting_date=rcrd['date_transplanted'],
                    weight=rcrd[cn.D_WEIGHT],
                    height=rcrd[cn.D_HEIGHT],
                    donor_dcd=rcrd[cn.D_DCD],
                    first_offer_type=rcrd[cn.TYPE_OFFER],
                    age=rcrd['donor_age'],
                    donor_death_cause_group=rcrd[cn.DONOR_DEATH_CAUSE_GROUP],
                    malignancy=rcrd[cn.D_MALIGNANCY],
                    tumor_history=rcrd[cn.D_TUMOR_HISTORY],
                    marginal=rcrd[cn.D_MARGINAL],
                    donor_marginal_free_text=rcrd[cn.D_MARGINAL_FREE_TEXT],
                    drug_abuse=rcrd[cn.D_DRUG_ABUSE],
                    donor_proc_center=rcrd[cn.D_PROC_CENTER],
                    diabetes=rcrd[cn.D_DIABETES],
                    rescue=rcrd[cn.RESCUE]
                )

            ml = MatchListCurrentELAS(
                    patients=[
                        pat
                        ],
                    donor=don,
                    match_date=rcrd['date_transplanted'],
                    type_offer=rcrd[cn.TYPE_OFFER],
                    obl=obligations,
                    construct_center_offers=False,
                    travel_time_dict=dict_travel_time
                )

            # Print match info.
            if verbosity >= 1:
                print(f'***** Record {id} *******')
                print(f'Original scale param: {rcrd["b"]}')
                print(f'Original shape param: {rcrd["a"]}')
                print(f'Original surv prob: {rcrd["surv_prob"]}')

            if len(ml.return_match_list()) > 0:
                offer = ml.return_match_list()[0]
                if hasattr(offer, '_initialize_acceptance_information'):
                    offer._initialize_acceptance_information()
                    offer.__dict__[cn.TYPE_TRANSPLANTED] = rcrd[cn.TYPE_OFFER]
                    offer._initialize_posttxp_information(ptp)

                surv_prob = ptp.calculate_survival(
                    offer=offer,
                    split=rcrd[cn.GRAFT_SPLIT],
                    time=rcrd['pred_time'],
                    verbosity=verbosity
                )

                assert (surv_prob - rcrd['surv_prob']) <= slack, \
                    f'Survival probability incorrect for test case {id}'
            else:
                n_bg_incompatible += 1

        assert n_bg_incompatible < d_txps.shape[0]/100*5, \
            f'{n_bg_incompatible}/{d_txps.shape[0]} empty ' \
            f'match lists. Due to BG incompatibility?'

    def test_posttxp_T(self) -> None:
        # Test whether post-transplant survival is correctly predicted
        # for T patients
        self.verify_posttxp_surv(
            urgency_code='T',
            verbosity=0,
            slack=0.01
        )

    def test_posttxp_HU(self):
        # Test whether post-transplant survival is correctly predicted
        # for HU patients
        self.verify_posttxp_surv(
            urgency_code='HU',
            verbosity=0,
            slack=0.01
        )


if __name__ == '__main__':
    unittest.main()
