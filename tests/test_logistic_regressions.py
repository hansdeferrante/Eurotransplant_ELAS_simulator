from re import I
import sys
import unittest
import os
from datetime import timedelta, datetime
from itertools import product
import pandas as pd
import numpy as np
from random import randint

sys.path.append('./')

if True:  # noqa: E402
    from tests.synthetic.SyntheticEntities import SyntheticEntities as SEG
    from simulator.code.AcceptanceModule import AcceptanceModule
    from simulator.code.entities import Patient, Donor, CountryObligations
    import simulator.magic_values.column_names as cn
    import simulator.magic_values.elass_settings as es
    from tests.synthetic.SyntheticEntities import SyntheticEntities as SEG
    from simulator.code.read_input_files import \
        read_sim_settings
    from simulator.magic_values.rules import DICT_CENTERS_TO_REGIONS
    from simulator.code.current_elas.CurrentELAS import \
        MatchListCurrentELAS
    from simulator.code.current_elas.CurrentELAS import \
        MatchRecordCurrentELAS
    from simulator.code.AllocationSystem import CenterOffer
    from simulator.code.read_input_files import read_travel_times
    import simulator.magic_values.magic_values_rules as mgr


class TestLogisticRegressions(unittest.TestCase):
    """Test whether logistic regressions yield correct acceptance probabilities
    """

    def acceptance_pcd(
            self,
            test_data_file: str,
            separate_huaco_model: bool = True,
            separate_ped_model: bool = True,
            verbose: bool = False
    ) -> None:
        """
        """

        # Read in simulation settings
        ss = read_sim_settings(
            os.path.join(
                es.DIR_SIM_SETTINGS,
                'sim_settings_tests.yml'
            )
        )
        seg = SEG(sim_set=ss)

        dict_travel_time = read_travel_times()

        d_acc_records = pd.read_csv(test_data_file)
        acc_module = AcceptanceModule(
            seed=1,
            patient_acc_policy='LR',
            center_acc_policy='LR',
            separate_huaco_model=separate_huaco_model,
            separate_ped_model=separate_ped_model,
            verbose=0
        )

        dummy_date = datetime(year=2000, month=1, day=1)

        n_bg_incompatible = 0
        for id, rcrd in enumerate(d_acc_records.to_dict(orient='records')):
            if (
                es.LR_TEST_FILES.get('pcd_ped') == test_data_file and
                rcrd[cn.R_MATCH_AGE] >= 16
            ):
                continue

            pat = Patient(
                id_recipient=1,
                r_dob=(
                    dummy_date -
                    timedelta(days=rcrd[cn.R_MATCH_AGE]*365.24)
                ),
                recipient_country=rcrd[cn.RECIPIENT_COUNTRY],
                recipient_region=rcrd[cn.RECIPIENT_REGION],
                recipient_center=rcrd[cn.RECIPIENT_CENTER],
                bloodgroup=rcrd[cn.R_BLOODGROUP],
                lab_meld=rcrd[cn.MELD_LAB],
                listing_date=dummy_date,
                weight=rcrd[cn.R_WEIGHT],
                urgency_code=rcrd[cn.URGENCY_CODE],
                r_aco=rcrd[cn.R_ACO],
                height=np.nan,
                sex=rcrd[cn.PATIENT_SEX],
                sim_set=ss,
                type_e=rcrd[cn.TYPE_E],
                listed_kidney=rcrd[cn.LISTED_KIDNEY],
                type_retx=(
                    cn.RETRANSPLANT if rcrd['retransplant'] else
                    cn.NO_RETRANSPLANT
                )
            )

            pat.meld_scores[cn.SCORE_UNOSMELD] = rcrd[cn.MELD_LAB]

            don = Donor(
                    id_donor=1255,
                    donor_country=rcrd[cn.D_COUNTRY],
                    donor_region=rcrd[cn.D_REGION],
                    donor_center=rcrd[cn.D_ALLOC_CENTER],
                    bloodgroup=rcrd[cn.D_BLOODGROUP],
                    reporting_date=dummy_date,
                    weight=rcrd[cn.D_WEIGHT],
                    height=rcrd[cn.D_HEIGHT],
                    donor_dcd=rcrd[cn.D_DCD],
                    first_offer_type=rcrd[cn.TYPE_OFFER],
                    age=rcrd['donor_age'],
                    donor_death_cause_group=rcrd[cn.DONOR_DEATH_CAUSE_GROUP],
                    malignancy=rcrd[cn.D_MALIGNANCY],
                    tumor_history=rcrd[cn.D_TUMOR_HISTORY],
                    donor_marginal_free_text=rcrd[cn.D_MARGINAL_FREE_TEXT],
                    drug_abuse=rcrd[cn.D_DRUG_ABUSE],
                    donor_proc_center=rcrd[cn.D_PROC_CENTER],
                    donor_hospital=rcrd[cn.D_HOSPITAL]
                )

            obl_tuples = seg.construct_obltuples_from_ml_records(
                [rcrd]
            )

            obligations = seg.construct_obligations_from_tuples(
                obligation_tuples=obl_tuples
            )

            ml = MatchListCurrentELAS(
                    patients=[
                        pat
                        ],
                    donor=don,
                    match_date=dummy_date,
                    type_offer=rcrd[cn.TYPE_OFFER],
                    obl=obligations,
                    construct_center_offers=True,
                    aggregate_obligations=False,
                    travel_time_dict=dict_travel_time
                )

            # Check for patient offer whether prob is correct.
            if len(ml.return_match_list()) > 0:
                offer = ml.return_match_list()[0]
                offer._initialize_acceptance_information()
                assert (
                    abs(
                        acc_module._calc_pcd_select(
                            offer.eligible_patients[0],
                            verbose=verbose
                        ) - rcrd['phat']
                    )
                ) <= 1e-3, f'Turndown probability incorrect for test case {id}'
            else:
                n_bg_incompatible += 1

        assert n_bg_incompatible < d_acc_records.shape[0]/100*10, \
            f'{n_bg_incompatible}/{d_acc_records.shape[0]} empty ' \
            f'match lists. Due to BG incompatibility?'

    def test_selected_patient(self):
        """Test whether logistic regression for selecting patient
             center is OK
        """

        for test_settings in (
            'pcd_ped',
            'pcd_adult'
        ):
            print(f'Testing acceptance probabilities  for {test_settings}')
            self.acceptance_pcd(
                es.LR_TEST_FILES.get(test_settings),
                verbose=False
            )

    def test_acceptance_centers(self):
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
            verbose=False
        )

        # Read in travel times
        dict_travel_time = read_travel_times()

        d_acc_records = pd.read_csv('data/test/acceptance_cd.csv')

        acc_module = AcceptanceModule(
            seed=1,
            patient_acc_policy='LR',
            center_acc_policy='LR',
            separate_huaco_model=ss.SEPARATE_HUACO_ACCEPTANCE,
            separate_ped_model=ss.SEPARATE_PED_ACCEPTANCE,
            verbose=0
        )

        dummy_date = datetime(year=2000, month=1, day=1)

        # d_acc_records = d_acc_records.loc[617:,]

        n_bg_incompatible = 0
        for id, rcrd in d_acc_records.iterrows():

            # Construct obligations
            seg = SEG(sim_set=ss)
            obl_tuples = seg.construct_obltuples_from_ml_records(
                [rcrd]
            )
            obligations = seg.construct_obligations_from_tuples(
                obligation_tuples=obl_tuples
            )

            pat = Patient(
                id_recipient=1,
                r_dob=dummy_date - timedelta(days=rcrd[cn.R_MATCH_AGE]*365.24),
                recipient_country=rcrd[cn.RECIPIENT_COUNTRY],
                recipient_region=DICT_CENTERS_TO_REGIONS.get(
                    rcrd[cn.RECIPIENT_CENTER]
                ),
                recipient_center=rcrd[cn.RECIPIENT_CENTER],
                bloodgroup=rcrd[cn.D_BLOODGROUP],
                lab_meld=20,
                listing_date=dummy_date,
                weight=70,
                urgency_code='T',
                r_aco=False,
                height=np.nan,
                sex='Male',
                sim_set=ss
            )

            don = Donor(
                    id_donor=1255,
                    donor_country=rcrd[cn.D_COUNTRY],
                    donor_region=rcrd[cn.D_REGION],
                    donor_center=rcrd[cn.D_ALLOC_CENTER],
                    donor_proc_center=rcrd[cn.D_PROC_CENTER],
                    bloodgroup=rcrd[cn.D_BLOODGROUP],
                    reporting_date=dummy_date,
                    weight=rcrd[cn.D_WEIGHT],
                    height=rcrd[cn.D_HEIGHT],
                    donor_dcd=rcrd[cn.D_DCD],
                    first_offer_type=rcrd[cn.TYPE_OFFER],
                    age=rcrd['donor_age'],
                    donor_death_cause_group=rcrd[cn.DONOR_DEATH_CAUSE_GROUP],
                    malignancy=rcrd[cn.D_MALIGNANCY],
                    tumor_history=rcrd[cn.D_TUMOR_HISTORY],
                    donor_marginal_free_text=rcrd[cn.D_MARGINAL_FREE_TEXT],
                    donor_hospital=rcrd[cn.D_HOSPITAL],
                    drug_abuse=rcrd[cn.D_DRUG_ABUSE]
                )

            ml = MatchListCurrentELAS(
                    patients=[
                        pat
                        ],
                    donor=don,
                    match_date=dummy_date,
                    type_offer=rcrd[cn.TYPE_OFFER],
                    obl=obligations,
                    construct_center_offers=True,
                    travel_time_dict=dict_travel_time
                )

            pd.set_option('display.max_rows', None)

            # Check for patient offer whether prob is correct.
            if len(ml.return_match_list()) > 0:
                offer = ml.return_match_list()[0]
                if hasattr(offer, '_initialize_acceptance_information'):
                    offer._initialize_acceptance_information()

                offer.__dict__[cn.PROFILE_COMPATIBLE] = (
                    rcrd[cn.PROFILE_COMPATIBLE]
                )

                assert (
                    abs(
                        acc_module.calculate_center_offer_accept(
                            offer,
                            verbose=0
                            ) - rcrd.phat
                        )
                    ) <= 1e-3, \
                    f'Turndown probability incorrect for test case {id}'
            else:
                n_bg_incompatible += 1

        assert n_bg_incompatible < d_acc_records.shape[0]/100, \
            f'{n_bg_incompatible}/{d_acc_records.shape[0]} empty match lists'

    def acceptance_recipients_test(
            self,
            test_data_file: str,
            separate_huaco_model: bool = True,
            separate_ped_model: bool = True,
            verbose: bool = False
    ) -> None:
        """ Test whether offer acceptance probabilities are
            correctly calculated at the recipient level
        """

        # Read in simulation settings
        ss = read_sim_settings(
            os.path.join(
                es.DIR_SIM_SETTINGS,
                'sim_settings_tests.yml'
            )
        )
        seg = SEG(sim_set=ss)

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

        d_acc_records = pd.read_csv(test_data_file)
        acc_module = AcceptanceModule(
            seed=1,
            patient_acc_policy='LR',
            center_acc_policy='LR',
            separate_huaco_model=separate_huaco_model,
            separate_ped_model=separate_ped_model,
            verbose=0,
            simulate_random_effects=False
        )

        dummy_date = datetime(year=2000, month=1, day=1)

        n_bg_incompatible = 0
        for id, rcrd in enumerate(d_acc_records.to_dict(orient='records')):

            pat = Patient(
                id_recipient=randint(0, 10000),
                id_reg=randint(0, 10000),
                r_dob=(
                    dummy_date -
                    timedelta(days=rcrd[cn.R_MATCH_AGE]*365.24)
                ),
                recipient_country=rcrd[cn.RECIPIENT_COUNTRY],
                recipient_region=rcrd[cn.RECIPIENT_REGION],
                recipient_center=rcrd[cn.RECIPIENT_CENTER],
                bloodgroup=rcrd[cn.R_BLOODGROUP],
                lab_meld=rcrd[cn.MELD_LAB],
                listing_date=dummy_date,
                weight=rcrd[cn.R_WEIGHT],
                urgency_code=rcrd[cn.URGENCY_CODE],
                r_aco=rcrd[cn.R_ACO],
                height=rcrd.get(cn.R_HEIGHT, np.nan),
                sex=rcrd[cn.PATIENT_SEX],
                sim_set=ss,
                type_e=rcrd[cn.TYPE_E],
                listed_kidney=rcrd[cn.LISTED_KIDNEY],
                type_retx=(
                    cn.NO_RETRANSPLANT if int(rcrd['retransplant']) == 0
                    else cn.RETRANSPLANT
                )
            )

            pat.meld_scores[cn.SCORE_UNOSMELD] = rcrd[cn.MELD_LAB]

            don = Donor(
                    id_donor=1255,
                    donor_country=rcrd[cn.D_COUNTRY],
                    donor_region=rcrd[cn.D_REGION],
                    donor_center=rcrd[cn.D_ALLOC_CENTER],
                    bloodgroup=rcrd[cn.D_BLOODGROUP],
                    reporting_date=dummy_date,
                    weight=rcrd[cn.D_WEIGHT],
                    height=rcrd[cn.D_HEIGHT],
                    donor_dcd=rcrd[cn.D_DCD],
                    first_offer_type=rcrd[cn.TYPE_OFFER],
                    age=rcrd['donor_age'],
                    donor_death_cause_group=rcrd[cn.DONOR_DEATH_CAUSE_GROUP],
                    malignancy=rcrd[cn.D_MALIGNANCY],
                    tumor_history=rcrd[cn.D_TUMOR_HISTORY],
                    donor_marginal_free_text=rcrd[cn.D_MARGINAL_FREE_TEXT],
                    drug_abuse=rcrd[cn.D_DRUG_ABUSE],
                    donor_proc_center=rcrd[cn.D_PROC_CENTER],
                    donor_hospital=rcrd[cn.D_HOSPITAL]
                )

            obl_tuples = seg.construct_obltuples_from_ml_records(
                [rcrd]
            )

            obligations = seg.construct_obligations_from_tuples(
                obligation_tuples=obl_tuples
            )

            ml = MatchListCurrentELAS(
                    patients=[
                        pat
                        ],
                    donor=don,
                    match_date=dummy_date,
                    type_offer=rcrd[cn.TYPE_OFFER],
                    obl=obligations,
                    construct_center_offers=False,
                    travel_time_dict=dict_travel_time,
                    aggregate_obligations=False
                )

            # Check for patient offer whether prob is correct.
            if len(ml.return_match_list()) > 0:
                offer = ml.return_match_list()[0]

                if cn.PROFILE_COMPATIBLE in rcrd:
                    offer.__dict__[cn.PROFILE_COMPATIBLE] = (
                        rcrd[cn.PROFILE_COMPATIBLE]
                    )

                offer._initialize_acceptance_information()
                assert (
                    abs(
                        acc_module.calculate_prob_patient_accept(
                            offer,
                            verbose=verbose
                        ) - rcrd['phat']
                    )
                ) <= 1e-2, f'Turndown probability incorrect for test case {id}'
            else:
                n_bg_incompatible += 1

        assert n_bg_incompatible < d_acc_records.shape[0]/100*10, \
            f'{n_bg_incompatible}/{d_acc_records.shape[0]} empty ' \
            f'match lists. Due to BG incompatibility?'

    def test_acceptance_patients(self):
        """Test whether acceptance in regular allocation is OK"""

        for test_settings in (
            'rd_adult_huaco', 'rd_adult_reg', 'rd_ped_reg', 'rd_ped_huaco'
        ):
            print(f'Testing acceptance probabilities  for: {test_settings}')
            self.acceptance_recipients_test(
                es.LR_TEST_FILES.get(test_settings),
                verbose=0
            )

    def test_lr_splits(self):
        """ Test whether graft split probabilities are predicted correctly.
        """
        print('Testing whether acceptance model is correct for splits')

        # Read in simulation settings
        ss = read_sim_settings(
            os.path.join(
                es.DIR_SIM_SETTINGS,
                'sim_settings_tests.yml'
            )
        )
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

        d_acc_records = pd.read_csv('data/test/test_cases_split.csv')

        acc_module = AcceptanceModule(
            seed=14,
            patient_acc_policy='LR',
            center_acc_policy='LR',
            separate_huaco_model=ss.SEPARATE_HUACO_ACCEPTANCE,
            separate_ped_model=ss.SEPARATE_PED_ACCEPTANCE
        )

        dummy_date = datetime(year=2000, month=1, day=1)

        n_bg_incompatible = 0
        for id, rcrd in d_acc_records.iterrows():

            # Construct obligations
            rcrd[cn.RECIPIENT_CENTER] = 'BLMTP'

            pat = Patient(
                id_recipient=1,
                r_dob=dummy_date - timedelta(days=50*365.24),
                recipient_country=mgr.BELGIUM,
                recipient_region='',
                recipient_center='BLMTP',
                bloodgroup='A',
                lab_meld=20,
                listing_date=dummy_date,
                weight=rcrd[cn.R_WEIGHT],
                urgency_code='T',
                r_aco=0,
                height=np.nan,
                sex='Male',
                sim_set=ss
            )

            don = Donor(
                    id_donor=1255,
                    donor_country=mgr.BELGIUM,
                    donor_region=mgr.BELGIUM,
                    donor_center='BBRTP',
                    bloodgroup='A',
                    reporting_date=dummy_date,
                    weight=rcrd[cn.D_WEIGHT],
                    height=180,
                    donor_dcd=False,
                    first_offer_type=rcrd[cn.TYPE_OFFER],
                    age=rcrd['donor_age'],
                    donor_death_cause_group='Anoxia',
                    malignancy=False,
                    tumor_history=False,
                    donor_marginal_free_text=False
                )

            ml = MatchListCurrentELAS(
                    patients=[pat],
                    donor=don,
                    match_date=dummy_date,
                    type_offer=rcrd[cn.TYPE_OFFER],
                    obl=obligations,
                    construct_center_offers=True
                )

            pd.set_option('display.max_rows', None)

            # Check for patient offer whether prob is correct.
            if len(ml.return_match_list()) > 0:
                offer = ml.return_match_list()[0]
                phat = acc_module.calculate_prob_split(
                            offer,
                            verbose=2
                        )
                assert ((phat == 0) | (abs(phat - rcrd.phat) <= 1e-2)), \
                    f'Turndown probability incorrect for test case {id}'
            else:
                n_bg_incompatible += 1

        assert n_bg_incompatible < d_acc_records.shape[0]/100, \
            f'{n_bg_incompatible}/{d_acc_records.shape[0]} empty match lists'


if __name__ == '__main__':
    unittest.main()
