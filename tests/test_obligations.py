import sys
import unittest
from copy import deepcopy
import pandas as pd
import os
from itertools import product
from datetime import datetime
sys.path.append('./')

if True:  # noqa: E402
    from tests.synthetic.SyntheticEntities import SyntheticEntities as SEG
    from simulator.code.entities import CountryObligations
    import simulator.magic_values.elass_settings as es
    import simulator.magic_values.magic_values_rules as mgr
    from simulator.code.read_input_files import read_sim_settings
    from simulator.code.entities import Obligation


DUMMY_DATE = datetime(year=2000, month=1, day=1)


class TestObligations(unittest.TestCase):
    """This tests whether the obligation module works as specified
        in the functional specifications (6 test cases in total).
    """

    # Read in simulation settings
    ss = read_sim_settings(
        os.path.join(
            es.DIR_SIM_SETTINGS,
            'sim_settings_tests.yml'
        )
    )

    # Synthetic entity generator.
    SEG = SEG(sim_set=ss)

    # Define 4 patients
    patient_A = SEG.construct_dummy_patient(1, mgr.AUSTRIA, 'AGATP')
    patient_B = SEG.construct_dummy_patient(2, mgr.BELGIUM, 'BGTP')
    patient_C = SEG.construct_dummy_patient(3, mgr.CROATIA, 'CGTP')
    patient_D = SEG.construct_dummy_patient(4, mgr.GERMANY, 'GBW')

    # Define 4 donors
    donor_A = SEG.construct_dummy_donor(1, mgr.AUSTRIA, 'AGATP')
    donor_B = SEG.construct_dummy_donor(2, mgr.BELGIUM, 'BGTP')
    donor_C = SEG.construct_dummy_donor(3, mgr.CROATIA, 'CGTP')
    donor_D = SEG.construct_dummy_donor(4, mgr.GERMANY, 'GBW')

    # Define 3 dates
    date_1 = datetime(year=2000, month=1, day=1)
    date_2 = datetime(year=2000, month=1, day=2)
    date_3 = datetime(year=2000, month=1, day=3)
    date_4 = datetime(year=2000, month=1, day=4)

    # Create empty set of initial obligations
    empty_initial_obligations = {
        bg: {
            (d, c): [] for (d, c) in product(
                es.OBLIGATION_PARTIES,
                es.OBLIGATION_PARTIES)
        } for bg in es.ALLOWED_BLOODGROUPS
    }

    emptyInitObligations = CountryObligations(
        parties=es.OBLIGATION_PARTIES,
        initial_obligations=empty_initial_obligations,
        verbose=True
    )

    def test_case_1(self):
        """ Test addition of a cycle.
        1. Obligation from A to B.
        2. Obligation from B to A.
        Expected output: Removal of the obligation.
        """
        print("*** Unit test case 1 for obligations. ***")

        obligations = deepcopy(self.emptyInitObligations)

        obl1 = Obligation.from_patient(
            self.patient_A, self.donor_B, self.date_1
            )
        obl2 = Obligation.from_patient(
            self.patient_A, self.donor_B, self.date_2
            )
        obl3 = Obligation.from_patient(
            self.patient_B, self.donor_A, self.date_3
            )

        obligations._add_obligation_return_linkables(
            obl1, creation_time=self.date_1
            )
        obligations._add_obligation_return_linkables(
            obl2, creation_time=self.date_2
            )

        linkable_obligations, linked_obl = obligations.\
            _add_obligation_return_linkables(obl3, self.date_3)

        self.assertIsNone(linked_obl)
        self.assertEqual(linkable_obligations[0].creditor,
                         linkable_obligations[1].debtor)
        self.assertEqual(linkable_obligations[0].debtor,
                         linkable_obligations[1].creditor)
        obligations.print_obligations_for_bg(bloodgroup='A')

    def test_case_2(self):
        """ Test a correct linked obligation is constructed for case 2.
        1. Obligation from A to B.
        2. Obligation from B to C.
        Expected output: Obligation from B to C, at date 2.
        """
        print("\n\n*** Unit test case 2 for obligations. ***")

        obligations = deepcopy(self.emptyInitObligations)

        obl1 = Obligation.from_patient(
            self.patient_A, self.donor_B, self.date_1
            )
        obl2 = Obligation.from_patient(
            self.patient_B, self.donor_C, self.date_2
            )

        obligations._add_obligation_return_linkables(obl1, self.date_1)
        linkable_obligations, linked_obl = obligations.\
            _add_obligation_return_linkables(obl2, self.date_2)

        self.assertEqual(linked_obl.debtor, linkable_obligations[0].debtor)
        self.assertEqual(linked_obl.creditor,
                         linkable_obligations[-1].creditor)
        self.assertEqual(linked_obl.obl_start_date, self.date_2)
        obligations.print_obligations_for_bg(bloodgroup='A')

    def test_case_3(self):
        """ Test correct linked obligation is constructed for case 3.
        1. Obligation from B to C.
        2. Obligation from A to B.
        Expected output: Obligation from B to C, at date 1.
        """
        print("\n\n*** Unit test case 3 for obligations. ***")

        obligations = deepcopy(self.emptyInitObligations)

        obl1 = Obligation.from_patient(
            self.patient_A, self.donor_B, self.date_2
            )
        obl2 = Obligation.from_patient(
            self.patient_B, self.donor_C, self.date_1
            )

        obligations._add_obligation_return_linkables(obl1, self.date_2)
        linkable_obligations, linked_obl = obligations.\
            _add_obligation_return_linkables(obl2, self.date_1)

        self.assertEqual(linked_obl.debtor, linkable_obligations[0].debtor)
        self.assertEqual(linked_obl.creditor,
                         linkable_obligations[-1].creditor)
        self.assertEqual(linked_obl.obl_start_date, self.date_1)

        obligations.print_obligations_for_bg(bloodgroup='A')

    def test_case_4(self):
        """ Test correct linked obligation is constructed for case 4.
        1. Obligation from D to A.
        2. Obligation from C to A.
        3. Obligation from A to B.
        Expected output: Obligation from D to B, at date 3.
        """
        print("\n\n*** Unit test case 4 for obligations. ***")

        obligations = deepcopy(self.emptyInitObligations)

        obl1 = Obligation.from_patient(
            self.patient_D, self.donor_A, self.date_1
            )
        obl2 = Obligation.from_patient(
            self.patient_C, self.donor_A, self.date_2
            )
        obl3 = Obligation.from_patient(
            self.patient_A, self.donor_B, self.date_3
            )

        print("Starting adding obligations")
        obligations._add_obligation_return_linkables(obl1, self.date_1)
        obligations._add_obligation_return_linkables(obl2, self.date_2)
        linkable_obligations, linked_obl = obligations.\
            _add_obligation_return_linkables(obl3, creation_time=self.date_3)
        self.assertEqual(linked_obl.debtor, linkable_obligations[0].debtor)
        self.assertEqual(linked_obl.creditor,
                         linkable_obligations[-1].creditor)
        self.assertEqual(linked_obl.obl_start_date, self.date_3)
        obligations.print_obligations_for_bg(bloodgroup='A')

    def test_case_5(self):
        """ Test correct linked obligation is constructed for case 5
        1. Obligation from B to D.
        2. Obligation from B to C.
        3. Obligation from A to B.
        Expected output: Obligation from A to D, at date 1.
        """
        print("\n\n*** Unit test case 5 for obligations. ***")

        obligations = deepcopy(self.emptyInitObligations)

        obl1 = Obligation.from_patient(
            self.patient_B, self.donor_D, self.date_1
            )
        obl2 = Obligation.from_patient(
            self.patient_B, self.donor_C, self.date_2
            )
        obl3 = Obligation.from_patient(
            self.patient_A, self.donor_B, self.date_3
            )

        obligations._add_obligation_return_linkables(obl1, self.date_1)
        obligations._add_obligation_return_linkables(obl2, self.date_2)

        linkable_obligations, linked_obl = obligations.\
            _add_obligation_return_linkables(obl3, self.date_3)

        self.assertEqual(linked_obl.debtor, linkable_obligations[0].debtor)
        self.assertEqual(linked_obl.creditor,
                         linkable_obligations[-1].creditor)
        self.assertEqual(linked_obl.obl_start_date, self.date_1)
        obligations.print_obligations_for_bg(bloodgroup='A')

    def test_case_6(self):
        """ Test correct linked obligation is constructed for case 6
        1. Obligation from A to B
        2. Obligation from C to D.
        3. Obligation from B to D.
        Expected output: Obligation from A to D, at date 2.
        """
        print("\n\n*** Unit test case 6 for obligations. ***")

        obligations = deepcopy(self.emptyInitObligations)

        obl1 = Obligation.from_patient(
            self.patient_A, self.donor_B, self.date_1
            )
        obl2 = Obligation.from_patient(
            self.patient_C, self.donor_D, self.date_2
            )
        obl3 = Obligation.from_patient(
            self.patient_B, self.donor_C, self.date_3
            )

        obligations._add_obligation_return_linkables(obl1, self.date_1)
        obligations._add_obligation_return_linkables(obl2, self.date_2)
        linkable_obligations, linked_obl = obligations.\
            _add_obligation_return_linkables(obl3, self.date_3)
        self.assertEqual(linked_obl.debtor, linkable_obligations[0].debtor)
        self.assertEqual(linked_obl.creditor,
                         linkable_obligations[-1].creditor)
        self.assertEqual(linked_obl.obl_start_date, self.date_2)
        obligations.print_obligations_for_bg(bloodgroup='A')

    def test_obligations_updating(self):
        """Test whether obligations are correctly updated."""
        obligations = deepcopy(self.emptyInitObligations)

        obligations.update_with_new_obligation(
                    self.patient_A,
                    self.donor_B,
                    self.date_1
                )

        obligations.update_with_new_obligation(
                    self.patient_B,
                    self.donor_C,
                    self.date_2
                )

        new_c = self.donor_C.donor_country
        new_d = self.patient_A.recipient_center

        # Check whether obligation is correctly added
        n_obl = len(
            obligations.obligations[
                self.donor_B.d_bloodgroup
                ][(new_d, new_c)]
        )

        self.assertEqual(n_obl, 1)

    def test_retrieve_obligation_ranks(self):
        """Test whether obligation ranks are correctly updated."""
        obligations = deepcopy(self.emptyInitObligations)
        obligations.update_with_new_obligation(
                    self.patient_A,
                    self.donor_D,
                    self.date_3
                )
        obligations.update_with_new_obligation(
                    self.patient_A,
                    self.donor_C,
                    self.date_2
                )
        obligations.update_with_new_obligation(
                    self.patient_A,
                    self.donor_B,
                    self.date_4
                )
        obligations.update_with_new_obligation(
                    self.patient_A,
                    self.donor_B,
                    self.date_1
                )

        creditor_ranks = obligations.return_obligation_ranks(
            self.patient_A.r_bloodgroup,
            self.patient_A.recipient_center
            )

        # Belgium has the oldest obligation, so the highest rank,
        # then Croatia, then Germany. Check this.
        self.assertEqual(creditor_ranks[self.donor_B.donor_country], 3)
        self.assertEqual(creditor_ranks[self.donor_C.donor_country], 2)
        self.assertEqual(creditor_ranks[self.donor_D.donor_country], 1)
        print(creditor_ranks)


if __name__ == '__main__':
    unittest.main()
