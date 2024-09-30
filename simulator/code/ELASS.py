#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 16-08-2022

@author: H.C. de Ferrante
"""

import os
from itertools import product
from datetime import timedelta, datetime
from typing import Optional, Dict, List, ValuesView
from collections import defaultdict
import time
import typing

from simulator.code.utils import DotDict, round_to_decimals, round_to_int
from simulator.code.current_elas.CurrentELAS import \
    MatchListCurrentELAS, MatchRecordCurrentELAS
from simulator.code.AcceptanceModule import AcceptanceModule
from simulator.code.PostTransplantPredictor import PostTransplantPredictor
from simulator.code.EventQueue import EventQueue
from simulator.code.Event import Event
from simulator.code.read_input_files import read_travel_times
from simulator.code.load_entities import \
        preload_profiles, preload_status_updates, load_patients, \
        load_donors, preload_aco_statuses, load_obligations, \
        load_retransplantations, preload_disease_groups
from simulator.code.entities import (
    ExceptionSystem, CountryObligations, Donor, Patient
)
from simulator.code.SimResults import SimResults
import simulator.magic_values.column_names as cn
import simulator.magic_values.column_groups as cg
import simulator.magic_values.elass_settings as es
from simulator.magic_values.rules import (
    BLOOD_GROUP_INCOMPATIBILITY_DICT, DEFAULT_MELD_ETCOMP_THRESH
)
from simulator.code.AllocationSystem import CenterOffer

import simulator.magic_values.rules as r

if typing.TYPE_CHECKING:
    from simulator.code import AllocationSystem


class ActiveList:
    """
        Class which tracks actively listed patients, per country.
        The ordering of the list is based on HU/ACO, locality,
        then MELD. Retrieving patients from the active lists
        prevents having to iterate over all patients, and speeds up
        sorting.
    """
    def __init__(self, init_patients):
        self._active_lists = {
            cntry: {
                id: pat
                for id, pat in init_patients.items()
                if pat.active
            }
            for cntry in es.ET_COUNTRIES_OR_OTHER
        }
        self.resort_lists()

    def get_active_list(self, priority_country: str) -> ValuesView[Patient]:
        """Returning patients, ordered with priority for priority_country"""
        return self._active_lists[priority_country].values()

    def print_active_list(self, priority_country: str) -> None:
        for pat in self.get_active_list(priority_country):
            print(
                list(
                    pat.__dict__[attr]
                    for attr in
                    (cn.PATIENT_IS_HU, cn.R_ACO, cn.RECIPIENT_COUNTRY)) + [
                        pat.meld_nat_match, pat.meld_int_match,
                        pat.meld_scores[pat.sim_set.LAB_MELD],
                        pat.meld_scores[cn.DM]
                    ]
            )

    def add_to_active_list(self, identifier: int, pat: Patient) -> None:
        """Adding an identifier to the active list"""
        for active_list in self._active_lists.values():
            active_list[identifier] = pat

    def pop_from_active_lists(self, identifier: int) -> None:
        """Remove a patient identifier from active list"""
        for active_list in self._active_lists.values():
            k = active_list.pop(identifier, None)
            if k is None:
                return None

    def is_active(self, identifier: int) -> bool:
        for active_list in self._active_lists.values():
            if identifier in active_list:
                return True
            else:
                return False
        return False

    def resort_lists(self):
        for prio_cntry, active_list in self._active_lists.items():

            # Sort dictionary by patient (i.e. match MELD)
            self._active_lists[prio_cntry] = dict(
                sorted(
                    active_list.items(),
                    key=lambda x: (
                        not x[1].__dict__[cn.PATIENT_IS_HU],
                        not x[1].__dict__[cn.R_ACO],
                        not x[1].__dict__[cn.RECIPIENT_COUNTRY] == prio_cntry,
                        (
                            -x[1].meld_nat_match if
                            x[1].__dict__[cn.RECIPIENT_COUNTRY] == prio_cntry
                            else -x[1].meld_int_match
                        )
                    )
                )
            )


class ELASS:
    """Class which implements an ELASS simulator

    Attributes   #noqa
    ----------
    patients: List[Patient]
        list of patients
    sim_set: DotDict
        simulation settings
    max_n_patients: Optional[int]
        maximum number of patients to read in
    max_n_statuses: Optional[int]
        maximum number of statuses to read in
    sim_time: float
        simulation time
    ...
    """
    def __init__(
        self,
        sim_set: DotDict,
        max_n_patients: Optional[int] = None,
        verbose: Optional[bool] = True
    ):

        # Read in simulation settings
        self.sim_set = sim_set
        self.verbose = verbose

        self.sim_rescue = sim_set.get('SIMULATE_RESCUE', False)
        self.allow_discards = sim_set.get('ALLOW_DISCARDS', False)

        # Save whether to aggregate obligations
        if sim_set.AGGREGATE_OBLIGATIONS:
            self.aggregate_obl = sim_set.AGGREGATE_OBLIGATIONS
        else:
            self.aggregate_obl = False

        # Read in travel time dictionary
        self.dict_travel_time = read_travel_times()

        # Set up ET compatiblity threshold
        self.meld_etcomp_thresh = int(
            sim_set.get(
                'MELD_ETCOMP_THRESH',
                DEFAULT_MELD_ETCOMP_THRESH
            )
        )

        # Set up times
        self.sim_time = 0

        # Keep track of number of obligations generated
        self.n_obligations = 0

        # Set up (empty) event queue
        self.event_queue = EventQueue()

        # Read in exception system
        self.se_system = ExceptionSystem(self.sim_set)

        # Check whether to allow discards
        self.allow_discards = sim_set.get('ALLOW_DISCARDS', False)

        self.max_sim_time = (
            (
                self.sim_set.SIM_END_DATE - self.sim_set.SIM_START_DATE
            ) / timedelta(days=1)
        )

        # Set-up obligations
        empty_initial_obligations = {
            bg: {
                (d, c): [] for (d, c) in
                product(es.OBLIGATION_PARTIES, es.OBLIGATION_PARTIES)
            } for bg in es.ALLOWED_BLOODGROUPS
        }
        if sim_set.PATH_OBLIGATIONS:
            init_obligations = load_obligations(
                sim_set=self.sim_set,
                empty_obl_dict=empty_initial_obligations
            )
        else:
            init_obligations = empty_initial_obligations
            print('No known obligation path. Starting with empty obligations.')

        self.obligations = CountryObligations(
            parties=es.OBLIGATION_PARTIES,
            initial_obligations=init_obligations,
            verbose=bool(sim_set.VERBOSITY_OBLIGATIONS)
        )

        # Initialize the acceptance module
        self.acceptance_module = AcceptanceModule(
            seed=self.sim_set.SEED,
            center_acc_policy=str(sim_set.CENTER_ACC_POLICY),
            patient_acc_policy=str(sim_set.PATIENT_ACC_POLICY),
            separate_huaco_model=bool(sim_set.SEPARATE_HUACO_ACCEPTANCE),
            separate_ped_model=bool(sim_set.SEPARATE_PED_ACCEPTANCE),
            center_offer_patient_method=str(
                sim_set.PATIENT_CENTER_OFFER_ASSIGNMENT
                ),
            simulate_rescue=self.sim_rescue,
            simulate_random_effects=sim_set.SIMULATE_RANDOM_EFFECTS,
            simulate_joint_random_effects=sim_set.JOINT_RANDOM_EFFECTS,
            re_variance_components=sim_set.VARCOMPS_RANDOM_EFFECTS
        )

        if self.verbose:
            print('Loaded acceptance module')

        # Load patients, and initialize a DataFrame with
        # blood group compatibilities.
        if self.verbose:
            print('Loading patient registrations, statuses, ACO, and profiles')
        self.patients = load_patients(
            sim_set=self.sim_set,
            nrows=max_n_patients
        )

        # Load donors
        self.donors = load_donors(
            sim_set=self.sim_set
        )

        # Initialize random effects for known levels.
        self.acceptance_module._initialize_random_effects(
            re_levels={
                cn.ID_REGISTRATION: set(
                    p.__dict__[cn.ID_REGISTRATION]
                    for p in self.patients.values()
                ),
                cn.RECIPIENT_CENTER: set(
                    p.__dict__[cn.RECIPIENT_CENTER]
                    for p in self.patients.values()
                ),
                cn.ID_DONOR: set(
                    d.__dict__[cn.ID_DONOR]
                    for d in self.donors.values()
                ),
            }
        )

        # Preload status updates
        preload_status_updates(
            patients=self.patients,
            sim_set=sim_set,
            se_sys=self.se_system
        )

        # Preload ACO statuses
        preload_aco_statuses(
            patients=self.patients,
            sim_set=sim_set
        )
        preload_disease_groups(
            patients=self.patients,
            sim_set=sim_set
        )

        # Preload profile information
        preload_profiles(
            patients=self.patients,
            sim_set=sim_set
        )

        # If simulating re-transplantations, load in retransplantations
        # Note that re-transplantations future to simulation start time
        # are not loaded in by load_patients if SIM_RETX.
        if sim_set.SIM_RETX:
            self.n_events = 0

            # Load retransplantations
            if self.verbose:
                print('Loading retransplantations')
            self.retransplantations = load_retransplantations(
                sim_set=self.sim_set,
                nrows=max_n_patients
            )
            preload_status_updates(
                patients=self.retransplantations,
                sim_set=sim_set,
                end_date_col='LOAD_RETXS_TO',
                se_sys=self.se_system
            )
            preload_aco_statuses(
                patients=self.retransplantations,
                sim_set=sim_set
            )
            preload_disease_groups(
                patients=self.retransplantations,
                sim_set=sim_set
            )
            self.remove_patients_without_statusupdates(
                pat_dict_name='retransplantations'
            )

        # Trigger all historic updates for the real patient list.
        for pat in self.patients.values():
            pat.trigger_historic_updates(
                se_sys=self.se_system
            )

        # Remove patients from real registrations that are
        # not initialized / never have any status update.
        self.remove_patients_without_statusupdates(
            pat_dict_name='patients'
        )

        # Initialize EventQueue with patient updates
        # for all patients, active or not
        for pat_id, pat in self.patients.items():
            next_update_at = pat.next_update_at()
            if next_update_at:
                self.event_queue.add(
                    Event(
                        type_event=cn.PAT,
                        event_time=next_update_at,
                        identifier=pat_id
                    )
                )
        # Initialize queue of active patients
        self.active_lists = ActiveList(
            init_patients=self.patients
        )
        if self.verbose:
            print('Initialized patients')

        # Initialize the post-transplant module.
        if sim_set.SIM_RETX:
            self.ptp = PostTransplantPredictor(
                seed=self.sim_set.SEED,
                offset_ids_transplants=max(
                    max(self.patients.keys()),
                    max(self.retransplantations.keys())
                ),
                retransplants=self.retransplantations,
                discrete_match_vars=es.POSTTXP_DISCRETE_MATCH_VARS,
                cvars_trafos=es.POSTTXP_TRANSFORMATIONS,
                cvars_caliper=es.POSTTXP_MATCH_CALIPERS,
                continuous_match_vars_rec=(
                    es.POSTTXP_CONTINUOUS_MATCH_VARS[cn.RETRANSPLANT]
                ),
                continuous_match_vars_off=(
                    es.POSTTXP_CONTINUOUS_MATCH_VARS[cn.OFFER]
                ),
                min_matches=es.POSTTXP_MIN_MATCHES,
                se_sys=self.se_system
            )

        # Schedule events for donors.
        for don_id, don in self.donors.items():
            self.event_queue.add(
                Event(
                    type_event=cn.DON,
                    event_time=don.arrival_at(
                        self.sim_set.SIM_START_DATE
                        ),
                    identifier=don_id
                )
            )

        # Initialize simulation results & path to match list file.
        self.sim_results = SimResults(
            cols_to_save_exit=es.OUTPUT_COLS_EXITS + (self.sim_set.LAB_MELD,),
            cols_to_save_discard=es.OUTPUT_COLS_DISCARDS,
            cols_to_save_patients=(
                es.OUTPUT_COLS_PATIENTS + (self.sim_set.LAB_MELD,)
            ),
            sim_set=self.sim_set
        )
        self.match_list_file = (
            str(self.sim_set.RESULTS_FOLDER) +
            str(self.sim_set.PATH_MATCH_LISTS)
        )

    def remove_patients_without_statusupdates(
        self, pat_dict_name: str, verbosity=0
    ) -> None:
        """Removes patients from dict without any updates."""

        n_removed = 0
        for id_reg, pat in list(self.__dict__[pat_dict_name].items()):
            if (
                not pat.is_initialized() or
                pat.__dict__[cn.EXIT_STATUS] is not None
            ):
                if verbosity > 0:
                    print(f'{id_reg}: {pat}')
                    input('')
                n_removed += 1

                del self.__dict__[pat_dict_name][id_reg]
        if self.verbose:
            print(
                f'Removed {n_removed} patients from self.{pat_dict_name} '
                f"that already exited or who don't have any status updates "
                f"(e.g., never any known biomarker)"
                )

    def process_nonacceptance(
            self, match_list: 'AllocationSystem.MatchList',
            current_date: datetime
    ):
        """ Process in case the graft was declined by all patients/centers.
            We either (i) force allocation to a FP-candidate or (ii) save
            the graft as a discard.

            Note that this happens very rarely (e.g. for HCV-positive donors.)
        """
        if hasattr(match_list, '_initialize_rescue_priorities'):
            match_list._initialize_rescue_priorities()

        if match_list.return_match_list() is None or self.allow_discards:
            self.sim_results.save_discard(
                matchl_=match_list
            )
        elif match_list.return_match_list() is not None:
            # Deduplicate list, then force acceptance by
            # candidate most likely to accept.
            filtered_match_records = list({
                off.__dict__[cn.ID_RECIPIENT]: off
                for off in match_list.return_match_list()
                if
                off.__dict__.get(cn.PROB_ACCEPT_P) is not None and
                off.__dict__.get(cn.ACCEPTANCE_REASON) not in
                {cg.TRANSPLANTED_CODES}
            }.values())

            filtered_match_records.sort(
                key=lambda x: (
                    x.__dict__[cn.RESCUE_PRIORITY],
                    x.__dict__[cn.PROB_ACCEPT_P]
                ),
                reverse=True
            )

            if len(filtered_match_records) > 0:
                mr = filtered_match_records[0]
                mr.set_acceptance(cn.T3)
                mr.__dict__[cn.TYPE_TRANSPLANTED] = (
                    mr.__dict__[cn.TYPE_OFFER_DETAILED]
                )
                self.process_accepted_mr(
                    current_date=current_date,
                    donor=mr.donor,
                    accepted_mr=mr
                )
            else:
                self.sim_results.save_discard(
                    match_list
                )

    def get_active_list(self, priority_country: str) -> ValuesView[Patient]:
        return self.active_lists.get_active_list(priority_country)

    def simulate_allocation(
            self,
            return_ml_donor: Optional[int] = None,
            verbose: bool = False,
            aggregate_obligations: Optional[bool] = None,
            print_progress_every_k_days=90
    ) -> Optional[MatchListCurrentELAS]:
        """Simulate the allocation algorithm"""

        # Set seed and maintain count for how many days were simulated.
        next_k_days = 1

        # Maintain when active waitlist was last updated
        _time_active_list_resorted = 0

        # Save start time, and print simulation period.
        start_time = time.time()
        print(
            f'Simulating ELASS from '
            f'{self.sim_set.SIM_START_DATE.strftime("%Y-%m-%d")}'
            f' to {self.sim_set.SIM_END_DATE.strftime("%Y-%m-%d")}'
            f' with allocation based on: \n  {self.sim_set.ALLOC_SCORE}'
        )

        # Start with an empty match list file.
        if self.sim_set.SAVE_MATCH_LISTS:
            if os.path.exists(self.match_list_file):
                os.remove(
                    self.match_list_file
                )
            if os.path.exists(self.match_list_file + '.gzip'):
                os.remove(
                    self.match_list_file + '.gzip'
                )

        # Save whether to aggregate obligations or not
        if aggregate_obligations is None:
            aggregate_obligations = bool(self.aggregate_obl)

        # Simulate until simulation is finished.
        while (
            not self.event_queue.is_empty() and
            (self.sim_time < self.max_sim_time)
        ):
            event = self.event_queue.next()

            # Print simulation progress
            if self.sim_time / print_progress_every_k_days >= next_k_days:
                print(
                    'Simulated up to {0}'.format(
                        (
                            self.sim_set.SIM_START_DATE +
                            timedelta(days=self.sim_time)
                        ).strftime('%Y-%m-%d')
                    )
                )
                next_k_days += 1

            self.sim_time = event.event_time
            if verbose:
                print(f'{self.sim_time:.2f}: {event}')

            if event.type_event == cn.PAT:
                # Update patient information, and schedule future event
                # if patient remains active.
                self.patients[event.identifier].update_patient(
                    se_sys=self.se_system,
                    sim_results=self.sim_results
                )
                # If patient does not exit, schedule a new future event
                # time at his next update.
                if self.patients[event.identifier].exit_status is None:
                    event.update_event_time(
                        new_time=self.patients[
                            event.identifier
                            ].next_update_at()
                    )
                    self.event_queue.add(
                        event
                    )
                # If patient becomes active, add to active list,
                if (
                    not self.active_lists.is_active(event.identifier) and
                    self.patients[event.identifier].active
                ):
                    self.active_lists.add_to_active_list(
                        event.identifier,
                        self.patients[event.identifier]
                        )
                if not self.patients[event.identifier].active:
                    self.active_lists.pop_from_active_lists(
                        event.identifier
                    )

                # Resort list of active patients every 30 days.
                if self.sim_time - _time_active_list_resorted > 30:
                    self.active_lists.resort_lists()
                    _time_active_list_resorted = self.sim_time

            elif event.type_event == cn.DON:
                donor = self.donors[event.identifier]
                donor_dict = donor.__dict__
                current_date = (
                    self.sim_set.SIM_START_DATE +
                    timedelta(days=self.sim_time)
                )
                # Construct an MatchList with all patients
                # that are BG-compatible, have an active status,
                # and a first MELD score
                match_list = MatchListCurrentELAS(
                    patients=(
                        p for p in self.get_active_list(
                            priority_country=donor.__dict__[cn.D_COUNTRY]
                            ) if (
                            p.r_bloodgroup not in
                            BLOOD_GROUP_INCOMPATIBILITY_DICT[
                                donor_dict[cn.D_BLOODGROUP]
                            ] and
                            (
                                donor_dict[cn.D_DCD] == 0 or
                                (
                                    p.__dict__[cn.RECIPIENT_COUNTRY]
                                    in es.DCD_ACCEPTING_COUNTRIES
                                )
                            )
                        )
                    ),
                    donor=donor,
                    match_date=current_date,
                    sim_start_date=self.sim_set.SIM_START_DATE,
                    type_offer=donor_dict[cn.FIRST_OFFER_TYPE],
                    obl=self.obligations,
                    aggregate_obligations=aggregate_obligations,
                    travel_time_dict=self.dict_travel_time,
                    et_comp_thresh=self.meld_etcomp_thresh
                )

                # Now determine which patient accepts the organ
                accepted_mr = (
                    self.acceptance_module.simulate_liver_allocation(
                        match_list
                    )
                )

                # Save match list.
                if self.sim_set.SAVE_MATCH_LISTS:
                    self.sim_results.save_match_list(
                        match_list
                    )

                if accepted_mr:

                    # Simulate for accepting patient if organ is splitted
                    accepted_mr.__dict__[cn.TYPE_TRANSPLANTED] = (
                        self.acceptance_module.
                        return_transplanted_organ(accepted_mr)
                    )

                    # Process the accepted MR. That is, set the candidate to
                    # transplanted.
                    self.process_accepted_mr(
                        current_date=current_date,
                        donor=donor,
                        accepted_mr=accepted_mr
                    )

                    # If organ is split, make match list for split organ
                    if (
                        accepted_mr.__dict__[cn.TYPE_TRANSPLANTED] !=
                        accepted_mr.__dict__[cn.TYPE_OFFER_DETAILED]
                    ):
                        all_pats = match_list.return_patient_ids()
                        split_match_list = MatchListCurrentELAS(
                            patients=(
                                p for p in self.get_active_list(
                                    donor.__dict__[cn.D_COUNTRY]
                                ) if
                                p.active and
                                p.id_recipient in all_pats[
                                    all_pats.index(
                                        accepted_mr.patient.id_recipient
                                        )+1:
                                ]
                            ),
                            donor=donor,
                            match_date=(
                                self.sim_set.SIM_START_DATE +
                                timedelta(days=self.sim_time)
                                ),
                            sim_start_date=self.sim_set.SIM_START_DATE,
                            type_offer=es.ORGAN_AVAILABILITY_SPLIT[
                                accepted_mr.__dict__[cn.TYPE_TRANSPLANTED]
                            ],
                            obl=self.obligations,
                            alloc_center=(
                                accepted_mr.__dict__[cn.RECIPIENT_CENTER]
                            ),
                            travel_time_dict=self.dict_travel_time,
                            et_comp_thresh=self.meld_etcomp_thresh
                        )

                        # Simulate acceptance of the split.
                        accepted_split = (
                            self.acceptance_module.simulate_liver_allocation(
                                split_match_list
                            )
                        )

                        if self.sim_set.SAVE_MATCH_LISTS:
                            self.sim_results.save_match_list(
                                split_match_list
                            )

                        # Set patient who accepted the organ to transplanted.
                        if accepted_split:
                            accepted_split.__dict__[cn.TYPE_TRANSPLANTED] = (
                                self.acceptance_module.
                                return_transplanted_organ(accepted_split)
                            )
                            self.process_accepted_mr(
                                current_date=current_date,
                                donor=donor,
                                accepted_mr=accepted_split
                            )
                        else:
                            self.process_nonacceptance(
                                match_list=split_match_list,
                                current_date=current_date
                            )

                    # Create an obligation if accepting patient is HU/ACO
                    if (
                        accepted_mr.patient.r_aco or
                        (accepted_mr.__dict__[cn.PATIENT_IS_HU])
                    ):
                        if accepted_mr.donor.donor_country in es.ET_COUNTRIES:
                            self.obligations.add_obligation_from_matchrecord(
                                match_record=accepted_mr,
                                track=True
                            )
                    # Redeem obligations if acceptance is based on obligation
                    elif accepted_mr.__dict__[cn.MTCH_OBL] > 0:
                        self.obligations.add_obligation_from_matchrecord(
                                match_record=accepted_mr,
                                track=False
                            )
                else:
                    self.process_nonacceptance(
                        match_list=match_list,
                        current_date=current_date
                    )
            else:
                raise ValueError(
                    f"{event.type_event} is not a valid event type."
                    )
        print(
            "--- Finished simulation in {0} seconds ---" .
            format(round_to_decimals(time.time() - start_time, 1))
        )

    def set_obligations(self, obl: CountryObligations):
        """Override initial obligations."""
        self.obligations = obl

    def simulate_posttxp(self, txp: 'AllocationSystem.MatchRecord'):
        """Code to simulate post-transplant outcomes for transplanted
            patients. This simulates failure and relisting dates, but
            also adds a synthetic re-registration to the simulation
            in case the re-listing occurs during the sim period.
        """

        # Simulate post-transplant survival
        date_fail, date_relist, cause_fail = self.ptp.simulate_failure_date(
            offer=txp,
            current_date=(
                self.sim_set.SIM_START_DATE +
                timedelta(days=self.sim_time)
            ),
            split=int(
                txp.__dict__[cn.TYPE_TRANSPLANTED] !=
                es.TYPE_OFFER_WLIV
            )
        )

        self.sim_results.save_posttransplant(
            date_failure=date_fail,
            date_relist=date_relist,
            cens_date=self.sim_set.SIM_END_DATE,
            matchr_=txp,
            rereg_id=(
                (
                    self.ptp.offset_ids_transplants +
                    self.ptp.synth_regs
                )
                if date_relist and date_relist < self.sim_set.SIM_END_DATE
                else None
            )
        )

        if (
            cause_fail == cn.PATIENT_RELISTING and
            date_relist and
            date_relist < self.sim_set.SIM_END_DATE
        ):
            # If time-to-event is longer in matched patient than non-matched
            # patient. But then only match to patients with
            # longer time-to-events.
            synth_reg = self.ptp.generate_synthetic_reregistration(
                offer=txp,
                relist_date=date_relist,
                fail_date=date_fail,
                curr_date=(
                    self.sim_set.SIM_START_DATE +
                    timedelta(days=self.sim_time)
                ),
                verbosity=0
            )

            self.patients[synth_reg.__dict__[cn.ID_REGISTRATION]] = synth_reg
            next_update_time = synth_reg.next_update_at()

            if next_update_time:
                self.event_queue.add(
                    Event(
                        type_event=cn.PAT,
                        event_time=next_update_time,
                        identifier=synth_reg.__dict__[cn.ID_REGISTRATION]
                    )
                )

    def process_accepted_mr(
            self,
            current_date: datetime,
            donor: Donor,
            accepted_mr: 'AllocationSystem.MatchRecord'
    ):
        if isinstance(accepted_mr, MatchRecordCurrentELAS):
            accepted_mr._initialize_posttxp_information(
                self.ptp
            )

        # Set patient who accepted the organ to transplanted.
        accepted_mr.patient.set_transplanted(
            tx_date=(current_date),
            donor=donor,
            match_record=accepted_mr,
            sim_results=self.sim_results
        )
        self.active_lists.pop_from_active_lists(
            accepted_mr.patient.id_registration
        )

        # Simulate post-transplant survival, i.e. simulate
        # patient / graft failure, and add a synthetic
        # reregistration if graft fails before end
        # of simulation period.
        if self.sim_set.SIM_RETX:
            self.simulate_posttxp(
                txp=accepted_mr
            )

    def __repr__(self):
        return(
            f'ELAS simulator from {self.sim_set.SIM_START_DATE.date()} to '
            f'{self.sim_set.SIM_END_DATE.date()}, with {len(self.patients)} '
            f'patients and {len(self.donors)} donors at time {self.sim_time}'
            )
