import os
from typing import Optional, Any, Union, List

import pandas as pd

from sandra.actions import Action, LongitudinalAction, LateralAction
from sandra.commonroad.reach import ReachVerifier
from sandra.describer import DescriberBase
from sandra.llm import get_structured_response
from sandra.config import SanDRAConfiguration
from sandra.verifier import VerifierBase, DummyVerifier, VerificationStatus

from highway_env.vehicle.controller import ControlledVehicle


class Decider:
    def __init__(
        self,
        config: SanDRAConfiguration,
        describer: DescriberBase,
        verifier: Optional[Union[VerifierBase, ReachVerifier]] = None,
        save_path: Optional[str] = None,
    ):
        self.config: SanDRAConfiguration = config
        self.time_step = 0
        self.action_ranking = None
        self.describer = describer
        self.verifier = verifier
        if verifier is None:
            self.verifier = DummyVerifier()
        if save_path is None:
            self.save_path = "batch_results.csv"
        else:
            self.save_path = save_path

        columns = [
            "iteration-id",
            "Lateral1",
            "Longitudinal1",
            "Lateral2",
            "Longitudinal2",
            "Lateral3",
            "Longitudinal3",
            "verified-id",
            "user-prompt",
            "system-prompt",
            "schema",
        ]

        if not os.path.exists(self.save_path + "/evaluation.csv"):
            empty_df = pd.DataFrame(columns=columns)
            empty_df.to_csv(self.save_path + "/evaluation.csv", index=False)
            print(f"Created new CSV file: {self.save_path + '/evaluation.csv'}")

    def _parse_action_ranking(self, llm_response: dict[str, Any]) -> list[Action]:
        action_ranking = []
        k = self.config.k
        ranking_prefixes = [
            "",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
        ]
        for i, prefix in enumerate(ranking_prefixes[:k]):
            key = f"{prefix}_best_combination" if prefix else "best_combination"
            try:
                action = llm_response[key]
                long_act = LongitudinalAction(action["longitudinal_action"])
                lat_act = LateralAction(action["lateral_action"])
                action_ranking.append((long_act, lat_act))
            except (KeyError, IndexError, TypeError) as e:
                print(f"[Warning] Could not parse rank {i + 1} ({key}): {e}")
                continue

        if len(action_ranking) != k:
            raise ValueError(f"Only {len(action_ranking)} of {k} actions could be parsed.")

        return action_ranking

    def save_iteration(self, row: dict[str, Any]):
        df = pd.read_csv(self.save_path + "/evaluation.csv")
        new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        new_df.to_csv(self.save_path + "/evaluation.csv", index=False)

    def decide(
        self, past_action: List[List[Union[LongitudinalAction, LateralAction]]] = None
    ) -> Optional[Action]:
        user_prompt = self.describer.user_prompt()
        system_prompt = self.describer.system_prompt(past_action)
        schema = self.describer.schema()
        structured_response = get_structured_response(
            user_prompt, system_prompt, schema, self.config
        )
        ranking = self._parse_action_ranking(structured_response)

        new_row = {
            "iteration-id": self.time_step,
            "Lateral1": None,
            "Longitudinal1": None,
            "Lateral2": None,
            "Longitudinal2": None,
            "Lateral3": None,
            "Longitudinal3": None,
            "verified-id": None,
            "user-prompt": user_prompt,
            "system-prompt": system_prompt,
            "schema": schema,
        }

        print("Ranking:")
        for i, (longitudinal, lateral) in enumerate(ranking):
            print(f"{i + 1}. ({longitudinal}, {lateral})")
            new_row[f"Lateral{i + 1}"] = lateral.value
            new_row[f"Longitudinal{i + 1}"] = longitudinal.value

        for i, action in enumerate(ranking):
            try:
                status = self.verifier.verify(list(action))
            except Exception as _:
                continue
            if status == VerificationStatus.SAFE:
                print(f"Successfully verified {action}.")
                new_row["verified-id"] = i
                self.save_iteration(new_row)

                ControlledVehicle.KP_A = 1 / 0.6
                ControlledVehicle.DELTA_SPEED = 5

                return action
            print(f"Failed to verify {action}.")
        new_row["verified-id"] = len(ranking)
        self.save_iteration(new_row)

        ControlledVehicle.KP_A = 1 / 0.2
        ControlledVehicle.DELTA_SPEED = 15

        return LongitudinalAction.DECELERATE, LateralAction.FOLLOW_LANE
