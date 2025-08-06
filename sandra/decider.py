import time
from typing import Optional, Any, Union, List

from sandra.actions import Action, LongitudinalAction, LateralAction
from sandra.commonroad.reach import ReachVerifier
from sandra.describer import DescriberBase
from sandra.llm import get_structured_response
from config.sandra import SanDRAConfiguration
from sandra.verifier import VerifierBase, DummyVerifier, VerificationStatus
from highway_env.vehicle.controller import ControlledVehicle

# ranking, verified_idx, inference_duration, verification_duration
Decision = tuple[list[Action], int, float, float]

class Decider:
    def __init__(
        self,
        config: SanDRAConfiguration,
        describer: DescriberBase,
        verifier: Optional[Union[VerifierBase, ReachVerifier]] = None,
    ):
        self.config: SanDRAConfiguration = config
        self.time_step = 0
        self.action_ranking = None
        self.describer = describer
        self.verifier = verifier
        if verifier is None:
            self.verifier = DummyVerifier()

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

    def decide(
        self, past_action: List[List[Union[LongitudinalAction, LateralAction]]] = None,
        include_metadata: bool = False,
        verbose: bool = False,
    ) -> Action | Decision:
        user_prompt = self.describer.user_prompt()
        system_prompt = self.describer.system_prompt(past_action)
        schema = self.describer.schema()

        try:
            start = time.time()
            structured_response = get_structured_response(
                user_prompt, system_prompt, schema, self.config
            )
            inference_time = time.time() - start
            ranking = self._parse_action_ranking(structured_response)
        except Exception as _:
            inference_time = 30.0
            ranking = [(LongitudinalAction.DECELERATE, LateralAction.FOLLOW_LANE)] * self.config.k

        if verbose:
            print("Ranking:")
            for i, (longitudinal, lateral) in enumerate(ranking):
                print(f"{i + 1}. ({longitudinal}, {lateral})")

        start = time.time()
        for i, action in enumerate(ranking):
            try:
                status = self.verifier.verify(list(action))
            except Exception as _:
                continue
            if status == VerificationStatus.SAFE:
                if verbose:
                    print(f"Successfully verified {action}.")
                ControlledVehicle.KP_A = 1 / 0.6
                ControlledVehicle.DELTA_SPEED = 5

                if include_metadata:
                    verification_time = time.time() - start
                    return ranking, i, inference_time, verification_time
                else:
                    return action
            if verbose:
                print(f"Failed to verify {action}.")
        verification_time = time.time() - start

        ControlledVehicle.KP_A = 1 / 0.2
        ControlledVehicle.DELTA_SPEED = 15
        if include_metadata:
            return ranking, len(ranking), inference_time, verification_time
        return LongitudinalAction.DECELERATE, LateralAction.FOLLOW_LANE
