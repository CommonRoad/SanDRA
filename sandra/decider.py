from typing import Optional, Any
from sandra.actions import Action, LongitudinalAction, LateralAction
from sandra.commonroad.describer import CommonRoadDescriber
from sandra.describer import DescriberBase
from sandra.llm import get_structured_response
from sandra.common.config import SanDRAConfiguration
from sandra.verifier import VerifierBase, DummyVerifier, VerificationStatus


class Decider:
    def __init__(
        self,
        config: SanDRAConfiguration,
        describer: DescriberBase,
        verifier: Optional[VerifierBase] = None,
        save_path: Optional[str] = None,
    ):
        self.config: SanDRAConfiguration = config
        self.action_ranking = None
        self.describer = describer
        self.verifier = verifier
        if verifier is None:
            self.verifier = DummyVerifier()
        self.save_path = save_path

    def _parse_action_ranking(self, llm_response: dict[str, Any]) -> list[Action]:
        action_ranking = []
        k = self.config.m
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
        for prefix in ranking_prefixes[:k]:
            key = f"{prefix}_best_combination" if prefix else "best_combination"
            action = llm_response[key]
            action_ranking.append(
                (
                    LongitudinalAction(action["longitudinal_action"]),
                    LateralAction(action["lateral_action"]),
                )
            )
        return action_ranking

    def decide(self) -> Optional[Action]:
        user_prompt = self.describer.user_prompt()
        system_prompt = self.describer.system_prompt()
        schema = self.describer.schema()
        structured_response = get_structured_response(
            user_prompt, system_prompt, schema, self.config, save_dir=self.save_path
        )
        ranking = self._parse_action_ranking(structured_response)

        print("Ranking:")
        for i, (longitudinal, lateral) in enumerate(ranking):
            print(f"{i + 1}. ({longitudinal}, {lateral})")

        for action in ranking:
            if self.verifier.verify(list(action)) == VerificationStatus.SAFE:
                print(f"Successfully verified {action}.")
                return action
            print(f"Failed to verify {action}.")
        return None
