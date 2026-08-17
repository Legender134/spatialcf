from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator


class RunProfile(StrEnum):
    SMOKE = "smoke"
    EVIDENCE = "evidence"


class ArtifactProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    run_profile: RunProfile
    evidence_eligible: bool

    @model_validator(mode="after")
    def validate_eligibility(self) -> "ArtifactProfile":
        expected = self.run_profile is RunProfile.EVIDENCE
        if self.evidence_eligible is not expected:
            if self.run_profile is RunProfile.SMOKE:
                raise ValueError("smoke artifacts must remain ineligible")
            raise ValueError("evidence artifacts must be eligible")
        return self

    @classmethod
    def for_run(cls, profile: RunProfile) -> "ArtifactProfile":
        return cls(
            run_profile=profile,
            evidence_eligible=profile is RunProfile.EVIDENCE,
        )


def profile_from_manifest(manifest: dict[str, object]) -> ArtifactProfile | None:
    if manifest.get("schema_version") != 4:
        return None
    run_profile = manifest.get("run_profile")
    if isinstance(run_profile, str):
        run_profile = RunProfile(run_profile)
    return ArtifactProfile.model_validate(
        {
            "run_profile": run_profile,
            "evidence_eligible": manifest.get("evidence_eligible"),
        }
    )
