"""Upright profile registration contracts and intrinsic operations."""

from __future__ import annotations

from typing import (
    ClassVar,
    Self,
)

from pydantic import (
    model_validator,
)

from spatialcf.domain.base import (
    Sha256Digest,
)

from spatialcf.domain.definitions import (
    CapabilityRef,
    HashBoundCanonicalModel,
)

from spatialcf.domain.profiles import (
    ActionSpaceProfile,
    OwnerRef,
    SemanticsProfile,
)

from spatialcf.domain._upright_se2.constants import (
    UPRIGHT_SE2_BACKEND_OWNER_REF,
    UPRIGHT_SE2_CARDINAL_BACKEND_CAPABILITY_REF,
    UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF,
    UPRIGHT_SE2_CARDINAL_COMPILER_CAPABILITY_REF,
    UPRIGHT_SE2_CHECKER_OWNER_REF,
    UPRIGHT_SE2_CONTINUOUS_BACKEND_CAPABILITY_REF,
    UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF,
    UPRIGHT_SE2_CONTINUOUS_COMPILER_CAPABILITY_REF,
    UPRIGHT_SE2_OPERATOR_REFS,
    UPRIGHT_SE2_PROFILE_CAPABILITY_REF,
    UPRIGHT_SE2_PROFILE_REF,
    UPRIGHT_SE2_SEMANTICS_PROFILE_REF,
    UPRIGHT_SE2_STAGED_CAPABILITY_REFS,
)


class UprightSE2ProfileRegistration(HashBoundCanonicalModel):
    """One immutable semantic/action profile plus staged static capabilities."""

    HASH_DOMAIN: ClassVar[str] = (
        "spatialcf/counterfactual/upright-se2/profile-registration/3.0"
    )
    SELF_DIGEST_FIELD: ClassVar[str] = "profile_registration_sha256"

    semantics_profile: SemanticsProfile
    action_space_profile: ActionSpaceProfile
    backend_owner_ref: OwnerRef
    checker_owner_ref: OwnerRef
    profile_capability_ref: CapabilityRef
    cardinal_compiler_capability_ref: CapabilityRef
    cardinal_backend_capability_ref: CapabilityRef
    cardinal_checker_capability_ref: CapabilityRef
    continuous_compiler_capability_ref: CapabilityRef
    continuous_backend_capability_ref: CapabilityRef
    continuous_checker_capability_ref: CapabilityRef
    profile_registration_sha256: Sha256Digest

    @model_validator(mode="after")
    def _validate_immutable_profile(self) -> Self:
        if (
            self.semantics_profile.semantics_profile_ref
            != UPRIGHT_SE2_SEMANTICS_PROFILE_REF
        ):
            raise ValueError("semantics profile reference must be fixed")
        if (
            self.action_space_profile.action_space_profile_ref
            != UPRIGHT_SE2_PROFILE_REF
        ):
            raise ValueError("action-space profile reference must be fixed")
        if (
            self.semantics_profile.transition_semantics_refs
            != UPRIGHT_SE2_OPERATOR_REFS
        ):
            raise ValueError(
                "semantics profile must name exactly four upright operators"
            )
        if self.action_space_profile.allowed_operator_refs != UPRIGHT_SE2_OPERATOR_REFS:
            raise ValueError(
                "action-space profile must name exactly four upright operators"
            )
        if (
            self.action_space_profile.backend_capability_requirements
            != UPRIGHT_SE2_STAGED_CAPABILITY_REFS
        ):
            raise ValueError(
                "action-space profile must retain both staged capability sets"
            )
        if self.backend_owner_ref != UPRIGHT_SE2_BACKEND_OWNER_REF:
            raise ValueError("backend owner reference must be fixed")
        if self.checker_owner_ref != UPRIGHT_SE2_CHECKER_OWNER_REF:
            raise ValueError("checker owner reference must be fixed")
        if self.backend_owner_ref == self.checker_owner_ref:
            raise ValueError("backend and checker owners must be distinct")
        expected_capabilities = (
            UPRIGHT_SE2_PROFILE_CAPABILITY_REF,
            UPRIGHT_SE2_CARDINAL_COMPILER_CAPABILITY_REF,
            UPRIGHT_SE2_CARDINAL_BACKEND_CAPABILITY_REF,
            UPRIGHT_SE2_CARDINAL_CHECKER_CAPABILITY_REF,
            UPRIGHT_SE2_CONTINUOUS_COMPILER_CAPABILITY_REF,
            UPRIGHT_SE2_CONTINUOUS_BACKEND_CAPABILITY_REF,
            UPRIGHT_SE2_CONTINUOUS_CHECKER_CAPABILITY_REF,
        )
        actual_capabilities = (
            self.profile_capability_ref,
            self.cardinal_compiler_capability_ref,
            self.cardinal_backend_capability_ref,
            self.cardinal_checker_capability_ref,
            self.continuous_compiler_capability_ref,
            self.continuous_backend_capability_ref,
            self.continuous_checker_capability_ref,
        )
        if actual_capabilities != expected_capabilities:
            raise ValueError("profile capability references must be fixed")
        return self


# Resolve local model forward references before restoring public identities.
UprightSE2ProfileRegistration.model_rebuild()


# Keep supported public import and pickle lookup stable.
UprightSE2ProfileRegistration.__module__ = "spatialcf.domain.upright_se2"
