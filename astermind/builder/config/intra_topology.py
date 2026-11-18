"""Physical SU membership, independent of whole-model eligibility."""
from pydantic import BaseModel, ConfigDict, model_validator


class IntraHierarchy(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    su_groups: tuple[tuple[str, ...], ...]
    # Optional, ordered hints for models that require one switch per domain.
    # Physical links remain authoritative; a declaration creates no edges.
    su_switches: tuple[str, ...] = ()

    @model_validator(mode='after')
    def validate_groups(self):
        if not self.su_groups or any(not group for group in self.su_groups):
            raise ValueError('su groups must be nonempty')
        ids = sum(self.su_groups, ()) + self.su_switches
        if any(not identifier.strip() for identifier in ids):
            raise ValueError('su device IDs must be nonempty')
        if len(set(ids)) != len(ids):
            raise ValueError('su GPU and switch IDs must be distinct')
        if self.su_switches and len(self.su_switches) != len(self.su_groups):
            raise ValueError('optional su_switches must correspond to su_groups in order')
        return self
