from typing import Any, Dict, List, Type, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="ExtensionItem")


@_attrs_define
class ExtensionItem:
    """
    Attributes:
        name (str): Extension name
        remote (str): Extension Repository URL
        branch (str): Extension Repository Branch
        commit_hash (str): Extension Repository Commit Hash
        version (str): Extension Version
        commit_date (int): Extension Repository Commit Date
        enabled (bool): Flag specifying whether this extension is enabled
    """

    name: str
    remote: str
    branch: str
    commit_hash: str
    version: str
    commit_date: int
    enabled: bool
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        remote = self.remote

        branch = self.branch

        commit_hash = self.commit_hash

        version = self.version

        commit_date = self.commit_date

        enabled = self.enabled

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "remote": remote,
                "branch": branch,
                "commit_hash": commit_hash,
                "version": version,
                "commit_date": commit_date,
                "enabled": enabled,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        remote = d.pop("remote")

        branch = d.pop("branch")

        commit_hash = d.pop("commit_hash")

        version = d.pop("version")

        commit_date = d.pop("commit_date")

        enabled = d.pop("enabled")

        extension_item = cls(
            name=name,
            remote=remote,
            branch=branch,
            commit_hash=commit_hash,
            version=version,
            commit_date=commit_date,
            enabled=enabled,
        )

        extension_item.additional_properties = d
        return extension_item

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
