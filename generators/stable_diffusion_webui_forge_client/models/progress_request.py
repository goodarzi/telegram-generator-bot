from typing import Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ProgressRequest")


@_attrs_define
class ProgressRequest:
    """
    Attributes:
        id_task (Union[Unset, str]): id of the task to get progress for
        id_live_preview (Union[Unset, int]): id of last received last preview image Default: -1.
        live_preview (Union[Unset, bool]): boolean flag indicating whether to include the live preview image Default:
            True.
    """

    id_task: Union[Unset, str] = UNSET
    id_live_preview: Union[Unset, int] = -1
    live_preview: Union[Unset, bool] = True
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id_task = self.id_task

        id_live_preview = self.id_live_preview

        live_preview = self.live_preview

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if id_task is not UNSET:
            field_dict["id_task"] = id_task
        if id_live_preview is not UNSET:
            field_dict["id_live_preview"] = id_live_preview
        if live_preview is not UNSET:
            field_dict["live_preview"] = live_preview

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        id_task = d.pop("id_task", UNSET)

        id_live_preview = d.pop("id_live_preview", UNSET)

        live_preview = d.pop("live_preview", UNSET)

        progress_request = cls(
            id_task=id_task,
            id_live_preview=id_live_preview,
            live_preview=live_preview,
        )

        progress_request.additional_properties = d
        return progress_request

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
