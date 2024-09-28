from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="FaceRestorerItem")


@_attrs_define
class FaceRestorerItem:
    """
    Attributes:
        name (str):
        cmd_dir (Union[None, str]):
    """

    name: str
    cmd_dir: Union[None, str]
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        cmd_dir: Union[None, str]
        cmd_dir = self.cmd_dir

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "cmd_dir": cmd_dir,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        def _parse_cmd_dir(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        cmd_dir = _parse_cmd_dir(d.pop("cmd_dir"))

        face_restorer_item = cls(
            name=name,
            cmd_dir=cmd_dir,
        )

        face_restorer_item.additional_properties = d
        return face_restorer_item

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
