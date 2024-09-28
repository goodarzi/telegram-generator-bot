from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="RealesrganItem")


@_attrs_define
class RealesrganItem:
    """
    Attributes:
        name (str):
        path (Union[None, str]):
        scale (Union[None, int]):
    """

    name: str
    path: Union[None, str]
    scale: Union[None, int]
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        path: Union[None, str]
        path = self.path

        scale: Union[None, int]
        scale = self.scale

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "path": path,
                "scale": scale,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        def _parse_path(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        path = _parse_path(d.pop("path"))

        def _parse_scale(data: object) -> Union[None, int]:
            if data is None:
                return data
            return cast(Union[None, int], data)

        scale = _parse_scale(d.pop("scale"))

        realesrgan_item = cls(
            name=name,
            path=path,
            scale=scale,
        )

        realesrgan_item.additional_properties = d
        return realesrgan_item

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
