from typing import Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="UpscalerItem")


@_attrs_define
class UpscalerItem:
    """
    Attributes:
        name (str):
        model_name (Union[Unset, str]):
        model_path (Union[Unset, str]):
        model_url (Union[Unset, str]):
        scale (Union[Unset, float]):
    """

    name: str
    model_name: Union[Unset, str] = UNSET
    model_path: Union[Unset, str] = UNSET
    model_url: Union[Unset, str] = UNSET
    scale: Union[Unset, float] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        model_name = self.model_name

        model_path = self.model_path

        model_url = self.model_url

        scale = self.scale

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
            }
        )
        if model_name is not UNSET:
            field_dict["model_name"] = model_name
        if model_path is not UNSET:
            field_dict["model_path"] = model_path
        if model_url is not UNSET:
            field_dict["model_url"] = model_url
        if scale is not UNSET:
            field_dict["scale"] = scale

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        model_name = d.pop("model_name", UNSET)

        model_path = d.pop("model_path", UNSET)

        model_url = d.pop("model_url", UNSET)

        scale = d.pop("scale", UNSET)

        upscaler_item = cls(
            name=name,
            model_name=model_name,
            model_path=model_path,
            model_url=model_url,
            scale=scale,
        )

        upscaler_item.additional_properties = d
        return upscaler_item

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
