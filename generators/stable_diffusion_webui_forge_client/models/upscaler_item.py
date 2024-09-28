from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="UpscalerItem")


@_attrs_define
class UpscalerItem:
    """
    Attributes:
        name (str):
        model_name (Union[None, str]):
        model_path (Union[None, str]):
        model_url (Union[None, str]):
        scale (Union[None, float]):
    """

    name: str
    model_name: Union[None, str]
    model_path: Union[None, str]
    model_url: Union[None, str]
    scale: Union[None, float]
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        model_name: Union[None, str]
        model_name = self.model_name

        model_path: Union[None, str]
        model_path = self.model_path

        model_url: Union[None, str]
        model_url = self.model_url

        scale: Union[None, float]
        scale = self.scale

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "model_name": model_name,
                "model_path": model_path,
                "model_url": model_url,
                "scale": scale,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        def _parse_model_name(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        model_name = _parse_model_name(d.pop("model_name"))

        def _parse_model_path(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        model_path = _parse_model_path(d.pop("model_path"))

        def _parse_model_url(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        model_url = _parse_model_url(d.pop("model_url"))

        def _parse_scale(data: object) -> Union[None, float]:
            if data is None:
                return data
            return cast(Union[None, float], data)

        scale = _parse_scale(d.pop("scale"))

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
