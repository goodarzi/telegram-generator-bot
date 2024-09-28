from typing import Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="InterrogateRequest")


@_attrs_define
class InterrogateRequest:
    """
    Attributes:
        image (Union[Unset, str]): Image to work on, must be a Base64 string containing the image's data. Default: ''.
        model (Union[Unset, str]): The interrogate model used. Default: 'clip'.
    """

    image: Union[Unset, str] = ""
    model: Union[Unset, str] = "clip"
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        image = self.image

        model = self.model

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if image is not UNSET:
            field_dict["image"] = image
        if model is not UNSET:
            field_dict["model"] = model

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        image = d.pop("image", UNSET)

        model = d.pop("model", UNSET)

        interrogate_request = cls(
            image=image,
            model=model,
        )

        interrogate_request.additional_properties = d
        return interrogate_request

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
