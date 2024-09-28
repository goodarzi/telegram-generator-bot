from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.text_to_image_response_parameters import TextToImageResponseParameters


T = TypeVar("T", bound="TextToImageResponse")


@_attrs_define
class TextToImageResponse:
    """
    Attributes:
        parameters (TextToImageResponseParameters):
        info (str):
        images (Union[Unset, List[str]]): The generated image in base64 format.
    """

    parameters: "TextToImageResponseParameters"
    info: str
    images: Union[Unset, List[str]] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        parameters = self.parameters.to_dict()

        info = self.info

        images: Union[Unset, List[str]] = UNSET
        if not isinstance(self.images, Unset):
            images = self.images

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "parameters": parameters,
                "info": info,
            }
        )
        if images is not UNSET:
            field_dict["images"] = images

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.text_to_image_response_parameters import TextToImageResponseParameters

        d = src_dict.copy()
        parameters = TextToImageResponseParameters.from_dict(d.pop("parameters"))

        info = d.pop("info")

        images = cast(List[str], d.pop("images", UNSET))

        text_to_image_response = cls(
            parameters=parameters,
            info=info,
            images=images,
        )

        text_to_image_response.additional_properties = d
        return text_to_image_response

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
