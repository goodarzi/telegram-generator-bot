from typing import Any, Dict, List, Type, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="ExtrasBatchImagesResponse")


@_attrs_define
class ExtrasBatchImagesResponse:
    """
    Attributes:
        html_info (str): A series of HTML tags containing the process info.
        images (List[str]): The generated images in base64 format.
    """

    html_info: str
    images: List[str]
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        html_info = self.html_info

        images = self.images

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "html_info": html_info,
                "images": images,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        html_info = d.pop("html_info")

        images = cast(List[str], d.pop("images"))

        extras_batch_images_response = cls(
            html_info=html_info,
            images=images,
        )

        extras_batch_images_response.additional_properties = d
        return extras_batch_images_response

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
