from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.image_to_image_response_parameters import ImageToImageResponseParameters


T = TypeVar("T", bound="ImageToImageResponse")


@_attrs_define
class ImageToImageResponse:
    """
    Attributes:
        parameters (ImageToImageResponseParameters):
        info (str):
        images (Union[List[str], None, Unset]): The generated image in base64 format.
    """

    parameters: "ImageToImageResponseParameters"
    info: str
    images: Union[List[str], None, Unset] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        parameters = self.parameters.to_dict()

        info = self.info

        images: Union[List[str], None, Unset]
        if isinstance(self.images, Unset):
            images = UNSET
        elif isinstance(self.images, list):
            images = self.images

        else:
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
        from ..models.image_to_image_response_parameters import ImageToImageResponseParameters

        d = src_dict.copy()
        parameters = ImageToImageResponseParameters.from_dict(d.pop("parameters"))

        info = d.pop("info")

        def _parse_images(data: object) -> Union[List[str], None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                images_type_0 = cast(List[str], data)

                return images_type_0
            except:  # noqa: E722
                pass
            return cast(Union[List[str], None, Unset], data)

        images = _parse_images(d.pop("images", UNSET))

        image_to_image_response = cls(
            parameters=parameters,
            info=info,
            images=images,
        )

        image_to_image_response.additional_properties = d
        return image_to_image_response

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
