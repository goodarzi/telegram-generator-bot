from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.png_info_response_items import PNGInfoResponseItems
    from ..models.png_info_response_parameters import PNGInfoResponseParameters


T = TypeVar("T", bound="PNGInfoResponse")


@_attrs_define
class PNGInfoResponse:
    """
    Attributes:
        info (str): A string with the parameters used to generate the image
        items (PNGInfoResponseItems): A dictionary containing all the other fields the image had
        parameters (PNGInfoResponseParameters): A dictionary with parsed generation info fields
    """

    info: str
    items: "PNGInfoResponseItems"
    parameters: "PNGInfoResponseParameters"
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        info = self.info

        items = self.items.to_dict()

        parameters = self.parameters.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "info": info,
                "items": items,
                "parameters": parameters,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.png_info_response_items import PNGInfoResponseItems
        from ..models.png_info_response_parameters import PNGInfoResponseParameters

        d = src_dict.copy()
        info = d.pop("info")

        items = PNGInfoResponseItems.from_dict(d.pop("items"))

        parameters = PNGInfoResponseParameters.from_dict(d.pop("parameters"))

        png_info_response = cls(
            info=info,
            items=items,
            parameters=parameters,
        )

        png_info_response.additional_properties = d
        return png_info_response

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
