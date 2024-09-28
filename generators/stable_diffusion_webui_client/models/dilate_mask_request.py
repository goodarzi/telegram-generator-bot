from typing import Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="DilateMaskRequest")


@_attrs_define
class DilateMaskRequest:
    """
    Attributes:
        input_image (str):
        mask (str):
        dilate_amount (Union[Unset, int]):  Default: 10.
    """

    input_image: str
    mask: str
    dilate_amount: Union[Unset, int] = 10
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        input_image = self.input_image

        mask = self.mask

        dilate_amount = self.dilate_amount

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "input_image": input_image,
                "mask": mask,
            }
        )
        if dilate_amount is not UNSET:
            field_dict["dilate_amount"] = dilate_amount

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        input_image = d.pop("input_image")

        mask = d.pop("mask")

        dilate_amount = d.pop("dilate_amount", UNSET)

        dilate_mask_request = cls(
            input_image=input_image,
            mask=mask,
            dilate_amount=dilate_amount,
        )

        dilate_mask_request.additional_properties = d
        return dilate_mask_request

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
