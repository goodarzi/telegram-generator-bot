from typing import Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ControlNetSegRequest")


@_attrs_define
class ControlNetSegRequest:
    """
    Attributes:
        input_image (str):
        sam_model_name (Union[Unset, str]):  Default: 'sam_vit_h_4b8939.pth'.
        processor (Union[Unset, str]):  Default: 'seg_ofade20k'.
        processor_res (Union[Unset, int]):  Default: 512.
        pixel_perfect (Union[Unset, bool]):  Default: False.
        resize_mode (Union[Unset, int]):  Default: 1.
        target_w (Union[Unset, int]):
        target_h (Union[Unset, int]):
    """

    input_image: str
    sam_model_name: Union[Unset, str] = "sam_vit_h_4b8939.pth"
    processor: Union[Unset, str] = "seg_ofade20k"
    processor_res: Union[Unset, int] = 512
    pixel_perfect: Union[Unset, bool] = False
    resize_mode: Union[Unset, int] = 1
    target_w: Union[Unset, int] = UNSET
    target_h: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        input_image = self.input_image

        sam_model_name = self.sam_model_name

        processor = self.processor

        processor_res = self.processor_res

        pixel_perfect = self.pixel_perfect

        resize_mode = self.resize_mode

        target_w = self.target_w

        target_h = self.target_h

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "input_image": input_image,
            }
        )
        if sam_model_name is not UNSET:
            field_dict["sam_model_name"] = sam_model_name
        if processor is not UNSET:
            field_dict["processor"] = processor
        if processor_res is not UNSET:
            field_dict["processor_res"] = processor_res
        if pixel_perfect is not UNSET:
            field_dict["pixel_perfect"] = pixel_perfect
        if resize_mode is not UNSET:
            field_dict["resize_mode"] = resize_mode
        if target_w is not UNSET:
            field_dict["target_W"] = target_w
        if target_h is not UNSET:
            field_dict["target_H"] = target_h

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        input_image = d.pop("input_image")

        sam_model_name = d.pop("sam_model_name", UNSET)

        processor = d.pop("processor", UNSET)

        processor_res = d.pop("processor_res", UNSET)

        pixel_perfect = d.pop("pixel_perfect", UNSET)

        resize_mode = d.pop("resize_mode", UNSET)

        target_w = d.pop("target_W", UNSET)

        target_h = d.pop("target_H", UNSET)

        control_net_seg_request = cls(
            input_image=input_image,
            sam_model_name=sam_model_name,
            processor=processor,
            processor_res=processor_res,
            pixel_perfect=pixel_perfect,
            resize_mode=resize_mode,
            target_w=target_w,
            target_h=target_h,
        )

        control_net_seg_request.additional_properties = d
        return control_net_seg_request

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
