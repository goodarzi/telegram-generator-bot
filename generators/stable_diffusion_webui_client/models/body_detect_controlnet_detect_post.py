from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="BodyDetectControlnetDetectPost")


@_attrs_define
class BodyDetectControlnetDetectPost:
    """
    Attributes:
        controlnet_module (Union[Unset, str]):  Default: 'none'.
        controlnet_input_images (Union[Unset, List[str]]):
        controlnet_processor_res (Union[Unset, int]):  Default: -1.
        controlnet_threshold_a (Union[Unset, float]):  Default: -1.0.
        controlnet_threshold_b (Union[Unset, float]):  Default: -1.0.
        controlnet_masks (Union[Unset, List[str]]):
        low_vram (Union[Unset, bool]):  Default: False.
    """

    controlnet_module: Union[Unset, str] = "none"
    controlnet_input_images: Union[Unset, List[str]] = UNSET
    controlnet_processor_res: Union[Unset, int] = -1
    controlnet_threshold_a: Union[Unset, float] = -1.0
    controlnet_threshold_b: Union[Unset, float] = -1.0
    controlnet_masks: Union[Unset, List[str]] = UNSET
    low_vram: Union[Unset, bool] = False
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        controlnet_module = self.controlnet_module

        controlnet_input_images: Union[Unset, List[str]] = UNSET
        if not isinstance(self.controlnet_input_images, Unset):
            controlnet_input_images = self.controlnet_input_images

        controlnet_processor_res = self.controlnet_processor_res

        controlnet_threshold_a = self.controlnet_threshold_a

        controlnet_threshold_b = self.controlnet_threshold_b

        controlnet_masks: Union[Unset, List[str]] = UNSET
        if not isinstance(self.controlnet_masks, Unset):
            controlnet_masks = self.controlnet_masks

        low_vram = self.low_vram

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if controlnet_module is not UNSET:
            field_dict["controlnet_module"] = controlnet_module
        if controlnet_input_images is not UNSET:
            field_dict["controlnet_input_images"] = controlnet_input_images
        if controlnet_processor_res is not UNSET:
            field_dict["controlnet_processor_res"] = controlnet_processor_res
        if controlnet_threshold_a is not UNSET:
            field_dict["controlnet_threshold_a"] = controlnet_threshold_a
        if controlnet_threshold_b is not UNSET:
            field_dict["controlnet_threshold_b"] = controlnet_threshold_b
        if controlnet_masks is not UNSET:
            field_dict["controlnet_masks"] = controlnet_masks
        if low_vram is not UNSET:
            field_dict["low_vram"] = low_vram

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        controlnet_module = d.pop("controlnet_module", UNSET)

        controlnet_input_images = cast(List[str], d.pop("controlnet_input_images", UNSET))

        controlnet_processor_res = d.pop("controlnet_processor_res", UNSET)

        controlnet_threshold_a = d.pop("controlnet_threshold_a", UNSET)

        controlnet_threshold_b = d.pop("controlnet_threshold_b", UNSET)

        controlnet_masks = cast(List[str], d.pop("controlnet_masks", UNSET))

        low_vram = d.pop("low_vram", UNSET)

        body_detect_controlnet_detect_post = cls(
            controlnet_module=controlnet_module,
            controlnet_input_images=controlnet_input_images,
            controlnet_processor_res=controlnet_processor_res,
            controlnet_threshold_a=controlnet_threshold_a,
            controlnet_threshold_b=controlnet_threshold_b,
            controlnet_masks=controlnet_masks,
            low_vram=low_vram,
        )

        body_detect_controlnet_detect_post.additional_properties = d
        return body_detect_controlnet_detect_post

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
