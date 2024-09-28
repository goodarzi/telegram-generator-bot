from typing import Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="DINOPredictRequest")


@_attrs_define
class DINOPredictRequest:
    """
    Attributes:
        input_image (str):
        text_prompt (str):
        dino_model_name (Union[Unset, str]):  Default: 'GroundingDINO_SwinT_OGC (694MB)'.
        box_threshold (Union[Unset, float]):  Default: 0.3.
    """

    input_image: str
    text_prompt: str
    dino_model_name: Union[Unset, str] = "GroundingDINO_SwinT_OGC (694MB)"
    box_threshold: Union[Unset, float] = 0.3
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        input_image = self.input_image

        text_prompt = self.text_prompt

        dino_model_name = self.dino_model_name

        box_threshold = self.box_threshold

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "input_image": input_image,
                "text_prompt": text_prompt,
            }
        )
        if dino_model_name is not UNSET:
            field_dict["dino_model_name"] = dino_model_name
        if box_threshold is not UNSET:
            field_dict["box_threshold"] = box_threshold

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        input_image = d.pop("input_image")

        text_prompt = d.pop("text_prompt")

        dino_model_name = d.pop("dino_model_name", UNSET)

        box_threshold = d.pop("box_threshold", UNSET)

        dino_predict_request = cls(
            input_image=input_image,
            text_prompt=text_prompt,
            dino_model_name=dino_model_name,
            box_threshold=box_threshold,
        )

        dino_predict_request.additional_properties = d
        return dino_predict_request

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
