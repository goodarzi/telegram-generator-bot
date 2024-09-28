from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="SamPredictRequest")


@_attrs_define
class SamPredictRequest:
    """
    Attributes:
        input_image (str):
        sam_model_name (Union[Unset, str]):  Default: 'sam_vit_h_4b8939.pth'.
        sam_positive_points (Union[Unset, List[List[float]]]):
        sam_negative_points (Union[Unset, List[List[float]]]):
        dino_enabled (Union[Unset, bool]):  Default: False.
        dino_model_name (Union[Unset, str]):  Default: 'GroundingDINO_SwinT_OGC (694MB)'.
        dino_text_prompt (Union[Unset, str]):
        dino_box_threshold (Union[Unset, float]):  Default: 0.3.
        dino_preview_checkbox (Union[Unset, bool]):  Default: False.
        dino_preview_boxes_selection (Union[Unset, List[int]]):
    """

    input_image: str
    sam_model_name: Union[Unset, str] = "sam_vit_h_4b8939.pth"
    sam_positive_points: Union[Unset, List[List[float]]] = UNSET
    sam_negative_points: Union[Unset, List[List[float]]] = UNSET
    dino_enabled: Union[Unset, bool] = False
    dino_model_name: Union[Unset, str] = "GroundingDINO_SwinT_OGC (694MB)"
    dino_text_prompt: Union[Unset, str] = UNSET
    dino_box_threshold: Union[Unset, float] = 0.3
    dino_preview_checkbox: Union[Unset, bool] = False
    dino_preview_boxes_selection: Union[Unset, List[int]] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        input_image = self.input_image

        sam_model_name = self.sam_model_name

        sam_positive_points: Union[Unset, List[List[float]]] = UNSET
        if not isinstance(self.sam_positive_points, Unset):
            sam_positive_points = []
            for sam_positive_points_item_data in self.sam_positive_points:
                sam_positive_points_item = sam_positive_points_item_data

                sam_positive_points.append(sam_positive_points_item)

        sam_negative_points: Union[Unset, List[List[float]]] = UNSET
        if not isinstance(self.sam_negative_points, Unset):
            sam_negative_points = []
            for sam_negative_points_item_data in self.sam_negative_points:
                sam_negative_points_item = sam_negative_points_item_data

                sam_negative_points.append(sam_negative_points_item)

        dino_enabled = self.dino_enabled

        dino_model_name = self.dino_model_name

        dino_text_prompt = self.dino_text_prompt

        dino_box_threshold = self.dino_box_threshold

        dino_preview_checkbox = self.dino_preview_checkbox

        dino_preview_boxes_selection: Union[Unset, List[int]] = UNSET
        if not isinstance(self.dino_preview_boxes_selection, Unset):
            dino_preview_boxes_selection = self.dino_preview_boxes_selection

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "input_image": input_image,
            }
        )
        if sam_model_name is not UNSET:
            field_dict["sam_model_name"] = sam_model_name
        if sam_positive_points is not UNSET:
            field_dict["sam_positive_points"] = sam_positive_points
        if sam_negative_points is not UNSET:
            field_dict["sam_negative_points"] = sam_negative_points
        if dino_enabled is not UNSET:
            field_dict["dino_enabled"] = dino_enabled
        if dino_model_name is not UNSET:
            field_dict["dino_model_name"] = dino_model_name
        if dino_text_prompt is not UNSET:
            field_dict["dino_text_prompt"] = dino_text_prompt
        if dino_box_threshold is not UNSET:
            field_dict["dino_box_threshold"] = dino_box_threshold
        if dino_preview_checkbox is not UNSET:
            field_dict["dino_preview_checkbox"] = dino_preview_checkbox
        if dino_preview_boxes_selection is not UNSET:
            field_dict["dino_preview_boxes_selection"] = dino_preview_boxes_selection

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        input_image = d.pop("input_image")

        sam_model_name = d.pop("sam_model_name", UNSET)

        sam_positive_points = []
        _sam_positive_points = d.pop("sam_positive_points", UNSET)
        for sam_positive_points_item_data in _sam_positive_points or []:
            sam_positive_points_item = cast(List[float], sam_positive_points_item_data)

            sam_positive_points.append(sam_positive_points_item)

        sam_negative_points = []
        _sam_negative_points = d.pop("sam_negative_points", UNSET)
        for sam_negative_points_item_data in _sam_negative_points or []:
            sam_negative_points_item = cast(List[float], sam_negative_points_item_data)

            sam_negative_points.append(sam_negative_points_item)

        dino_enabled = d.pop("dino_enabled", UNSET)

        dino_model_name = d.pop("dino_model_name", UNSET)

        dino_text_prompt = d.pop("dino_text_prompt", UNSET)

        dino_box_threshold = d.pop("dino_box_threshold", UNSET)

        dino_preview_checkbox = d.pop("dino_preview_checkbox", UNSET)

        dino_preview_boxes_selection = cast(List[int], d.pop("dino_preview_boxes_selection", UNSET))

        sam_predict_request = cls(
            input_image=input_image,
            sam_model_name=sam_model_name,
            sam_positive_points=sam_positive_points,
            sam_negative_points=sam_negative_points,
            dino_enabled=dino_enabled,
            dino_model_name=dino_model_name,
            dino_text_prompt=dino_text_prompt,
            dino_box_threshold=dino_box_threshold,
            dino_preview_checkbox=dino_preview_checkbox,
            dino_preview_boxes_selection=dino_preview_boxes_selection,
        )

        sam_predict_request.additional_properties = d
        return sam_predict_request

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
