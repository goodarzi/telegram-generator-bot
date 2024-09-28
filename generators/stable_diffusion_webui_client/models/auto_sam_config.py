from typing import Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="AutoSAMConfig")


@_attrs_define
class AutoSAMConfig:
    """
    Attributes:
        points_per_side (Union[Unset, int]):  Default: 32.
        points_per_batch (Union[Unset, int]):  Default: 64.
        pred_iou_thresh (Union[Unset, float]):  Default: 0.88.
        stability_score_thresh (Union[Unset, float]):  Default: 0.95.
        stability_score_offset (Union[Unset, float]):  Default: 1.0.
        box_nms_thresh (Union[Unset, float]):  Default: 0.7.
        crop_n_layers (Union[Unset, int]):  Default: 0.
        crop_nms_thresh (Union[Unset, float]):  Default: 0.7.
        crop_overlap_ratio (Union[Unset, float]):  Default: 0.3413333333333333.
        crop_n_points_downscale_factor (Union[Unset, int]):  Default: 1.
        min_mask_region_area (Union[Unset, int]):  Default: 0.
    """

    points_per_side: Union[Unset, int] = 32
    points_per_batch: Union[Unset, int] = 64
    pred_iou_thresh: Union[Unset, float] = 0.88
    stability_score_thresh: Union[Unset, float] = 0.95
    stability_score_offset: Union[Unset, float] = 1.0
    box_nms_thresh: Union[Unset, float] = 0.7
    crop_n_layers: Union[Unset, int] = 0
    crop_nms_thresh: Union[Unset, float] = 0.7
    crop_overlap_ratio: Union[Unset, float] = 0.3413333333333333
    crop_n_points_downscale_factor: Union[Unset, int] = 1
    min_mask_region_area: Union[Unset, int] = 0
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        points_per_side = self.points_per_side

        points_per_batch = self.points_per_batch

        pred_iou_thresh = self.pred_iou_thresh

        stability_score_thresh = self.stability_score_thresh

        stability_score_offset = self.stability_score_offset

        box_nms_thresh = self.box_nms_thresh

        crop_n_layers = self.crop_n_layers

        crop_nms_thresh = self.crop_nms_thresh

        crop_overlap_ratio = self.crop_overlap_ratio

        crop_n_points_downscale_factor = self.crop_n_points_downscale_factor

        min_mask_region_area = self.min_mask_region_area

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if points_per_side is not UNSET:
            field_dict["points_per_side"] = points_per_side
        if points_per_batch is not UNSET:
            field_dict["points_per_batch"] = points_per_batch
        if pred_iou_thresh is not UNSET:
            field_dict["pred_iou_thresh"] = pred_iou_thresh
        if stability_score_thresh is not UNSET:
            field_dict["stability_score_thresh"] = stability_score_thresh
        if stability_score_offset is not UNSET:
            field_dict["stability_score_offset"] = stability_score_offset
        if box_nms_thresh is not UNSET:
            field_dict["box_nms_thresh"] = box_nms_thresh
        if crop_n_layers is not UNSET:
            field_dict["crop_n_layers"] = crop_n_layers
        if crop_nms_thresh is not UNSET:
            field_dict["crop_nms_thresh"] = crop_nms_thresh
        if crop_overlap_ratio is not UNSET:
            field_dict["crop_overlap_ratio"] = crop_overlap_ratio
        if crop_n_points_downscale_factor is not UNSET:
            field_dict["crop_n_points_downscale_factor"] = crop_n_points_downscale_factor
        if min_mask_region_area is not UNSET:
            field_dict["min_mask_region_area"] = min_mask_region_area

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        points_per_side = d.pop("points_per_side", UNSET)

        points_per_batch = d.pop("points_per_batch", UNSET)

        pred_iou_thresh = d.pop("pred_iou_thresh", UNSET)

        stability_score_thresh = d.pop("stability_score_thresh", UNSET)

        stability_score_offset = d.pop("stability_score_offset", UNSET)

        box_nms_thresh = d.pop("box_nms_thresh", UNSET)

        crop_n_layers = d.pop("crop_n_layers", UNSET)

        crop_nms_thresh = d.pop("crop_nms_thresh", UNSET)

        crop_overlap_ratio = d.pop("crop_overlap_ratio", UNSET)

        crop_n_points_downscale_factor = d.pop("crop_n_points_downscale_factor", UNSET)

        min_mask_region_area = d.pop("min_mask_region_area", UNSET)

        auto_sam_config = cls(
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers,
            crop_nms_thresh=crop_nms_thresh,
            crop_overlap_ratio=crop_overlap_ratio,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
        )

        auto_sam_config.additional_properties = d
        return auto_sam_config

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
