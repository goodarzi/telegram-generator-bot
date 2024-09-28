from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.extras_batch_images_request_resize_mode import ExtrasBatchImagesRequestResizeMode
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.file_data import FileData


T = TypeVar("T", bound="ExtrasBatchImagesRequest")


@_attrs_define
class ExtrasBatchImagesRequest:
    """
    Attributes:
        image_list (List['FileData']): List of images to work on. Must be Base64 strings
        resize_mode (Union[Unset, ExtrasBatchImagesRequestResizeMode]): Sets the resize mode: 0 to upscale by
            upscaling_resize amount, 1 to upscale up to upscaling_resize_h x upscaling_resize_w. Default:
            ExtrasBatchImagesRequestResizeMode.VALUE_0.
        show_extras_results (Union[Unset, bool]): Should the backend return the generated image? Default: True.
        gfpgan_visibility (Union[Unset, float]): Sets the visibility of GFPGAN, values should be between 0 and 1.
            Default: 0.0.
        codeformer_visibility (Union[Unset, float]): Sets the visibility of CodeFormer, values should be between 0 and
            1. Default: 0.0.
        codeformer_weight (Union[Unset, float]): Sets the weight of CodeFormer, values should be between 0 and 1.
            Default: 0.0.
        upscaling_resize (Union[Unset, float]): By how much to upscale the image, only used when resize_mode=0. Default:
            2.0.
        upscaling_resize_w (Union[Unset, int]): Target width for the upscaler to hit. Only used when resize_mode=1.
            Default: 512.
        upscaling_resize_h (Union[Unset, int]): Target height for the upscaler to hit. Only used when resize_mode=1.
            Default: 512.
        upscaling_crop (Union[Unset, bool]): Should the upscaler crop the image to fit in the chosen size? Default:
            True.
        upscaler_1 (Union[Unset, str]): The name of the main upscaler to use, it has to be one of this list:  Default:
            'None'.
        upscaler_2 (Union[Unset, str]): The name of the secondary upscaler to use, it has to be one of this list:
            Default: 'None'.
        extras_upscaler_2_visibility (Union[Unset, float]): Sets the visibility of secondary upscaler, values should be
            between 0 and 1. Default: 0.0.
        upscale_first (Union[Unset, bool]): Should the upscaler run before restoring faces? Default: False.
    """

    image_list: List["FileData"]
    resize_mode: Union[Unset, ExtrasBatchImagesRequestResizeMode] = ExtrasBatchImagesRequestResizeMode.VALUE_0
    show_extras_results: Union[Unset, bool] = True
    gfpgan_visibility: Union[Unset, float] = 0.0
    codeformer_visibility: Union[Unset, float] = 0.0
    codeformer_weight: Union[Unset, float] = 0.0
    upscaling_resize: Union[Unset, float] = 2.0
    upscaling_resize_w: Union[Unset, int] = 512
    upscaling_resize_h: Union[Unset, int] = 512
    upscaling_crop: Union[Unset, bool] = True
    upscaler_1: Union[Unset, str] = "None"
    upscaler_2: Union[Unset, str] = "None"
    extras_upscaler_2_visibility: Union[Unset, float] = 0.0
    upscale_first: Union[Unset, bool] = False
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        image_list = []
        for image_list_item_data in self.image_list:
            image_list_item = image_list_item_data.to_dict()
            image_list.append(image_list_item)

        resize_mode: Union[Unset, int] = UNSET
        if not isinstance(self.resize_mode, Unset):
            resize_mode = self.resize_mode.value

        show_extras_results = self.show_extras_results

        gfpgan_visibility = self.gfpgan_visibility

        codeformer_visibility = self.codeformer_visibility

        codeformer_weight = self.codeformer_weight

        upscaling_resize = self.upscaling_resize

        upscaling_resize_w = self.upscaling_resize_w

        upscaling_resize_h = self.upscaling_resize_h

        upscaling_crop = self.upscaling_crop

        upscaler_1 = self.upscaler_1

        upscaler_2 = self.upscaler_2

        extras_upscaler_2_visibility = self.extras_upscaler_2_visibility

        upscale_first = self.upscale_first

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "imageList": image_list,
            }
        )
        if resize_mode is not UNSET:
            field_dict["resize_mode"] = resize_mode
        if show_extras_results is not UNSET:
            field_dict["show_extras_results"] = show_extras_results
        if gfpgan_visibility is not UNSET:
            field_dict["gfpgan_visibility"] = gfpgan_visibility
        if codeformer_visibility is not UNSET:
            field_dict["codeformer_visibility"] = codeformer_visibility
        if codeformer_weight is not UNSET:
            field_dict["codeformer_weight"] = codeformer_weight
        if upscaling_resize is not UNSET:
            field_dict["upscaling_resize"] = upscaling_resize
        if upscaling_resize_w is not UNSET:
            field_dict["upscaling_resize_w"] = upscaling_resize_w
        if upscaling_resize_h is not UNSET:
            field_dict["upscaling_resize_h"] = upscaling_resize_h
        if upscaling_crop is not UNSET:
            field_dict["upscaling_crop"] = upscaling_crop
        if upscaler_1 is not UNSET:
            field_dict["upscaler_1"] = upscaler_1
        if upscaler_2 is not UNSET:
            field_dict["upscaler_2"] = upscaler_2
        if extras_upscaler_2_visibility is not UNSET:
            field_dict["extras_upscaler_2_visibility"] = extras_upscaler_2_visibility
        if upscale_first is not UNSET:
            field_dict["upscale_first"] = upscale_first

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.file_data import FileData

        d = src_dict.copy()
        image_list = []
        _image_list = d.pop("imageList")
        for image_list_item_data in _image_list:
            image_list_item = FileData.from_dict(image_list_item_data)

            image_list.append(image_list_item)

        _resize_mode = d.pop("resize_mode", UNSET)
        resize_mode: Union[Unset, ExtrasBatchImagesRequestResizeMode]
        if isinstance(_resize_mode, Unset):
            resize_mode = UNSET
        else:
            resize_mode = ExtrasBatchImagesRequestResizeMode(_resize_mode)

        show_extras_results = d.pop("show_extras_results", UNSET)

        gfpgan_visibility = d.pop("gfpgan_visibility", UNSET)

        codeformer_visibility = d.pop("codeformer_visibility", UNSET)

        codeformer_weight = d.pop("codeformer_weight", UNSET)

        upscaling_resize = d.pop("upscaling_resize", UNSET)

        upscaling_resize_w = d.pop("upscaling_resize_w", UNSET)

        upscaling_resize_h = d.pop("upscaling_resize_h", UNSET)

        upscaling_crop = d.pop("upscaling_crop", UNSET)

        upscaler_1 = d.pop("upscaler_1", UNSET)

        upscaler_2 = d.pop("upscaler_2", UNSET)

        extras_upscaler_2_visibility = d.pop("extras_upscaler_2_visibility", UNSET)

        upscale_first = d.pop("upscale_first", UNSET)

        extras_batch_images_request = cls(
            image_list=image_list,
            resize_mode=resize_mode,
            show_extras_results=show_extras_results,
            gfpgan_visibility=gfpgan_visibility,
            codeformer_visibility=codeformer_visibility,
            codeformer_weight=codeformer_weight,
            upscaling_resize=upscaling_resize,
            upscaling_resize_w=upscaling_resize_w,
            upscaling_resize_h=upscaling_resize_h,
            upscaling_crop=upscaling_crop,
            upscaler_1=upscaler_1,
            upscaler_2=upscaler_2,
            extras_upscaler_2_visibility=extras_upscaler_2_visibility,
            upscale_first=upscale_first,
        )

        extras_batch_images_request.additional_properties = d
        return extras_batch_images_request

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
