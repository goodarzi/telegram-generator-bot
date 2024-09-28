from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="BodyReactorImageReactorImagePost")


@_attrs_define
class BodyReactorImageReactorImagePost:
    """
    Attributes:
        source_image (Union[Unset, str]):  Default: ''.
        target_image (Union[Unset, str]):  Default: ''.
        source_faces_index (Union[Unset, List[int]]):
        face_index (Union[Unset, List[int]]):
        upscaler (Union[Unset, str]):  Default: 'None'.
        scale (Union[Unset, float]):  Default: 1.0.
        upscale_visibility (Union[Unset, float]):  Default: 1.0.
        face_restorer (Union[Unset, str]):  Default: 'None'.
        restorer_visibility (Union[Unset, float]):  Default: 1.0.
        codeformer_weight (Union[Unset, float]):  Default: 0.5.
        restore_first (Union[Unset, int]):  Default: 1.
        model (Union[Unset, str]):  Default: 'inswapper_128.onnx'.
        gender_source (Union[Unset, int]):  Default: 0.
        gender_target (Union[Unset, int]):  Default: 0.
        save_to_file (Union[Unset, int]):  Default: 0.
        result_file_path (Union[Unset, str]):  Default: ''.
        device (Union[Unset, str]):  Default: 'CPU'.
        mask_face (Union[Unset, int]):  Default: 0.
        select_source (Union[Unset, int]):  Default: 0.
        face_model (Union[Unset, str]):  Default: 'None'.
        source_folder (Union[Unset, str]):  Default: ''.
        random_image (Union[Unset, int]):  Default: 0.
        upscale_force (Union[Unset, int]):  Default: 0.
        det_thresh (Union[Unset, float]):  Default: 0.5.
        det_maxnum (Union[Unset, int]):  Default: 0.
    """

    source_image: Union[Unset, str] = ""
    target_image: Union[Unset, str] = ""
    source_faces_index: Union[Unset, List[int]] = UNSET
    face_index: Union[Unset, List[int]] = UNSET
    upscaler: Union[Unset, str] = "None"
    scale: Union[Unset, float] = 1.0
    upscale_visibility: Union[Unset, float] = 1.0
    face_restorer: Union[Unset, str] = "None"
    restorer_visibility: Union[Unset, float] = 1.0
    codeformer_weight: Union[Unset, float] = 0.5
    restore_first: Union[Unset, int] = 1
    model: Union[Unset, str] = "inswapper_128.onnx"
    gender_source: Union[Unset, int] = 0
    gender_target: Union[Unset, int] = 0
    save_to_file: Union[Unset, int] = 0
    result_file_path: Union[Unset, str] = ""
    device: Union[Unset, str] = "CPU"
    mask_face: Union[Unset, int] = 0
    select_source: Union[Unset, int] = 0
    face_model: Union[Unset, str] = "None"
    source_folder: Union[Unset, str] = ""
    random_image: Union[Unset, int] = 0
    upscale_force: Union[Unset, int] = 0
    det_thresh: Union[Unset, float] = 0.5
    det_maxnum: Union[Unset, int] = 0
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        source_image = self.source_image

        target_image = self.target_image

        source_faces_index: Union[Unset, List[int]] = UNSET
        if not isinstance(self.source_faces_index, Unset):
            source_faces_index = self.source_faces_index

        face_index: Union[Unset, List[int]] = UNSET
        if not isinstance(self.face_index, Unset):
            face_index = self.face_index

        upscaler = self.upscaler

        scale = self.scale

        upscale_visibility = self.upscale_visibility

        face_restorer = self.face_restorer

        restorer_visibility = self.restorer_visibility

        codeformer_weight = self.codeformer_weight

        restore_first = self.restore_first

        model = self.model

        gender_source = self.gender_source

        gender_target = self.gender_target

        save_to_file = self.save_to_file

        result_file_path = self.result_file_path

        device = self.device

        mask_face = self.mask_face

        select_source = self.select_source

        face_model = self.face_model

        source_folder = self.source_folder

        random_image = self.random_image

        upscale_force = self.upscale_force

        det_thresh = self.det_thresh

        det_maxnum = self.det_maxnum

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if source_image is not UNSET:
            field_dict["source_image"] = source_image
        if target_image is not UNSET:
            field_dict["target_image"] = target_image
        if source_faces_index is not UNSET:
            field_dict["source_faces_index"] = source_faces_index
        if face_index is not UNSET:
            field_dict["face_index"] = face_index
        if upscaler is not UNSET:
            field_dict["upscaler"] = upscaler
        if scale is not UNSET:
            field_dict["scale"] = scale
        if upscale_visibility is not UNSET:
            field_dict["upscale_visibility"] = upscale_visibility
        if face_restorer is not UNSET:
            field_dict["face_restorer"] = face_restorer
        if restorer_visibility is not UNSET:
            field_dict["restorer_visibility"] = restorer_visibility
        if codeformer_weight is not UNSET:
            field_dict["codeformer_weight"] = codeformer_weight
        if restore_first is not UNSET:
            field_dict["restore_first"] = restore_first
        if model is not UNSET:
            field_dict["model"] = model
        if gender_source is not UNSET:
            field_dict["gender_source"] = gender_source
        if gender_target is not UNSET:
            field_dict["gender_target"] = gender_target
        if save_to_file is not UNSET:
            field_dict["save_to_file"] = save_to_file
        if result_file_path is not UNSET:
            field_dict["result_file_path"] = result_file_path
        if device is not UNSET:
            field_dict["device"] = device
        if mask_face is not UNSET:
            field_dict["mask_face"] = mask_face
        if select_source is not UNSET:
            field_dict["select_source"] = select_source
        if face_model is not UNSET:
            field_dict["face_model"] = face_model
        if source_folder is not UNSET:
            field_dict["source_folder"] = source_folder
        if random_image is not UNSET:
            field_dict["random_image"] = random_image
        if upscale_force is not UNSET:
            field_dict["upscale_force"] = upscale_force
        if det_thresh is not UNSET:
            field_dict["det_thresh"] = det_thresh
        if det_maxnum is not UNSET:
            field_dict["det_maxnum"] = det_maxnum

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        source_image = d.pop("source_image", UNSET)

        target_image = d.pop("target_image", UNSET)

        source_faces_index = cast(List[int], d.pop("source_faces_index", UNSET))

        face_index = cast(List[int], d.pop("face_index", UNSET))

        upscaler = d.pop("upscaler", UNSET)

        scale = d.pop("scale", UNSET)

        upscale_visibility = d.pop("upscale_visibility", UNSET)

        face_restorer = d.pop("face_restorer", UNSET)

        restorer_visibility = d.pop("restorer_visibility", UNSET)

        codeformer_weight = d.pop("codeformer_weight", UNSET)

        restore_first = d.pop("restore_first", UNSET)

        model = d.pop("model", UNSET)

        gender_source = d.pop("gender_source", UNSET)

        gender_target = d.pop("gender_target", UNSET)

        save_to_file = d.pop("save_to_file", UNSET)

        result_file_path = d.pop("result_file_path", UNSET)

        device = d.pop("device", UNSET)

        mask_face = d.pop("mask_face", UNSET)

        select_source = d.pop("select_source", UNSET)

        face_model = d.pop("face_model", UNSET)

        source_folder = d.pop("source_folder", UNSET)

        random_image = d.pop("random_image", UNSET)

        upscale_force = d.pop("upscale_force", UNSET)

        det_thresh = d.pop("det_thresh", UNSET)

        det_maxnum = d.pop("det_maxnum", UNSET)

        body_reactor_image_reactor_image_post = cls(
            source_image=source_image,
            target_image=target_image,
            source_faces_index=source_faces_index,
            face_index=face_index,
            upscaler=upscaler,
            scale=scale,
            upscale_visibility=upscale_visibility,
            face_restorer=face_restorer,
            restorer_visibility=restorer_visibility,
            codeformer_weight=codeformer_weight,
            restore_first=restore_first,
            model=model,
            gender_source=gender_source,
            gender_target=gender_target,
            save_to_file=save_to_file,
            result_file_path=result_file_path,
            device=device,
            mask_face=mask_face,
            select_source=select_source,
            face_model=face_model,
            source_folder=source_folder,
            random_image=random_image,
            upscale_force=upscale_force,
            det_thresh=det_thresh,
            det_maxnum=det_maxnum,
        )

        body_reactor_image_reactor_image_post.additional_properties = d
        return body_reactor_image_reactor_image_post

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
