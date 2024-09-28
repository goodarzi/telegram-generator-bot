from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.auto_sam_config import AutoSAMConfig
    from ..models.control_net_seg_request import ControlNetSegRequest


T = TypeVar("T", bound="BodyApiControlnetSegSamControlnetSegPost")


@_attrs_define
class BodyApiControlnetSegSamControlnetSegPost:
    """
    Attributes:
        payload (ControlNetSegRequest):
        autosam_conf (AutoSAMConfig):
    """

    payload: "ControlNetSegRequest"
    autosam_conf: "AutoSAMConfig"
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = self.payload.to_dict()

        autosam_conf = self.autosam_conf.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "payload": payload,
                "autosam_conf": autosam_conf,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.auto_sam_config import AutoSAMConfig
        from ..models.control_net_seg_request import ControlNetSegRequest

        d = src_dict.copy()
        payload = ControlNetSegRequest.from_dict(d.pop("payload"))

        autosam_conf = AutoSAMConfig.from_dict(d.pop("autosam_conf"))

        body_api_controlnet_seg_sam_controlnet_seg_post = cls(
            payload=payload,
            autosam_conf=autosam_conf,
        )

        body_api_controlnet_seg_sam_controlnet_seg_post.additional_properties = d
        return body_api_controlnet_seg_sam_controlnet_seg_post

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
