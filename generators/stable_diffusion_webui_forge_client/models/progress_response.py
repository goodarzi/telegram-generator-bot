from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.progress_response_state import ProgressResponseState


T = TypeVar("T", bound="ProgressResponse")


@_attrs_define
class ProgressResponse:
    """
    Attributes:
        progress (float): The progress with a range of 0 to 1
        eta_relative (float):
        state (ProgressResponseState): The current state snapshot
        current_image (Union[None, Unset, str]): The current image in base64 format. opts.show_progress_every_n_steps is
            required for this to work.
        textinfo (Union[None, Unset, str]): Info text used by WebUI.
    """

    progress: float
    eta_relative: float
    state: "ProgressResponseState"
    current_image: Union[None, Unset, str] = UNSET
    textinfo: Union[None, Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        progress = self.progress

        eta_relative = self.eta_relative

        state = self.state.to_dict()

        current_image: Union[None, Unset, str]
        if isinstance(self.current_image, Unset):
            current_image = UNSET
        else:
            current_image = self.current_image

        textinfo: Union[None, Unset, str]
        if isinstance(self.textinfo, Unset):
            textinfo = UNSET
        else:
            textinfo = self.textinfo

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "progress": progress,
                "eta_relative": eta_relative,
                "state": state,
            }
        )
        if current_image is not UNSET:
            field_dict["current_image"] = current_image
        if textinfo is not UNSET:
            field_dict["textinfo"] = textinfo

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.progress_response_state import ProgressResponseState

        d = src_dict.copy()
        progress = d.pop("progress")

        eta_relative = d.pop("eta_relative")

        state = ProgressResponseState.from_dict(d.pop("state"))

        def _parse_current_image(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        current_image = _parse_current_image(d.pop("current_image", UNSET))

        def _parse_textinfo(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        textinfo = _parse_textinfo(d.pop("textinfo", UNSET))

        progress_response = cls(
            progress=progress,
            eta_relative=eta_relative,
            state=state,
            current_image=current_image,
            textinfo=textinfo,
        )

        progress_response.additional_properties = d
        return progress_response

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
