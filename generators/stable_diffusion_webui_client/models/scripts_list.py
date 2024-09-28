from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ScriptsList")


@_attrs_define
class ScriptsList:
    """
    Attributes:
        txt2img (Union[Unset, List[Any]]): Titles of scripts (txt2img)
        img2img (Union[Unset, List[Any]]): Titles of scripts (img2img)
    """

    txt2img: Union[Unset, List[Any]] = UNSET
    img2img: Union[Unset, List[Any]] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        txt2img: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.txt2img, Unset):
            txt2img = self.txt2img

        img2img: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.img2img, Unset):
            img2img = self.img2img

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if txt2img is not UNSET:
            field_dict["txt2img"] = txt2img
        if img2img is not UNSET:
            field_dict["img2img"] = img2img

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        txt2img = cast(List[Any], d.pop("txt2img", UNSET))

        img2img = cast(List[Any], d.pop("img2img", UNSET))

        scripts_list = cls(
            txt2img=txt2img,
            img2img=img2img,
        )

        scripts_list.additional_properties = d
        return scripts_list

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
