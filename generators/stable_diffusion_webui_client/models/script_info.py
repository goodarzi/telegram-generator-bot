from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.script_arg import ScriptArg


T = TypeVar("T", bound="ScriptInfo")


@_attrs_define
class ScriptInfo:
    """
    Attributes:
        args (List['ScriptArg']): List of script's arguments
        name (Union[Unset, str]): Script name
        is_alwayson (Union[Unset, bool]): Flag specifying whether this script is an alwayson script
        is_img2img (Union[Unset, bool]): Flag specifying whether this script is an img2img script
    """

    args: List["ScriptArg"]
    name: Union[Unset, str] = UNSET
    is_alwayson: Union[Unset, bool] = UNSET
    is_img2img: Union[Unset, bool] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        args = []
        for args_item_data in self.args:
            args_item = args_item_data.to_dict()
            args.append(args_item)

        name = self.name

        is_alwayson = self.is_alwayson

        is_img2img = self.is_img2img

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "args": args,
            }
        )
        if name is not UNSET:
            field_dict["name"] = name
        if is_alwayson is not UNSET:
            field_dict["is_alwayson"] = is_alwayson
        if is_img2img is not UNSET:
            field_dict["is_img2img"] = is_img2img

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.script_arg import ScriptArg

        d = src_dict.copy()
        args = []
        _args = d.pop("args")
        for args_item_data in _args:
            args_item = ScriptArg.from_dict(args_item_data)

            args.append(args_item)

        name = d.pop("name", UNSET)

        is_alwayson = d.pop("is_alwayson", UNSET)

        is_img2img = d.pop("is_img2img", UNSET)

        script_info = cls(
            args=args,
            name=name,
            is_alwayson=is_alwayson,
            is_img2img=is_img2img,
        )

        script_info.additional_properties = d
        return script_info

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
