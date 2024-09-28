from typing import Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="SDModelItem")


@_attrs_define
class SDModelItem:
    """
    Attributes:
        title (str):
        model_name (str):
        filename (str):
        hash_ (Union[Unset, str]):
        sha256 (Union[Unset, str]):
        config (Union[Unset, str]):
    """

    title: str
    model_name: str
    filename: str
    hash_: Union[Unset, str] = UNSET
    sha256: Union[Unset, str] = UNSET
    config: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        title = self.title

        model_name = self.model_name

        filename = self.filename

        hash_ = self.hash_

        sha256 = self.sha256

        config = self.config

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "title": title,
                "model_name": model_name,
                "filename": filename,
            }
        )
        if hash_ is not UNSET:
            field_dict["hash"] = hash_
        if sha256 is not UNSET:
            field_dict["sha256"] = sha256
        if config is not UNSET:
            field_dict["config"] = config

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        title = d.pop("title")

        model_name = d.pop("model_name")

        filename = d.pop("filename")

        hash_ = d.pop("hash", UNSET)

        sha256 = d.pop("sha256", UNSET)

        config = d.pop("config", UNSET)

        sd_model_item = cls(
            title=title,
            model_name=model_name,
            filename=filename,
            hash_=hash_,
            sha256=sha256,
            config=config,
        )

        sd_model_item.additional_properties = d
        return sd_model_item

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
