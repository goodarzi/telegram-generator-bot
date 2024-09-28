from typing import Any, Dict, List, Type, TypeVar, Union, cast

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
        hash_ (Union[None, str]):
        sha256 (Union[None, str]):
        filename (str):
        config (Union[None, Unset, str]):
    """

    title: str
    model_name: str
    hash_: Union[None, str]
    sha256: Union[None, str]
    filename: str
    config: Union[None, Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        title = self.title

        model_name = self.model_name

        hash_: Union[None, str]
        hash_ = self.hash_

        sha256: Union[None, str]
        sha256 = self.sha256

        filename = self.filename

        config: Union[None, Unset, str]
        if isinstance(self.config, Unset):
            config = UNSET
        else:
            config = self.config

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "title": title,
                "model_name": model_name,
                "hash": hash_,
                "sha256": sha256,
                "filename": filename,
            }
        )
        if config is not UNSET:
            field_dict["config"] = config

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        title = d.pop("title")

        model_name = d.pop("model_name")

        def _parse_hash_(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        hash_ = _parse_hash_(d.pop("hash"))

        def _parse_sha256(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        sha256 = _parse_sha256(d.pop("sha256"))

        filename = d.pop("filename")

        def _parse_config(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        config = _parse_config(d.pop("config", UNSET))

        sd_model_item = cls(
            title=title,
            model_name=model_name,
            hash_=hash_,
            sha256=sha256,
            filename=filename,
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
