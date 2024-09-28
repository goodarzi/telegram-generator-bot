from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="SimplePredictBody")


@_attrs_define
class SimplePredictBody:
    """
    Attributes:
        data (List[Any]):
        session_hash (Union[None, Unset, str]):
    """

    data: List[Any]
    session_hash: Union[None, Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = self.data

        session_hash: Union[None, Unset, str]
        if isinstance(self.session_hash, Unset):
            session_hash = UNSET
        else:
            session_hash = self.session_hash

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "data": data,
            }
        )
        if session_hash is not UNSET:
            field_dict["session_hash"] = session_hash

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        data = cast(List[Any], d.pop("data"))

        def _parse_session_hash(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        session_hash = _parse_session_hash(d.pop("session_hash", UNSET))

        simple_predict_body = cls(
            data=data,
            session_hash=session_hash,
        )

        simple_predict_body.additional_properties = d
        return simple_predict_body

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
