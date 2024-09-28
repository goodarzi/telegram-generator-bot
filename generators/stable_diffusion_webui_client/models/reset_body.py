from typing import Any, Dict, List, Type, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="ResetBody")


@_attrs_define
class ResetBody:
    """
    Attributes:
        session_hash (str):
        fn_index (int):
    """

    session_hash: str
    fn_index: int
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        session_hash = self.session_hash

        fn_index = self.fn_index

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "session_hash": session_hash,
                "fn_index": fn_index,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        session_hash = d.pop("session_hash")

        fn_index = d.pop("fn_index")

        reset_body = cls(
            session_hash=session_hash,
            fn_index=fn_index,
        )

        reset_body.additional_properties = d
        return reset_body

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
