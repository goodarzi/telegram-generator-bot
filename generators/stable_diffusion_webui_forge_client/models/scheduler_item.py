from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="SchedulerItem")


@_attrs_define
class SchedulerItem:
    """
    Attributes:
        name (str):
        label (str):
        aliases (Union[List[str], None]):
        default_rho (Union[None, float]):
        need_inner_model (Union[None, bool]):
    """

    name: str
    label: str
    aliases: Union[List[str], None]
    default_rho: Union[None, float]
    need_inner_model: Union[None, bool]
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        label = self.label

        aliases: Union[List[str], None]
        if isinstance(self.aliases, list):
            aliases = self.aliases

        else:
            aliases = self.aliases

        default_rho: Union[None, float]
        default_rho = self.default_rho

        need_inner_model: Union[None, bool]
        need_inner_model = self.need_inner_model

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "label": label,
                "aliases": aliases,
                "default_rho": default_rho,
                "need_inner_model": need_inner_model,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        label = d.pop("label")

        def _parse_aliases(data: object) -> Union[List[str], None]:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                aliases_type_0 = cast(List[str], data)

                return aliases_type_0
            except:  # noqa: E722
                pass
            return cast(Union[List[str], None], data)

        aliases = _parse_aliases(d.pop("aliases"))

        def _parse_default_rho(data: object) -> Union[None, float]:
            if data is None:
                return data
            return cast(Union[None, float], data)

        default_rho = _parse_default_rho(d.pop("default_rho"))

        def _parse_need_inner_model(data: object) -> Union[None, bool]:
            if data is None:
                return data
            return cast(Union[None, bool], data)

        need_inner_model = _parse_need_inner_model(d.pop("need_inner_model"))

        scheduler_item = cls(
            name=name,
            label=label,
            aliases=aliases,
            default_rho=default_rho,
            need_inner_model=need_inner_model,
        )

        scheduler_item.additional_properties = d
        return scheduler_item

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
