from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.sampler_item_options import SamplerItemOptions


T = TypeVar("T", bound="SamplerItem")


@_attrs_define
class SamplerItem:
    """
    Attributes:
        name (str):
        aliases (List[str]):
        options (SamplerItemOptions):
    """

    name: str
    aliases: List[str]
    options: "SamplerItemOptions"
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        aliases = self.aliases

        options = self.options.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "aliases": aliases,
                "options": options,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.sampler_item_options import SamplerItemOptions

        d = src_dict.copy()
        name = d.pop("name")

        aliases = cast(List[str], d.pop("aliases"))

        options = SamplerItemOptions.from_dict(d.pop("options"))

        sampler_item = cls(
            name=name,
            aliases=aliases,
            options=options,
        )

        sampler_item.additional_properties = d
        return sampler_item

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
