from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.embedding_item import EmbeddingItem


T = TypeVar("T", bound="EmbeddingsResponseLoaded")


@_attrs_define
class EmbeddingsResponseLoaded:
    """Embeddings loaded for the current model"""

    additional_properties: Dict[str, "EmbeddingItem"] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        field_dict: Dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.embedding_item import EmbeddingItem

        d = src_dict.copy()
        embeddings_response_loaded = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = EmbeddingItem.from_dict(prop_dict)

            additional_properties[prop_name] = additional_property

        embeddings_response_loaded.additional_properties = additional_properties
        return embeddings_response_loaded

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> "EmbeddingItem":
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: "EmbeddingItem") -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
