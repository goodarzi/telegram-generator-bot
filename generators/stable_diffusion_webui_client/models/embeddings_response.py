from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.embeddings_response_loaded import EmbeddingsResponseLoaded
    from ..models.embeddings_response_skipped import EmbeddingsResponseSkipped


T = TypeVar("T", bound="EmbeddingsResponse")


@_attrs_define
class EmbeddingsResponse:
    """
    Attributes:
        loaded (EmbeddingsResponseLoaded): Embeddings loaded for the current model
        skipped (EmbeddingsResponseSkipped): Embeddings skipped for the current model (likely due to architecture
            incompatibility)
    """

    loaded: "EmbeddingsResponseLoaded"
    skipped: "EmbeddingsResponseSkipped"
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        loaded = self.loaded.to_dict()

        skipped = self.skipped.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "loaded": loaded,
                "skipped": skipped,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.embeddings_response_loaded import EmbeddingsResponseLoaded
        from ..models.embeddings_response_skipped import EmbeddingsResponseSkipped

        d = src_dict.copy()
        loaded = EmbeddingsResponseLoaded.from_dict(d.pop("loaded"))

        skipped = EmbeddingsResponseSkipped.from_dict(d.pop("skipped"))

        embeddings_response = cls(
            loaded=loaded,
            skipped=skipped,
        )

        embeddings_response.additional_properties = d
        return embeddings_response

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
