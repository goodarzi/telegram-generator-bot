from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="EmbeddingItem")


@_attrs_define
class EmbeddingItem:
    """
    Attributes:
        step (Union[None, int]): The number of steps that were used to train this embedding, if available
        sd_checkpoint (Union[None, str]): The hash of the checkpoint this embedding was trained on, if available
        sd_checkpoint_name (Union[None, str]): The name of the checkpoint this embedding was trained on, if available.
            Note that this is the name that was used by the trainer; for a stable identifier, use `sd_checkpoint` instead
        shape (int): The length of each individual vector in the embedding
        vectors (int): The number of vectors in the embedding
    """

    step: Union[None, int]
    sd_checkpoint: Union[None, str]
    sd_checkpoint_name: Union[None, str]
    shape: int
    vectors: int
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        step: Union[None, int]
        step = self.step

        sd_checkpoint: Union[None, str]
        sd_checkpoint = self.sd_checkpoint

        sd_checkpoint_name: Union[None, str]
        sd_checkpoint_name = self.sd_checkpoint_name

        shape = self.shape

        vectors = self.vectors

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "step": step,
                "sd_checkpoint": sd_checkpoint,
                "sd_checkpoint_name": sd_checkpoint_name,
                "shape": shape,
                "vectors": vectors,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()

        def _parse_step(data: object) -> Union[None, int]:
            if data is None:
                return data
            return cast(Union[None, int], data)

        step = _parse_step(d.pop("step"))

        def _parse_sd_checkpoint(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        sd_checkpoint = _parse_sd_checkpoint(d.pop("sd_checkpoint"))

        def _parse_sd_checkpoint_name(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        sd_checkpoint_name = _parse_sd_checkpoint_name(d.pop("sd_checkpoint_name"))

        shape = d.pop("shape")

        vectors = d.pop("vectors")

        embedding_item = cls(
            step=step,
            sd_checkpoint=sd_checkpoint,
            sd_checkpoint_name=sd_checkpoint_name,
            shape=shape,
            vectors=vectors,
        )

        embedding_item.additional_properties = d
        return embedding_item

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
