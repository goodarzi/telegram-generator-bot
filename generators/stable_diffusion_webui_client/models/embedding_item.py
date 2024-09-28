from typing import Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="EmbeddingItem")


@_attrs_define
class EmbeddingItem:
    """
    Attributes:
        shape (int): The length of each individual vector in the embedding
        vectors (int): The number of vectors in the embedding
        step (Union[Unset, int]): The number of steps that were used to train this embedding, if available
        sd_checkpoint (Union[Unset, str]): The hash of the checkpoint this embedding was trained on, if available
        sd_checkpoint_name (Union[Unset, str]): The name of the checkpoint this embedding was trained on, if available.
            Note that this is the name that was used by the trainer; for a stable identifier, use `sd_checkpoint` instead
    """

    shape: int
    vectors: int
    step: Union[Unset, int] = UNSET
    sd_checkpoint: Union[Unset, str] = UNSET
    sd_checkpoint_name: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        shape = self.shape

        vectors = self.vectors

        step = self.step

        sd_checkpoint = self.sd_checkpoint

        sd_checkpoint_name = self.sd_checkpoint_name

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "shape": shape,
                "vectors": vectors,
            }
        )
        if step is not UNSET:
            field_dict["step"] = step
        if sd_checkpoint is not UNSET:
            field_dict["sd_checkpoint"] = sd_checkpoint
        if sd_checkpoint_name is not UNSET:
            field_dict["sd_checkpoint_name"] = sd_checkpoint_name

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        shape = d.pop("shape")

        vectors = d.pop("vectors")

        step = d.pop("step", UNSET)

        sd_checkpoint = d.pop("sd_checkpoint", UNSET)

        sd_checkpoint_name = d.pop("sd_checkpoint_name", UNSET)

        embedding_item = cls(
            shape=shape,
            vectors=vectors,
            step=step,
            sd_checkpoint=sd_checkpoint,
            sd_checkpoint_name=sd_checkpoint_name,
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
