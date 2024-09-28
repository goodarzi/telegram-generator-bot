from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="PromptStyleItem")


@_attrs_define
class PromptStyleItem:
    """
    Attributes:
        name (str):
        prompt (Union[None, str]):
        negative_prompt (Union[None, str]):
    """

    name: str
    prompt: Union[None, str]
    negative_prompt: Union[None, str]
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name

        prompt: Union[None, str]
        prompt = self.prompt

        negative_prompt: Union[None, str]
        negative_prompt = self.negative_prompt

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        def _parse_prompt(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        prompt = _parse_prompt(d.pop("prompt"))

        def _parse_negative_prompt(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        negative_prompt = _parse_negative_prompt(d.pop("negative_prompt"))

        prompt_style_item = cls(
            name=name,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

        prompt_style_item.additional_properties = d
        return prompt_style_item

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
