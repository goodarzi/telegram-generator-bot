from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ScriptArg")


@_attrs_define
class ScriptArg:
    """
    Attributes:
        label (Union[Unset, str]): Name of the argument in UI
        value (Union[Unset, Any]): Default value of the argument
        minimum (Union[Unset, Any]): Minimum allowed value for the argumentin UI
        maximum (Union[Unset, Any]): Maximum allowed value for the argumentin UI
        step (Union[Unset, Any]): Step for changing value of the argumentin UI
        choices (Union[Unset, List[str]]): Possible values for the argument
    """

    label: Union[Unset, str] = UNSET
    value: Union[Unset, Any] = UNSET
    minimum: Union[Unset, Any] = UNSET
    maximum: Union[Unset, Any] = UNSET
    step: Union[Unset, Any] = UNSET
    choices: Union[Unset, List[str]] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        label = self.label

        value = self.value

        minimum = self.minimum

        maximum = self.maximum

        step = self.step

        choices: Union[Unset, List[str]] = UNSET
        if not isinstance(self.choices, Unset):
            choices = self.choices

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if label is not UNSET:
            field_dict["label"] = label
        if value is not UNSET:
            field_dict["value"] = value
        if minimum is not UNSET:
            field_dict["minimum"] = minimum
        if maximum is not UNSET:
            field_dict["maximum"] = maximum
        if step is not UNSET:
            field_dict["step"] = step
        if choices is not UNSET:
            field_dict["choices"] = choices

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        label = d.pop("label", UNSET)

        value = d.pop("value", UNSET)

        minimum = d.pop("minimum", UNSET)

        maximum = d.pop("maximum", UNSET)

        step = d.pop("step", UNSET)

        choices = cast(List[str], d.pop("choices", UNSET))

        script_arg = cls(
            label=label,
            value=value,
            minimum=minimum,
            maximum=maximum,
            step=step,
            choices=choices,
        )

        script_arg.additional_properties = d
        return script_arg

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
