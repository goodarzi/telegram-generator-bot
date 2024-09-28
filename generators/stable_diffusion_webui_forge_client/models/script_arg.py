from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ScriptArg")


@_attrs_define
class ScriptArg:
    """
    Attributes:
        label (Union[None, Unset, str]): Name of the argument in UI
        value (Union[Any, None, Unset]): Default value of the argument
        minimum (Union[Any, None, Unset]): Minimum allowed value for the argumentin UI
        maximum (Union[Any, None, Unset]): Maximum allowed value for the argumentin UI
        step (Union[Any, None, Unset]): Step for changing value of the argumentin UI
        choices (Union[List[str], None, Unset]): Possible values for the argument
    """

    label: Union[None, Unset, str] = UNSET
    value: Union[Any, None, Unset] = UNSET
    minimum: Union[Any, None, Unset] = UNSET
    maximum: Union[Any, None, Unset] = UNSET
    step: Union[Any, None, Unset] = UNSET
    choices: Union[List[str], None, Unset] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        label: Union[None, Unset, str]
        if isinstance(self.label, Unset):
            label = UNSET
        else:
            label = self.label

        value: Union[Any, None, Unset]
        if isinstance(self.value, Unset):
            value = UNSET
        else:
            value = self.value

        minimum: Union[Any, None, Unset]
        if isinstance(self.minimum, Unset):
            minimum = UNSET
        else:
            minimum = self.minimum

        maximum: Union[Any, None, Unset]
        if isinstance(self.maximum, Unset):
            maximum = UNSET
        else:
            maximum = self.maximum

        step: Union[Any, None, Unset]
        if isinstance(self.step, Unset):
            step = UNSET
        else:
            step = self.step

        choices: Union[List[str], None, Unset]
        if isinstance(self.choices, Unset):
            choices = UNSET
        elif isinstance(self.choices, list):
            choices = self.choices

        else:
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

        def _parse_label(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        label = _parse_label(d.pop("label", UNSET))

        def _parse_value(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        value = _parse_value(d.pop("value", UNSET))

        def _parse_minimum(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        minimum = _parse_minimum(d.pop("minimum", UNSET))

        def _parse_maximum(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        maximum = _parse_maximum(d.pop("maximum", UNSET))

        def _parse_step(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        step = _parse_step(d.pop("step", UNSET))

        def _parse_choices(data: object) -> Union[List[str], None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                choices_type_0 = cast(List[str], data)

                return choices_type_0
            except:  # noqa: E722
                pass
            return cast(Union[List[str], None, Unset], data)

        choices = _parse_choices(d.pop("choices", UNSET))

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
