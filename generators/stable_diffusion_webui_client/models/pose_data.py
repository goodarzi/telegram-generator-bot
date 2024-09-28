from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.person import Person


T = TypeVar("T", bound="PoseData")


@_attrs_define
class PoseData:
    """
    Attributes:
        people (List['Person']):
        canvas_width (int):
        canvas_height (int):
    """

    people: List["Person"]
    canvas_width: int
    canvas_height: int
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        people = []
        for people_item_data in self.people:
            people_item = people_item_data.to_dict()
            people.append(people_item)

        canvas_width = self.canvas_width

        canvas_height = self.canvas_height

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "people": people,
                "canvas_width": canvas_width,
                "canvas_height": canvas_height,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.person import Person

        d = src_dict.copy()
        people = []
        _people = d.pop("people")
        for people_item_data in _people:
            people_item = Person.from_dict(people_item_data)

            people.append(people_item)

        canvas_width = d.pop("canvas_width")

        canvas_height = d.pop("canvas_height")

        pose_data = cls(
            people=people,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )

        pose_data.additional_properties = d
        return pose_data

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
