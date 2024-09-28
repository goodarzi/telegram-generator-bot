from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="Person")


@_attrs_define
class Person:
    """
    Attributes:
        pose_keypoints_2d (List[float]):
        hand_right_keypoints_2d (Union[Unset, List[float]]):
        hand_left_keypoints_2d (Union[Unset, List[float]]):
        face_keypoints_2d (Union[Unset, List[float]]):
    """

    pose_keypoints_2d: List[float]
    hand_right_keypoints_2d: Union[Unset, List[float]] = UNSET
    hand_left_keypoints_2d: Union[Unset, List[float]] = UNSET
    face_keypoints_2d: Union[Unset, List[float]] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        pose_keypoints_2d = self.pose_keypoints_2d

        hand_right_keypoints_2d: Union[Unset, List[float]] = UNSET
        if not isinstance(self.hand_right_keypoints_2d, Unset):
            hand_right_keypoints_2d = self.hand_right_keypoints_2d

        hand_left_keypoints_2d: Union[Unset, List[float]] = UNSET
        if not isinstance(self.hand_left_keypoints_2d, Unset):
            hand_left_keypoints_2d = self.hand_left_keypoints_2d

        face_keypoints_2d: Union[Unset, List[float]] = UNSET
        if not isinstance(self.face_keypoints_2d, Unset):
            face_keypoints_2d = self.face_keypoints_2d

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "pose_keypoints_2d": pose_keypoints_2d,
            }
        )
        if hand_right_keypoints_2d is not UNSET:
            field_dict["hand_right_keypoints_2d"] = hand_right_keypoints_2d
        if hand_left_keypoints_2d is not UNSET:
            field_dict["hand_left_keypoints_2d"] = hand_left_keypoints_2d
        if face_keypoints_2d is not UNSET:
            field_dict["face_keypoints_2d"] = face_keypoints_2d

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        pose_keypoints_2d = cast(List[float], d.pop("pose_keypoints_2d"))

        hand_right_keypoints_2d = cast(List[float], d.pop("hand_right_keypoints_2d", UNSET))

        hand_left_keypoints_2d = cast(List[float], d.pop("hand_left_keypoints_2d", UNSET))

        face_keypoints_2d = cast(List[float], d.pop("face_keypoints_2d", UNSET))

        person = cls(
            pose_keypoints_2d=pose_keypoints_2d,
            hand_right_keypoints_2d=hand_right_keypoints_2d,
            hand_left_keypoints_2d=hand_left_keypoints_2d,
            face_keypoints_2d=face_keypoints_2d,
        )

        person.additional_properties = d
        return person

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
