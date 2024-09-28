from typing import Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="Estimation")


@_attrs_define
class Estimation:
    """
    Attributes:
        queue_size (int):
        queue_eta (float):
        msg (Union[Unset, str]):  Default: 'estimation'.
        rank (Union[Unset, int]):
        avg_event_process_time (Union[Unset, float]):
        avg_event_concurrent_process_time (Union[Unset, float]):
        rank_eta (Union[Unset, float]):
    """

    queue_size: int
    queue_eta: float
    msg: Union[Unset, str] = "estimation"
    rank: Union[Unset, int] = UNSET
    avg_event_process_time: Union[Unset, float] = UNSET
    avg_event_concurrent_process_time: Union[Unset, float] = UNSET
    rank_eta: Union[Unset, float] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        queue_size = self.queue_size

        queue_eta = self.queue_eta

        msg = self.msg

        rank = self.rank

        avg_event_process_time = self.avg_event_process_time

        avg_event_concurrent_process_time = self.avg_event_concurrent_process_time

        rank_eta = self.rank_eta

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "queue_size": queue_size,
                "queue_eta": queue_eta,
            }
        )
        if msg is not UNSET:
            field_dict["msg"] = msg
        if rank is not UNSET:
            field_dict["rank"] = rank
        if avg_event_process_time is not UNSET:
            field_dict["avg_event_process_time"] = avg_event_process_time
        if avg_event_concurrent_process_time is not UNSET:
            field_dict["avg_event_concurrent_process_time"] = avg_event_concurrent_process_time
        if rank_eta is not UNSET:
            field_dict["rank_eta"] = rank_eta

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        queue_size = d.pop("queue_size")

        queue_eta = d.pop("queue_eta")

        msg = d.pop("msg", UNSET)

        rank = d.pop("rank", UNSET)

        avg_event_process_time = d.pop("avg_event_process_time", UNSET)

        avg_event_concurrent_process_time = d.pop("avg_event_concurrent_process_time", UNSET)

        rank_eta = d.pop("rank_eta", UNSET)

        estimation = cls(
            queue_size=queue_size,
            queue_eta=queue_eta,
            msg=msg,
            rank=rank,
            avg_event_process_time=avg_event_process_time,
            avg_event_concurrent_process_time=avg_event_concurrent_process_time,
            rank_eta=rank_eta,
        )

        estimation.additional_properties = d
        return estimation

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
