from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.estimation_message_msg import EstimationMessageMsg
from ..types import UNSET, Unset

T = TypeVar("T", bound="EstimationMessage")


@_attrs_define
class EstimationMessage:
    """
    Attributes:
        queue_size (int):
        msg (Union[Unset, EstimationMessageMsg]):  Default: EstimationMessageMsg.ESTIMATION.
        event_id (Union[None, Unset, str]):
        rank (Union[None, Unset, int]):
        rank_eta (Union[None, Unset, float]):
    """

    queue_size: int
    msg: Union[Unset, EstimationMessageMsg] = EstimationMessageMsg.ESTIMATION
    event_id: Union[None, Unset, str] = UNSET
    rank: Union[None, Unset, int] = UNSET
    rank_eta: Union[None, Unset, float] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        queue_size = self.queue_size

        msg: Union[Unset, str] = UNSET
        if not isinstance(self.msg, Unset):
            msg = self.msg.value

        event_id: Union[None, Unset, str]
        if isinstance(self.event_id, Unset):
            event_id = UNSET
        else:
            event_id = self.event_id

        rank: Union[None, Unset, int]
        if isinstance(self.rank, Unset):
            rank = UNSET
        else:
            rank = self.rank

        rank_eta: Union[None, Unset, float]
        if isinstance(self.rank_eta, Unset):
            rank_eta = UNSET
        else:
            rank_eta = self.rank_eta

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "queue_size": queue_size,
            }
        )
        if msg is not UNSET:
            field_dict["msg"] = msg
        if event_id is not UNSET:
            field_dict["event_id"] = event_id
        if rank is not UNSET:
            field_dict["rank"] = rank
        if rank_eta is not UNSET:
            field_dict["rank_eta"] = rank_eta

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        queue_size = d.pop("queue_size")

        _msg = d.pop("msg", UNSET)
        msg: Union[Unset, EstimationMessageMsg]
        if isinstance(_msg, Unset):
            msg = UNSET
        else:
            msg = EstimationMessageMsg(_msg)

        def _parse_event_id(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        event_id = _parse_event_id(d.pop("event_id", UNSET))

        def _parse_rank(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        rank = _parse_rank(d.pop("rank", UNSET))

        def _parse_rank_eta(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        rank_eta = _parse_rank_eta(d.pop("rank_eta", UNSET))

        estimation_message = cls(
            queue_size=queue_size,
            msg=msg,
            event_id=event_id,
            rank=rank,
            rank_eta=rank_eta,
        )

        estimation_message.additional_properties = d
        return estimation_message

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
