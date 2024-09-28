from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.predict_body_data_item import PredictBodyDataItem
    from ..models.predict_body_event_data import PredictBodyEventData
    from ..models.predict_body_request import PredictBodyRequest


T = TypeVar("T", bound="PredictBody")


@_attrs_define
class PredictBody:
    """
    Attributes:
        data (List['PredictBodyDataItem']):
        session_hash (Union[Unset, str]):
        event_id (Union[Unset, str]):
        event_data (Union[Unset, PredictBodyEventData]):
        fn_index (Union[Unset, int]):
        trigger_id (Union[Unset, int]):
        simple_format (Union[Unset, bool]):
        batched (Union[Unset, bool]):
        request (Union[Unset, PredictBodyRequest]):
    """

    data: List["PredictBodyDataItem"]
    session_hash: Union[Unset, str] = UNSET
    event_id: Union[Unset, str] = UNSET
    event_data: Union[Unset, "PredictBodyEventData"] = UNSET
    fn_index: Union[Unset, int] = UNSET
    trigger_id: Union[Unset, int] = UNSET
    simple_format: Union[Unset, bool] = UNSET
    batched: Union[Unset, bool] = UNSET
    request: Union[Unset, "PredictBodyRequest"] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = []
        for data_item_data in self.data:
            data_item = data_item_data.to_dict()
            data.append(data_item)

        session_hash = self.session_hash

        event_id = self.event_id

        event_data: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.event_data, Unset):
            event_data = self.event_data.to_dict()

        fn_index = self.fn_index

        trigger_id = self.trigger_id

        simple_format = self.simple_format

        batched = self.batched

        request: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.request, Unset):
            request = self.request.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "data": data,
            }
        )
        if session_hash is not UNSET:
            field_dict["session_hash"] = session_hash
        if event_id is not UNSET:
            field_dict["event_id"] = event_id
        if event_data is not UNSET:
            field_dict["event_data"] = event_data
        if fn_index is not UNSET:
            field_dict["fn_index"] = fn_index
        if trigger_id is not UNSET:
            field_dict["trigger_id"] = trigger_id
        if simple_format is not UNSET:
            field_dict["simple_format"] = simple_format
        if batched is not UNSET:
            field_dict["batched"] = batched
        if request is not UNSET:
            field_dict["request"] = request

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.predict_body_data_item import PredictBodyDataItem
        from ..models.predict_body_event_data import PredictBodyEventData
        from ..models.predict_body_request import PredictBodyRequest

        d = src_dict.copy()
        data = []
        _data = d.pop("data")
        for data_item_data in _data:
            data_item = PredictBodyDataItem.from_dict(data_item_data)

            data.append(data_item)

        session_hash = d.pop("session_hash", UNSET)

        event_id = d.pop("event_id", UNSET)

        _event_data = d.pop("event_data", UNSET)
        event_data: Union[Unset, PredictBodyEventData]
        if isinstance(_event_data, Unset):
            event_data = UNSET
        else:
            event_data = PredictBodyEventData.from_dict(_event_data)

        fn_index = d.pop("fn_index", UNSET)

        trigger_id = d.pop("trigger_id", UNSET)

        simple_format = d.pop("simple_format", UNSET)

        batched = d.pop("batched", UNSET)

        _request = d.pop("request", UNSET)
        request: Union[Unset, PredictBodyRequest]
        if isinstance(_request, Unset):
            request = UNSET
        else:
            request = PredictBodyRequest.from_dict(_request)

        predict_body = cls(
            data=data,
            session_hash=session_hash,
            event_id=event_id,
            event_data=event_data,
            fn_index=fn_index,
            trigger_id=trigger_id,
            simple_format=simple_format,
            batched=batched,
            request=request,
        )

        predict_body.additional_properties = d
        return predict_body

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
