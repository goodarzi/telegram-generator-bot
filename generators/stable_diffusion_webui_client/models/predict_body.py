from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.predict_body_request_type_0 import PredictBodyRequestType0
    from ..models.predict_body_request_type_1_item import PredictBodyRequestType1Item


T = TypeVar("T", bound="PredictBody")


@_attrs_define
class PredictBody:
    """
    Attributes:
        data (List[Any]):
        session_hash (Union[Unset, str]):
        event_id (Union[Unset, str]):
        event_data (Union[Unset, Any]):
        fn_index (Union[Unset, int]):
        batched (Union[Unset, bool]):  Default: False.
        request (Union['PredictBodyRequestType0', List['PredictBodyRequestType1Item'], Unset]):
    """

    data: List[Any]
    session_hash: Union[Unset, str] = UNSET
    event_id: Union[Unset, str] = UNSET
    event_data: Union[Unset, Any] = UNSET
    fn_index: Union[Unset, int] = UNSET
    batched: Union[Unset, bool] = False
    request: Union["PredictBodyRequestType0", List["PredictBodyRequestType1Item"], Unset] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        from ..models.predict_body_request_type_0 import PredictBodyRequestType0

        data = self.data

        session_hash = self.session_hash

        event_id = self.event_id

        event_data = self.event_data

        fn_index = self.fn_index

        batched = self.batched

        request: Union[Dict[str, Any], List[Dict[str, Any]], Unset]
        if isinstance(self.request, Unset):
            request = UNSET
        elif isinstance(self.request, PredictBodyRequestType0):
            request = self.request.to_dict()
        else:
            request = []
            for request_type_1_item_data in self.request:
                request_type_1_item = request_type_1_item_data.to_dict()
                request.append(request_type_1_item)

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
        if batched is not UNSET:
            field_dict["batched"] = batched
        if request is not UNSET:
            field_dict["request"] = request

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.predict_body_request_type_0 import PredictBodyRequestType0
        from ..models.predict_body_request_type_1_item import PredictBodyRequestType1Item

        d = src_dict.copy()
        data = cast(List[Any], d.pop("data"))

        session_hash = d.pop("session_hash", UNSET)

        event_id = d.pop("event_id", UNSET)

        event_data = d.pop("event_data", UNSET)

        fn_index = d.pop("fn_index", UNSET)

        batched = d.pop("batched", UNSET)

        def _parse_request(
            data: object,
        ) -> Union["PredictBodyRequestType0", List["PredictBodyRequestType1Item"], Unset]:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                request_type_0 = PredictBodyRequestType0.from_dict(data)

                return request_type_0
            except:  # noqa: E722
                pass
            if not isinstance(data, list):
                raise TypeError()
            request_type_1 = []
            _request_type_1 = data
            for request_type_1_item_data in _request_type_1:
                request_type_1_item = PredictBodyRequestType1Item.from_dict(request_type_1_item_data)

                request_type_1.append(request_type_1_item)

            return request_type_1

        request = _parse_request(d.pop("request", UNSET))

        predict_body = cls(
            data=data,
            session_hash=session_hash,
            event_id=event_id,
            event_data=event_data,
            fn_index=fn_index,
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
