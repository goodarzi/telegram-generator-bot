from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    page: Union[Unset, str] = "",
    tabname: Union[Unset, str] = "",
    name: Union[Unset, str] = "",
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    params["page"] = page

    params["tabname"] = tabname

    params["name"] = name

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: Dict[str, Any] = {
        "method": "get",
        "url": "/sd_extra_networks/get-single-card",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, HTTPValidationError]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = response.json()
        return response_200
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[Any, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    page: Union[Unset, str] = "",
    tabname: Union[Unset, str] = "",
    name: Union[Unset, str] = "",
) -> Response[Union[Any, HTTPValidationError]]:
    """Get Single Card

    Args:
        page (Union[Unset, str]):  Default: ''.
        tabname (Union[Unset, str]):  Default: ''.
        name (Union[Unset, str]):  Default: ''.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        page=page,
        tabname=tabname,
        name=name,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    page: Union[Unset, str] = "",
    tabname: Union[Unset, str] = "",
    name: Union[Unset, str] = "",
) -> Optional[Union[Any, HTTPValidationError]]:
    """Get Single Card

    Args:
        page (Union[Unset, str]):  Default: ''.
        tabname (Union[Unset, str]):  Default: ''.
        name (Union[Unset, str]):  Default: ''.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        page=page,
        tabname=tabname,
        name=name,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    page: Union[Unset, str] = "",
    tabname: Union[Unset, str] = "",
    name: Union[Unset, str] = "",
) -> Response[Union[Any, HTTPValidationError]]:
    """Get Single Card

    Args:
        page (Union[Unset, str]):  Default: ''.
        tabname (Union[Unset, str]):  Default: ''.
        name (Union[Unset, str]):  Default: ''.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        page=page,
        tabname=tabname,
        name=name,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    page: Union[Unset, str] = "",
    tabname: Union[Unset, str] = "",
    name: Union[Unset, str] = "",
) -> Optional[Union[Any, HTTPValidationError]]:
    """Get Single Card

    Args:
        page (Union[Unset, str]):  Default: ''.
        tabname (Union[Unset, str]):  Default: ''.
        name (Union[Unset, str]):  Default: ''.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            page=page,
            tabname=tabname,
            name=name,
        )
    ).parsed
