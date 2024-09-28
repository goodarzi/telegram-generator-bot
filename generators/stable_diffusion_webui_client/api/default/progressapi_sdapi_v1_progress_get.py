from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.progress_response import ProgressResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    skip_current_image: Union[Unset, bool] = False,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    params["skip_current_image"] = skip_current_image

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: Dict[str, Any] = {
        "method": "get",
        "url": "/sdapi/v1/progress",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, ProgressResponse]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = ProgressResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, ProgressResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    skip_current_image: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, ProgressResponse]]:
    """Progressapi

    Args:
        skip_current_image (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, ProgressResponse]]
    """

    kwargs = _get_kwargs(
        skip_current_image=skip_current_image,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    skip_current_image: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, ProgressResponse]]:
    """Progressapi

    Args:
        skip_current_image (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, ProgressResponse]
    """

    return sync_detailed(
        client=client,
        skip_current_image=skip_current_image,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    skip_current_image: Union[Unset, bool] = False,
) -> Response[Union[HTTPValidationError, ProgressResponse]]:
    """Progressapi

    Args:
        skip_current_image (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, ProgressResponse]]
    """

    kwargs = _get_kwargs(
        skip_current_image=skip_current_image,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    skip_current_image: Union[Unset, bool] = False,
) -> Optional[Union[HTTPValidationError, ProgressResponse]]:
    """Progressapi

    Args:
        skip_current_image (Union[Unset, bool]):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, ProgressResponse]
    """

    return (
        await asyncio_detailed(
            client=client,
            skip_current_image=skip_current_image,
        )
    ).parsed
