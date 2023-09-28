import pytest
import websockets


paths = [
    '/',
    "/api/datasets/detect/",
    "/api/datasets/schema/",
]


@pytest.mark.parametrize(
    'path', paths,
)
@pytest.mark.parametrize(
    'method', [
        'GET',
        'POST',
        'PUT',
        'DELETE',
    ]
)
@pytest.mark.asyncio
async def test_missing_token(base_url, path, method, http_client):
    full_url_without_token = base_url + path

    async with http_client.request(method, full_url_without_token) as resp:
        assert resp.status == 400 or resp.status == 405, \
            f"response code must be 'bad request' or 'method not allowed', got '{resp.status}'"


@pytest.mark.parametrize(
    'path', paths,
)
@pytest.mark.parametrize(
    'method', [
        'GET',
        'POST',
        'PUT',
        'DELETE',
    ]
)
@pytest.mark.asyncio
async def test_wrong_token(base_url, path, method, http_client):
    full_url_wrong_token = base_url + path + "?token=wrong"

    async with http_client.request(method, full_url_wrong_token) as resp:
        assert resp.status == 400 or resp.status == 405, \
            f"response code must be 'bad request' or 'method not allowed', got '{resp.status}'"


@pytest.mark.asyncio
async def test_token_for_ws_events(base_url, http_client, server_port):
    # fails without token:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/"
    with pytest.raises(websockets.InvalidStatusCode):
        async with websockets.connect(ws_url):
            pass


@pytest.mark.asyncio
async def test_no_token_server(base_url_no_token, http_client):
    # works without token:
    full_url = base_url_no_token + "/"
    async with http_client.get(full_url) as resp:
        assert resp.status == 200


# slow variants of some of the tests above:

paths_full = paths + [
    "/api/datasets/ds-uuid-here/",
    "/api/browse/localfs/",
    "/api/jobs/job-id-here/",
    "/api/compoundAnalyses/compound-uuid-here/analyses/analysis-uuid-here/",
    "/api/compoundAnalyses/compiund-uuid-here/analyses/analysis-uuid-here/download/HDF5/",
    "/api/compoundAnalyses/uuid-here/copy/notebook/",
    "/api/compoundAnalyses/uuid-here/download/notebook/",
    "/api/compoundAnalyses/uuid-here/",
    "/api/events/",
    "/api/shutdown/",
    "/api/config/",
    "/api/config/cluster/",
    "/api/config/connection/",
]


@pytest.mark.web_api
@pytest.mark.parametrize(
    'path', paths_full,
)
@pytest.mark.parametrize(
    'method', [
        'GET',
        'POST',
        'PUT',
        'DELETE',
        'HEAD',
        'PATCH',
        'OPTIONS',
    ]
)
@pytest.mark.asyncio
async def test_wrong_token_full(base_url, path, method, http_client):
    full_url_wrong_token = base_url + path + "?token=wrong"

    async with http_client.request(method, full_url_wrong_token) as resp:
        assert resp.status == 400 or resp.status == 405, \
            f"response code must be 'bad request' or 'method not allowed', got '{resp.status}'"


@pytest.mark.web_api
@pytest.mark.parametrize(
    'path', paths_full,
)
@pytest.mark.parametrize(
    'method', [
        'GET',
        'POST',
        'PUT',
        'DELETE',
        'HEAD',
        'PATCH',
        'OPTIONS',
    ]
)
@pytest.mark.asyncio
async def test_missing_token_full(base_url, path, method, http_client):
    full_url_without_token = base_url + path

    async with http_client.request(method, full_url_without_token) as resp:
        assert resp.status == 400 or resp.status == 405, \
            f"response code must be 'bad request' or 'method not allowed', got '{resp.status}'"
