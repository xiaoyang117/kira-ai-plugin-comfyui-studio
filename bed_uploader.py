"""Image-bed uploader for the comfyui_studio plugin.

The host at ``tc.fyszj.top`` is a small self-built service. Its upload API was
recovered from the site's own frontend bundle (``/assets/sakura-v1.js``):

- ``POST <base>/upload``  multipart/form-data, file field ``files``
- optional auth header ``X-Token: <upload token>``
- response JSON: ``{"ok": bool, "files": [{"name", "url", ...}], "errors": [...]}`
- the ``url`` of each entry is the direct link and starts with ``/f/``

Every parameter that could change on the server side (base URL, path, field
name, token, url field) is configurable from the plugin settings page.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import httpx

#: ``/f/xxx`` style relative link -> join with the base URL.
_RELATIVE_URL_RE = re.compile(r"^/")


class BedUploadError(RuntimeError):
    """Raised when the image bed rejects or fails an upload."""


def file_stem(name: str) -> str:
    """Return the filename without its extension."""
    return name.rsplit(".", 1)[0] if "." in name else name


async def upload_image(
    base_url: str,
    upload_path: str,
    file_field: str,
    token: str,
    file_bytes: bytes,
    filename: str,
    timeout: float = 120.0,
) -> Tuple[Optional[str], str]:
    """Upload one image to the bed.

    Returns ``(direct_url, error)`` — exactly one of the two is empty.
    """
    base = (base_url or "").rstrip("/")
    if not base:
        return None, "图床地址未配置"
    path = upload_path or "/upload"
    url = f"{base}{path if path.startswith('/') else '/' + path}"

    headers = {}
    if token:
        headers["X-Token"] = token

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                url,
                files={file_field or "files": (filename, file_bytes)},
                headers=headers,
            )
    except Exception as exc:  # noqa: BLE001 - normalize transport errors
        return None, f"上传图床失败：{exc}"

    if resp.status_code < 200 or resp.status_code >= 300:
        return None, f"图床返回 HTTP {resp.status_code}：{resp.text[:200]}"

    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return None, f"图床响应不是 JSON：{resp.text[:200]}"

    if not data.get("ok", False):
        return None, f"图床拒绝上传：{data.get('msg', '') or data}"

    files = data.get("files") or []
    if not files:
        errors = data.get("errors") or []
        return None, f"图床未保存文件：{errors or data}"

    first = files[0] if isinstance(files[0], dict) else {}
    link = str(first.get("url", "") or "")
    if not link:
        return None, f"图床响应缺少直链字段：{first}"

    if _RELATIVE_URL_RE.match(link):
        link = f"{base}{link}"
    return link, ""


async def test_upload(
    base_url: str,
    upload_path: str,
    file_field: str,
    token: str,
    timeout: float = 60.0,
) -> Tuple[Optional[str], str]:
    """Upload a 1x1 PNG to verify the configured bed settings work."""
    # Smallest valid PNG (1x1 transparent), generated locally - no external fetch.
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d4944415478da63fcffff3f030005fe02fea72d4b1e0000000049454e44ae426082"
    )
    return await upload_image(
        base_url,
        upload_path,
        file_field,
        token,
        png,
        "kira_studio_test.png",
        timeout=timeout,
    )
