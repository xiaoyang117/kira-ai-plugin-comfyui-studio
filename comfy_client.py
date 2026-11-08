"""ComfyUI HTTP client for the comfyui_studio plugin.

Talks to a ComfyUI server over its plain HTTP API:
- ``GET  /system_stats``          health check
- ``GET  /object_info[/{node}]``  node schemas (available checkpoints, samplers...)
- ``POST /prompt``                queue a prompt graph
- ``GET  /history/{prompt_id}``   poll execution results
- ``GET  /view``                  download generated images

Only the standard httpx async client is used; no websocket is needed since
history polling is fully adequate for one-shot generation jobs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx


@dataclass
class ComfyImageRef:
    """One output image reported by ComfyUI history."""

    filename: str
    subfolder: str = ""
    folder_type: str = "output"


@dataclass
class ComfyJobResult:
    """Outcome of waiting for a queued prompt."""

    ok: bool
    prompt_id: str = ""
    images: list[ComfyImageRef] = field(default_factory=list)
    error: str = ""

    @property
    def status(self) -> str:
        return "done" if self.ok else "failed"


class ComfyClientError(RuntimeError):
    """Raised when the ComfyUI server cannot be reached or misbehaves."""


class ComfyClient:
    """Small async wrapper around the ComfyUI REST API."""

    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout)

    # -- server info -------------------------------------------------

    async def system_stats(self) -> dict:
        """Return /system_stats; raises ComfyClientError when unreachable."""
        try:
            async with self._client() as client:
                resp = await client.get(f"{self.base_url}/system_stats")
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:  # noqa: BLE001 - normalize transport errors
            raise ComfyClientError(
                f"无法连接 ComfyUI（{self.base_url}）：{type(exc).__name__}: {exc}"
            ) from exc

    async def object_info(self, node: Optional[str] = None) -> dict:
        """Return node schemas, either all or for a single node class."""
        url = f"{self.base_url}/object_info"
        if node:
            url = f"{url}/{node}"
        try:
            async with self._client() as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:  # noqa: BLE001
            raise ComfyClientError(f"获取 object_info 失败：{exc}") from exc

    @staticmethod
    def _required_options(info: dict, node_class: str, input_key: str) -> list[str]:
        """Pull the static option list of one required combo input."""
        node = info.get(node_class, {})
        opts = node.get("input", {}).get("required", {}).get(input_key, [])
        if opts and isinstance(opts[0], list):
            return [str(x) for x in opts[0]]
        return []

    async def list_unets(self) -> list[str]:
        """List models available to UNETLoader (diffusion_models folder)."""
        info = await self.object_info("UNETLoader")
        return self._required_options(info, "UNETLoader", "unet_name")

    async def list_clips(self) -> list[str]:
        """List text encoders available to CLIPLoader."""
        info = await self.object_info("CLIPLoader")
        return self._required_options(info, "CLIPLoader", "clip_name")

    async def list_clip_types(self) -> list[str]:
        """List CLIPLoader type options (model architectures)."""
        info = await self.object_info("CLIPLoader")
        return self._required_options(info, "CLIPLoader", "type")

    async def list_vaes(self) -> list[str]:
        """List VAEs available to VAELoader."""
        info = await self.object_info("VAELoader")
        return self._required_options(info, "VAELoader", "vae_name")

    async def list_loras(self) -> list[str]:
        """List LoRAs available to LoraLoader."""
        info = await self.object_info("LoraLoader")
        return self._required_options(info, "LoraLoader", "lora_name")

    async def list_weight_dtypes(self) -> list[str]:
        """List UNETLoader weight_dtype options."""
        info = await self.object_info("UNETLoader")
        return self._required_options(info, "UNETLoader", "weight_dtype")

    async def list_sampler_names(self) -> list[str]:
        """List sampler names accepted by KSampler."""
        info = await self.object_info("KSampler")
        node = info.get("KSampler", {})
        opts = node.get("input", {}).get("required", {}).get("sampler_name", [])
        if opts and isinstance(opts[0], list):
            return [str(x) for x in opts[0]]
        return []

    async def list_scheduler_names(self) -> list[str]:
        """List scheduler names accepted by KSampler."""
        info = await self.object_info("KSampler")
        node = info.get("KSampler", {})
        opts = node.get("input", {}).get("required", {}).get("scheduler", [])
        if opts and isinstance(opts[0], list):
            return [str(x) for x in opts[0]]
        return []

    # -- generation --------------------------------------------------

    async def upload_input_image(self, data: bytes, filename: str) -> str:
        """Upload PNG input and return ComfyUI's input-relative image identifier."""
        try:
            async with self._client() as client:
                response = await client.post(
                    f"{self.base_url}/upload/image",
                    data={"type": "input", "overwrite": "false", "subfolder": "kira_studio"},
                    files={"image": (filename, data, "image/png")},
                )
                response.raise_for_status()
                result = response.json()
            name = str(result.get("name") or "")
            folder = str(result.get("subfolder") or "").strip("/")
            if not name or "/" in name or "\\" in name or name in (".", ".."):
                raise ValueError("invalid image name")
            if ".." in folder.split("/") or "\\" in folder:
                raise ValueError("invalid input subfolder")
            return f"{folder}/{name}" if folder else name
        except Exception as exc:
            raise ComfyClientError("上传图生图原图到 ComfyUI 失败，请检查服务连接") from exc

    async def submit(self, graph: dict, client_id: str) -> str:
        """Queue a prompt graph; returns the prompt_id."""
        try:
            async with self._client() as client:
                resp = await client.post(
                    f"{self.base_url}/prompt",
                    json={"prompt": graph, "client_id": client_id},
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            # ComfyUI returns 400 with node errors when the graph is invalid.
            detail = ""
            try:
                detail = str(exc.response.json())[:800]
            except Exception:  # noqa: BLE001
                detail = exc.response.text[:800] if exc.response is not None else ""
            raise ComfyClientError(f"提交工作流被拒绝：{detail}") from exc
        except Exception as exc:  # noqa: BLE001
            raise ComfyClientError(f"提交工作流失败：{exc}") from exc
        prompt_id = data.get("prompt_id", "")
        if not prompt_id:
            raise ComfyClientError(f"ComfyUI 未返回 prompt_id：{data}")
        return prompt_id

    async def wait_result(
        self,
        prompt_id: str,
        timeout: float,
        poll_interval: float = 1.5,
    ) -> ComfyJobResult:
        """Poll /history until the prompt finishes, fails or times out."""
        deadline = asyncio.get_event_loop().time() + timeout
        last_error = ""
        async with self._client() as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(f"{self.base_url}/history/{prompt_id}")
                    resp.raise_for_status()
                    history = resp.json()
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    await asyncio.sleep(poll_interval)
                    continue

                entry = history.get(prompt_id)
                if entry:
                    images = self._extract_images(entry)
                    status = entry.get("status", {}) or {}
                    status_str = str(status.get("status_str", "")).lower()
                    completed = bool(status.get("completed", False))
                    if images:
                        return ComfyJobResult(ok=True, prompt_id=prompt_id, images=images)
                    if status_str == "error" or (completed and not images):
                        messages = status.get("messages", []) or []
                        detail = ""
                        for msg in messages:
                            if isinstance(msg, list) and msg and msg[0] == "execution_error":
                                payload = msg[1] if len(msg) > 1 else {}
                                detail = str(payload.get("exception_message", ""))[:400]
                                break
                        return ComfyJobResult(
                            ok=False,
                            prompt_id=prompt_id,
                            error=detail or "ComfyUI 执行出错（无输出图像）",
                        )
                await asyncio.sleep(poll_interval)
        return ComfyJobResult(
            ok=False,
            prompt_id=prompt_id,
            error=last_error or f"等待 ComfyUI 结果超时（{timeout:.0f}s）",
        )

    @staticmethod
    def _extract_images(entry: dict) -> list[ComfyImageRef]:
        """Pull image refs out of a history entry's outputs."""
        refs: list[ComfyImageRef] = []
        outputs = entry.get("outputs", {}) or {}
        for node_out in outputs.values():
            if not isinstance(node_out, dict):
                continue
            for img in node_out.get("images", []) or []:
                if not isinstance(img, dict) or not img.get("filename"):
                    continue
                refs.append(
                    ComfyImageRef(
                        filename=str(img["filename"]),
                        subfolder=str(img.get("subfolder", "") or ""),
                        folder_type=str(img.get("type", "output") or "output"),
                    )
                )
        return refs

    async def download_image(self, ref: ComfyImageRef) -> bytes:
        """Download one generated image via /view."""
        params = {
            "filename": ref.filename,
            "subfolder": ref.subfolder,
            "type": ref.folder_type,
        }
        try:
            async with self._client() as client:
                resp = await client.get(f"{self.base_url}/view", params=params)
                resp.raise_for_status()
                return resp.content
        except Exception as exc:  # noqa: BLE001
            raise ComfyClientError(f"下载输出图片失败（{ref.filename}）：{exc}") from exc

    # -- helpers -----------------------------------------------------

    async def resolve_unet(self, configured: str) -> str:
        """Return a valid unet model name, falling back to the first one."""
        unets = await self.list_unets()
        if not unets:
            raise ComfyClientError("ComfyUI 没有可用绘图模型（UNETLoader 列表为空）")
        if configured:
            if configured not in unets:
                raise ComfyClientError(f"配置的绘图模型不存在：{configured}，请刷新模型列表后重新选择")
            return configured
        return unets[0]

    async def queue_status(self) -> dict:
        """Return the raw /queue payload (for the sidebar page)."""
        try:
            async with self._client() as client:
                resp = await client.get(f"{self.base_url}/queue")
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:  # noqa: BLE001
            raise ComfyClientError(f"获取队列状态失败：{exc}") from exc
