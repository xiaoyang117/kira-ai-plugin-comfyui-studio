"""ComfyUI Studio - LLM-driven image generation plugin for KiraAI.

Design:
- The LLM decides everything creative (prompt content, resolution, count) via a
  registered tool. Built-in positive/negative quality words are injected by the
  plugin so the LLM only writes the actual scene description.
- AI jobs send local image files directly without progress messages or image-bed uploads.
- Command jobs retain configured image-bed / QQ markdown delivery and progress notices.
- ``/comfy WxH [count] prompt`` is a direct bypass command that skips the LLM.
- Tool calls have a hard 60s framework timeout (asyncio.wait_for + cancel), so
  the tool only waits ``sync_wait_seconds`` for the job; if it is still running
  the job continues in a background task and the result is sent directly later.
"""

import asyncio
import copy
import json
import math
import random
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse

from core.chat.message_elements import Image, Text
from core.chat.message_utils import KiraMessageEvent, MessageChain
from core.utils.network import download_file
from core.utils.path_utils import get_data_path
from core.plugin import (
    BasePlugin,
    PluginContext,
    PluginPage,
    PageMenu,
    Priority,
    logger,
    on,
    register,
)

from .bed_uploader import test_upload, upload_image
from .comfy_client import ComfyClient, ComfyClientError
from .image_inputs import ImageInputStore, ImageInputError
from .chat_settings import ChatSettingsMixin, SettingsError

PLUGIN_ID = "comfyui_studio"

#: Built-in quality words (kept verbatim from the user's original workflow).
DEFAULT_POSITIVE_QUALITY = (
    "best quality, score_7, score_9, nsfw, sensitive, very aesthetic, "
    "ultra detailed, pale skin"
)
DEFAULT_NEGATIVE_QUALITY = (
    "worst quality, low quality, score_1, score_2, score_3, blurry, "
    "jpeg artifacts, sepia, mosaic censoring, bar censor, censored, "
    "bad anatomy, bad hands"
)

PLUGIN_DIR = Path(__file__).resolve().parent
WORKFLOW_TEMPLATE = PLUGIN_DIR / "workflows" / "txt2img.json"

#: How long the LoRA name list (injected into the tool description) stays fresh.
_LORA_CACHE_TTL = 300.0


#: Module-level job registry: survives plugin re-init (config save hot-reload).
_JOBS: list[dict] = []
_TASKS: set[asyncio.Task] = set()
_RECENT_INPUTS: dict[tuple[str, str], dict] = {}
_JOBS_MAX = 50

#: ``/comfy [#model] 1024x1024 [count] [lora:strength,...] prompt...``
#: The LoRA bracket must contain a colon (strength is mandatory), so that
#: prompts which merely start with ``[...]`` are never mis-parsed.
_CMD_RE = re.compile(
    r"^(?:#\s*(?P<model>[^\s\[]+)\s+)?"
    r"(?P<width>\d{2,5})\s*[xX\u00d7*]\s*(?P<height>\d{2,5})"
    r"(?:\s+(?P<count>\d{1,2}))?"
    r"(?:\s+\[(?P<loras>[^\]]*:[^\]]*)\])?"
    r"\s+(?P<prompt>[\s\S]+)$"
)

TOOL_DESCRIPTION = (
    "调用本机 ComfyUI 绘制图片，画完自动把图片文件发到当前会话，无需再发 Markdown 或图片链接。"
    "prompt：只描述画面本身（人物、发型、服装、动作、表情、场景、构图、光线等），"
    "不要写任何质量词（画质词由系统按所选模型自动附加）。"
    "prompt 的写法要跟随 model：动漫/插画系预设用英文 danbooru 标签（逗号分隔）；"
    "写实/通用大模型预设（如 qwen 系）用自然语言完整句子，中文英文都可以，"
    "写得越具体越好（这类模型还支持在图里渲染文字，需要文字时直接描述）。"
    "width/height：图片宽与高（像素、8 的倍数），根据画面选合适比例——"
    "方图 1024x1024，竖图 832x1216 或 896x1152，横图 1216x832 或 1152x896；"
    "超出管理员设定的像素上限会被等比缩小。"
    "count：一次生成几张（1 到管理员设定的单次上限，默认 1），"
    "用户说“多来几张”时才调大。"
    "model：可选的模型预设（可用列表见下方系统提示），按画面风格选最合适的，"
    "不传则用管理员设置的默认模型。"
    "用户要求修改已有图片、换画风或局部修改时，应改用 comfyui_img2img 工具。"
)

IMG2IMG_DESCRIPTION = (
    "基于用户提供的图片进行图生图、风格转换或局部重绘；不能用文生图代替。"
    "原图优先取当前消息或引用消息中的图片，否则取当前会话同一用户最近发送的图片。"
    "image_index 从 1 开始选择原图；source_image 可填用户给出的图片链接或上传编号，"
    "不要编造链接或本地路径。没有原图时请用户发图。"
    "prompt 描述希望修改后的画面，语言跟随模型；denoise 越低越接近原图，越高改动越大，"
    "通常 0.35~0.75。width/height 留 0 时按原图比例自动确定尺寸。"
    "mask_image 可传与原图同尺寸的黑白蒙版：白色修改、黑色保留。"
    "蒙版和参考图来自聊天附件时，使用 mask_index 和 reference_indices 选择图片序号。"
    "模型预设和 LoRA 与文生图共用；Qwen 原生编辑需要选择对应的编辑模型预设。"
    "Qwen Edit Plus 和 Qwen Image 2.1 支持 reference_images（最多两张额外参考图链接或上传编号）。"
)

IMG2IMG_PARAMS = {
    "type": "object", "required": ["prompt"],
    "properties": {
        "prompt": {"type": "string", "description": "修改目标或修改后画面，不写质量词"},
        "source_image": {"type": "string", "description": "可选图片链接或 input:上传编号；留空取聊天图片"},
        "image_index": {"type": "integer", "minimum": 1, "description": "第几张聊天图片，默认 1"},
        "denoise": {"type": "number", "minimum": 0, "maximum": 1, "description": "重绘幅度 0~1；省略使用配置默认值"},
        "width": {"type": "integer", "minimum": 0, "description": "输出宽度；0 跟随原图"},
        "height": {"type": "integer", "minimum": 0, "description": "输出高度；0 跟随原图"},
        "count": {"type": "integer", "minimum": 1, "description": "输出张数，默认 1"},
        "model": {"type": "string", "description": "模型预设编号或名称，省略使用默认模型"},
        "seed": {"type": "integer", "minimum": 0, "maximum": 281474976710655, "description": "可选固定种子，不填则随机"},
        "resize_mode": {"type": "string", "enum": ["contain", "crop", "stretch"], "description": "留白适配、居中裁切、拉伸；省略使用配置"},
        "mask_image": {"type": "string", "description": "可选黑白蒙版链接或 input:上传编号"},
        "mask_index": {"type": "integer", "minimum": 1, "description": "可选蒙版在聊天附件中的序号；与 mask_image 二选一"},
        "reference_indices": {"type": "array", "maxItems": 2, "items": {"type": "integer", "minimum": 1}, "description": "额外参考图在聊天附件中的序号，如 [2,3]"},
        "reference_images": {"type": "array", "maxItems": 2, "items": {"type": "string"}, "description": "Qwen Edit Plus / 2.1 的额外参考图链接或上传编号"},
        "loras": {"type": "array", "items": {"type": "object", "required": ["name"], "properties": {
            "name": {"type": "string"}, "strength_model": {"type": "number"}, "strength_clip": {"type": "number"}
        }}, "description": "可选 LoRA，规则与文生图一致"},
    },
}


def _now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _new_job(source: str, session: str, prompt: str, width: int, height: int,
             count: int) -> dict:
    job = {
        "id": uuid.uuid4().hex[:12],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "session": session,
        "prompt": prompt,
        "width": width,
        "height": height,
        "count": count,
        "status": "pending",
        "urls": [],
        "files": [],
        "loras": [],
        "error": "",
        "delivered": False,
        "deliver_directly": False,
        "done": asyncio.Event(),
    }
    _JOBS.append(job)
    while len(_JOBS) > _JOBS_MAX:
        _JOBS.pop(0)
    return job


def _job_public(job: dict) -> dict:
    """Serializable view of a job record (for the sidebar page)."""
    return {k: v for k, v in job.items() if k != "done" and not k.startswith("_")}


class ComfyStudioPlugin(ChatSettingsMixin, BasePlugin):
    """ComfyUI generation with markdown delivery and a sidebar config page."""

    def __init__(self, ctx: PluginContext, cfg: dict):
        super().__init__(ctx, cfg)
        self._tasks = _TASKS
        self._data_dir = Path(ctx.get_plugin_data_dir())
        self._lora_cache: tuple[float, list[str]] = (0.0, [])
        self._deliver_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def initialize(self):
        logger.info(
            "[comfyui_studio] loaded (comfy=%s, send_mode=%s, bed=%s)",
            self.plugin_cfg.get("comfy_url"),
            self.plugin_cfg.get("send_mode", "markdown"),
            bool(self.plugin_cfg.get("bed_enabled", True)),
        )

    async def terminate(self):
        manager = getattr(self.ctx, "plugin_mgr", None)
        next_cfg = getattr(manager, "plugin_configs", {}).get(PLUGIN_ID)
        # Config saves replace the instance, but must not discard active jobs.
        if (next_cfg is not None and next_cfg is not self.plugin_cfg
                and next_cfg.get("enabled", True)
                and manager.is_plugin_enabled(PLUGIN_ID)):
            return
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    # ------------------------------------------------------------------
    # config helpers
    # ------------------------------------------------------------------
    def _int(self, key: str, default: int) -> int:
        try:
            return int(self.plugin_cfg.get(key, default))
        except (TypeError, ValueError):
            return default

    def _float(self, key: str, default: float) -> float:
        try:
            return float(self.plugin_cfg.get(key, default))
        except (TypeError, ValueError):
            return default

    @property
    def _send_mode(self) -> str:
        return str(self.plugin_cfg.get("send_mode", "markdown") or "markdown")

    def _input_store(self) -> ImageInputStore:
        return ImageInputStore(
            self._data_dir / "inputs",
            max_bytes=max(1, min(100, self._int("img2img_max_input_mb", 20))) * 1048576,
            max_pixels=max(4096, min(100000000, self._int("img2img_max_input_pixels", 40000000))),
            downloader=download_file, timeout=min(45, self._float("request_timeout", 120)),
        )

    @staticmethod
    def _chain_images(chain, *, archived=False, depth=0) -> list:
        if depth > 2:
            return []
        direct, quoted = [], []
        for element in chain or []:
            if isinstance(element, dict):
                kind = element.get("type")
                if kind in ("image", "file") and (kind == "image" or str(element.get("mime", "")).startswith("image/")):
                    value = element.get("file")
                    if value and archived and element.get("file_type") == "archive":
                        path = (get_data_path() / value).resolve()
                        media_root = (get_data_path() / "session_media").resolve()
                        if path.is_relative_to(media_root):
                            direct.append({"kind": "trusted_path", "value": str(path)})
                elif kind == "reply":
                    quoted.extend(ComfyStudioPlugin._chain_images(element.get("chain"), archived=archived, depth=depth + 1))
                continue
            kind = getattr(getattr(element, "type", None), "value", "")
            if isinstance(element, Image) or (kind == "file" and str(getattr(element, "mime", "")).startswith("image/")):
                value = getattr(element, "file", None) or getattr(element, "image", None)
                file_type = getattr(element, "file_type", None)
                if value:
                    if file_type == "path":
                        direct.append({"kind": "trusted_path", "value": value})
                    elif file_type == "base64":
                        direct.append({"kind": "base64", "value": value})
                    else:
                        direct.append(value)
            elif kind == "reply" or element.__class__.__name__ == "Reply":
                quoted.extend(ComfyStudioPlugin._chain_images(getattr(element, "chain", None), depth=depth + 1))
        return direct + quoted

    @staticmethod
    def _event_owner(event) -> tuple[str, str]:
        message = getattr(event, "message", None)
        if message is None:
            messages = getattr(event, "messages", None) or []
            message = messages[-1] if messages else None
        user = getattr(getattr(message, "sender", None), "user_id", "")
        # KiraMessageEvent (im_message hook) has no .sid; only the batch event
        # does. Both carry a .session whose sid is the same string.
        sid = getattr(event, "sid", None) or getattr(
            getattr(event, "session", None), "sid", "")
        return str(sid or ""), str(user or "")

    def _remember_images(self, event) -> list:
        images = self._chain_images(getattr(getattr(event, "message", None), "chain", None))
        key = self._event_owner(event)
        if images and key[1]:
            _RECENT_INPUTS[key] = {"time": time.monotonic(), "images": images}
            while len(_RECENT_INPUTS) > 128:
                _RECENT_INPUTS.pop(next(iter(_RECENT_INPUTS)))
        return images

    async def _select_source(self, event, source_image=None, image_index=1):
        if source_image:
            return self._public_image_source(source_image)
        images = self._remember_images(event)
        sid, user = self._event_owner(event)
        history = getattr(self.ctx, "message_history", None)
        records = None
        quote_ids = {
            str(getattr(e, "message_id", ""))
            for e in (getattr(getattr(event, "message", None), "chain", None) or [])
            if getattr(getattr(e, "type", None), "value", "") == "reply"
        }
        # Some adapters only retain the quoted message id, without its chain.
        if not images and history is not None:
            try:
                records = (await history.list_messages(sid, limit=max(1, min(100, self._int("img2img_history_limit", 30))))).get("messages", [])
                if quote_ids:
                    for record in records:
                        if (record.get("session_id") == sid
                                and str(record.get("platform_message_id")) in quote_ids):
                            images = self._chain_images(record.get("chain"), archived=True)
                            if images:
                                break
                    if not images:
                        raise ImageInputError("引用的消息中没有可读取的图片，请重新附带原图")
            except ImageInputError:
                raise
            except Exception as exc:
                logger.debug("[comfyui_studio] image history lookup failed: %s", type(exc).__name__)
        if not images and quote_ids:
            raise ImageInputError("无法读取引用的图片，请重新附带原图")
        ttl = max(0, self._int("img2img_history_minutes", 10)) * 60
        if not images and user and ttl:
            cached = _RECENT_INPUTS.get((sid, user))
            if cached and time.monotonic() - cached["time"] <= ttl:
                images = cached["images"]
            elif cached:
                _RECENT_INPUTS.pop((sid, user), None)
            if not images and records is not None:
                now = time.time()
                for record in records:
                    if (record.get("session_id") == sid and record.get("direction") == "incoming"
                            and str(record.get("sender_id")) == user):
                        try:
                            stamp = float(record.get("timestamp", 0))
                        except (TypeError, ValueError):
                            continue
                        if stamp > 1e12:
                            stamp /= 1000
                        if not 0 <= now - stamp <= ttl:
                            continue
                        images = self._chain_images(record.get("chain"), archived=True)
                        if images:
                            break
        if not images:
            raise ImageInputError("没有找到原图，请在当前消息附带图片或引用含图片的消息，再说明希望怎样修改")
        try:
            index = int(image_index)
        except (TypeError, ValueError) as exc:
            raise ImageInputError("图片序号必须从 1 开始") from exc
        if not 1 <= index <= len(images):
            raise ImageInputError(f"当前有 {len(images)} 张可用图片，请选择 1~{len(images)} 的图片序号")
        return images[index - 1]

    @staticmethod
    def _public_image_source(value):
        if not isinstance(value, str) or not value.strip():
            raise ImageInputError("图片来源必须是图片链接或上传编号")
        value = value.strip()
        if (value.startswith(("http://", "https://", "data:image/", "input:", "upload:"))
                or re.fullmatch(r"[0-9a-f]{32}", value)):
            return value
        raise ImageInputError("图片来源仅接受 HTTP(S) 图片链接或上传编号，不接受本地路径")

    def _optional_image_sources(self, mask_image=None, reference_images=None):
        mask = self._public_image_source(mask_image) if mask_image else None
        if reference_images is not None and not isinstance(reference_images, list):
            raise ImageInputError("额外参考图必须是图片链接或编号列表")
        references = reference_images or []
        if len(references) > 2:
            raise ImageInputError("最多可添加两张额外参考图")
        return mask, [self._public_image_source(value) for value in references]

    def _img2img_options(self, denoise=None, resize_mode=None, seed=None) -> dict:
        if not self.plugin_cfg.get("img2img_enabled", True):
            raise ImageInputError("图生图功能当前已关闭")
        try:
            amount = float(self.plugin_cfg.get("img2img_denoise", 0.6) if denoise is None else denoise)
        except (TypeError, ValueError) as exc:
            raise ImageInputError("重绘幅度必须是 0~1 的数字") from exc
        if not math.isfinite(amount) or not 0 <= amount <= 1:
            raise ImageInputError("重绘幅度必须在 0~1 之间")
        mode = resize_mode or str(self.plugin_cfg.get("img2img_resize_mode") or "contain")
        if mode not in ("contain", "crop", "stretch"):
            raise ImageInputError("请选择留白适配、居中裁切或拉伸")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**48:
                raise ImageInputError("种子必须是 0~281474976710655 的整数")
        return {"denoise": amount, "resize_mode": mode, "seed": seed}

    def _select_model(self, model):
        if not model:
            return None
        preset, ambiguous = self._resolve_preset(str(model), self._get_presets())
        if not preset:
            reason = "匹配到多个预设：" + "、".join(ambiguous) if ambiguous else "未找到对应预设"
            raise ImageInputError(f"模型「{model}」{reason}，请使用正确的预设编号或完整名称")
        return preset

    # ------------------------------------------------------------------
    # generation pipeline
    # ------------------------------------------------------------------
    def _clamp_size(self, width: int, height: int, count: int) -> tuple[int, int, int]:
        """Clamp resolution to max side / max pixels; round to multiples of 8."""
        width = max(64, int(width or 1024))
        height = max(64, int(height or 1024))
        max_side = max(64, self._int("max_side", 1536))
        scale = min(1.0, max_side / max(width, height))
        width, height = max(64, int(width * scale)), max(64, int(height * scale))
        width = max(64, (width // 8) * 8)
        height = max(64, (height // 8) * 8)
        max_pixels = max(64 * 64, self._int("max_pixels", 1048576))
        if width * height > max_pixels:
            scale = (max_pixels / (width * height)) ** 0.5
            width = max(64, (int(width * scale) // 8) * 8)
            height = max(64, (int(height * scale) // 8) * 8)
            if width * height > max_pixels:
                width = max(64, min(width, (max_pixels // height // 8) * 8))
                height = max(64, min(height, (max_pixels // width // 8) * 8))
        count = max(1, min(self._int("max_batch", 4), int(count or 1)))
        return width, height, count

    def _build_graph(self, unet: str, clip: str, clip_type: str, vae: str,
                     weight_dtype: str, positive: str, negative: str,
                     width: int, height: int, batch: int, seed: int,
                     steps: int, cfg_scale: float, sampler: str,
                     scheduler: str, loras: Optional[list[dict]] = None,
                     input_image: Optional[str] = None, denoise: float = 1.0,
                     mask_image: Optional[str] = None, img2img_backend: str = "standard",
                     reference_images: Optional[list[str]] = None) -> dict:
        raw = WORKFLOW_TEMPLATE.read_text(encoding="utf-8")
        replacements = {
            "__UNET__": unet,
            "__CLIP__": clip,
            "__CLIP_TYPE__": clip_type,
            "__VAE__": vae,
            "__WEIGHT_DTYPE__": weight_dtype,
            "__POSITIVE__": positive,
            "__NEGATIVE__": negative,
            "__WIDTH__": width,
            "__HEIGHT__": height,
            "__BATCH__": batch,
            "__SEED__": seed,
            "__STEPS__": steps,
            "__CFG__": cfg_scale,
            "__SAMPLER__": sampler,
            "__SCHEDULER__": scheduler,
        }
        graph = json.loads(raw)
        for node in graph.values():
            for key, value in node.get("inputs", {}).items():
                if isinstance(value, str) and value in replacements:
                    node["inputs"][key] = replacements[value]
        if input_image:
            graph["10"] = {"class_type": "LoadImage", "inputs": {"image": input_image}}
            graph["6"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["3", 0]}}
            graph["11"] = {"class_type": "RepeatLatentBatch", "inputs": {"samples": ["6", 0], "amount": batch}}
            graph["7"]["inputs"].update(latent_image=["11", 0], denoise=denoise)
            if mask_image:
                graph["12"] = {"class_type": "LoadImage", "inputs": {"image": mask_image}}
                graph["13"] = {"class_type": "ImageToMask", "inputs": {"image": ["12", 0], "channel": "red"}}
                graph["6"] = {"class_type": "VAEEncodeForInpaint", "inputs": {
                    "pixels": ["10", 0], "vae": ["3", 0], "mask": ["13", 0], "grow_mask_by": 6,
                }}
                graph["14"] = {"class_type": "ImageCompositeMasked", "inputs": {
                    "destination": ["17", 0], "source": ["8", 0], "x": 0, "y": 0,
                    "resize_source": False, "mask": ["13", 0],
                }}
                graph["17"] = {"class_type": "RepeatImageBatch", "inputs": {"image": ["10", 0], "amount": batch}}
                graph["9"]["inputs"]["images"] = ["14", 0]
            if img2img_backend in ("qwen_edit", "qwen_edit_plus"):
                node_type = "TextEncodeQwenImageEditPlus" if img2img_backend == "qwen_edit_plus" else "TextEncodeQwenImageEdit"
                image_key = "image1" if img2img_backend == "qwen_edit_plus" else "image"
                reference_links = {}
                for index, reference in enumerate(reference_images or []):
                    node_id = str(15 + index)
                    graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": reference}}
                    reference_links[f"image{index + 2}"] = [node_id, 0]
                for node_id, prompt in (("4", positive), ("5", negative)):
                    graph[node_id] = {"class_type": node_type, "inputs": {
                        "clip": ["2", 0], "prompt": prompt, "vae": ["3", 0],
                        image_key: ["10", 0], **reference_links,
                    }}
            elif img2img_backend == "qwen_image_2_1":
                image_links = {"images.image_1": ["10", 0]}
                for index, reference in enumerate(reference_images or []):
                    node_id = str(15 + index)
                    graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": reference}}
                    image_links[f"images.image_{index + 2}"] = [node_id, 0]
                graph["4"] = {"class_type": "TextEncodeQwenImage21", "inputs": {
                    "clip": ["2", 0], "prompt": positive, "negative_prompt": negative,
                    "vae": ["3", 0], "resolution": 0, **image_links,
                }}
                del graph["5"]
                graph["7"]["inputs"]["negative"] = ["4", 1]
        self._apply_loras(graph, loras or [])
        return graph

    @staticmethod
    def _apply_loras(graph: dict, loras: list[dict]) -> None:
        """Chain LoraLoader nodes between the model/clip loaders and consumers."""
        if not loras:
            return
        prev_model, prev_clip = ["1", 0], ["2", 0]
        next_id = max((int(k) for k in graph if str(k).isdigit()), default=0) + 1
        for lora in loras:
            node_id = str(next_id)
            next_id += 1
            graph[node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora["name"],
                    "strength_model": lora["strength_model"],
                    "strength_clip": lora["strength_clip"],
                    "model": prev_model,
                    "clip": prev_clip,
                },
            }
            prev_model, prev_clip = [node_id, 0], [node_id, 1]
        for node in graph.values():
            inputs = node.get("inputs", {})
            if node.get("class_type") == "KSampler" and inputs.get("model") == ["1", 0]:
                inputs["model"] = prev_model
            elif node.get("class_type") in ("CLIPTextEncode", "TextEncodeQwenImageEdit", "TextEncodeQwenImageEditPlus", "TextEncodeQwenImage21") and inputs.get("clip") == ["2", 0]:
                inputs["clip"] = prev_clip

    @staticmethod
    def _clamp_strength(value) -> float:
        try:
            return max(-1.0, min(2.0, float(value)))
        except (TypeError, ValueError):
            return 1.0

    def _parse_default_loras(self) -> list[dict]:
        """Parse the default_loras textarea: one entry per line, ``name[:strength]``."""
        out: list[dict] = []
        raw = str(self.plugin_cfg.get("default_loras") or "")
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            strength = 1.0
            name = line
            if ":" in line:
                head, _, suffix = line.rpartition(":")
                try:
                    strength = max(-1.0, min(2.0, float(suffix.strip())))
                    name = head.strip()
                except ValueError:
                    name = line  # invalid strength suffix -> keep the whole line
            if name:
                out.append({
                    "name": name,
                    "strength_model": strength,
                    "strength_clip": strength,
                })
        return out

    def _resolve_loras(self, tool_loras, available: list[str]) -> list[dict]:
        """Merge default LoRAs with tool-requested ones, validated against the server.

        Tool entries win over defaults for the same name; missing files are skipped.
        """
        merged: dict[str, dict] = {}
        for entry in self._parse_default_loras():
            merged[entry["name"]] = entry
        for item in tool_loras or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            sm = self._clamp_strength(item.get("strength_model", 1.0))
            sc = self._clamp_strength(item.get("strength_clip", sm))
            merged[name] = {"name": name, "strength_model": sm, "strength_clip": sc}
        resolved = []
        for name, entry in merged.items():
            if name not in available:
                logger.warning("[comfyui_studio] skip missing lora: %s", name)
                continue
            resolved.append(entry)
        return resolved

    async def _get_lora_names(self) -> list[str]:
        """Fetch (cached) LoRA file names from the server for prompt injection."""
        now = time.time()
        cached_ts, cached = self._lora_cache
        if now - cached_ts < _LORA_CACHE_TTL:
            return cached
        try:
            client = ComfyClient(
                str(self.plugin_cfg.get("comfy_url") or ""),
                timeout=15.0,
            )
            names = await client.list_loras()
            self._lora_cache = (now, names)
            return names
        except Exception as exc:  # noqa: BLE001 - keep the old list on failure
            logger.debug("[comfyui_studio] lora list fetch failed: %s", exc)
            return cached

    def _positive_prompt(self, prompt: str, preset: Optional[dict] = None) -> str:
        """Quality words (preset override, else global) + the model's content."""
        quality = str((preset or {}).get("positive_quality") or "").strip()
        if not quality:
            quality = str(self.plugin_cfg.get("positive_quality") or DEFAULT_POSITIVE_QUALITY)
        quality = quality.strip().rstrip(",")
        content = prompt.strip()
        return f"{quality}, {content}" if quality else content

    def _negative_prompt(self, preset: Optional[dict] = None) -> str:
        quality = str((preset or {}).get("negative_quality") or "").strip()
        if not quality:
            quality = str(self.plugin_cfg.get("negative_quality") or DEFAULT_NEGATIVE_QUALITY)
        return quality

    def _preset_cfg(self, preset: Optional[dict]) -> float:
        """Per-preset CFG; empty/0 inherits the global value."""
        raw = (preset or {}).get("cfg")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return self._float("cfg", 4.0)
        return value if value > 0 else self._float("cfg", 4.0)

    def _image_dir(self) -> Path:
        path = self._data_dir / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def _upload_one(self, path: Path) -> tuple[Optional[str], str]:
        """Upload one local image to the bed; returns (url, error)."""
        return await upload_image(
            base_url=str(self.plugin_cfg.get("bed_base_url") or ""),
            upload_path=str(self.plugin_cfg.get("bed_upload_path") or "/upload"),
            file_field=str(self.plugin_cfg.get("bed_file_field") or "files"),
            token=str(self.plugin_cfg.get("bed_token") or ""),
            file_bytes=path.read_bytes(),
            filename=path.name,
            timeout=self._float("request_timeout", 120.0),
        )

    @staticmethod
    def _md_lines(urls: list[str], width: int, height: int) -> list[str]:
        return [f"![img #{width}px #{height}px]({url})" for url in urls]

    def _buttons_enabled(self) -> bool:
        return (self.plugin_cfg.get("buttons_enabled", True)
                and self._send_mode == "markdown")

    @property
    def _commands_enabled(self) -> bool:
        # A new opt-in key also keeps upgrades from inheriting the old default-on flag.
        return self.plugin_cfg.get("command_features_enabled", False) is True

    def _is_plugin_command(self, text: str) -> bool:
        prefixes = [str(self.plugin_cfg.get(key) or default).strip() for key, default in (
            ("command_prefix", "/comfy"), ("img2img_command_prefix", "/img2img"),
            ("upscale_command_prefix", "/upscale"), ("model_command_prefix", "/model"),
            ("lora_command_prefix", "/lora"))]
        prefixes.append("/__comfy_settings")
        settings = str(self.plugin_cfg.get("settings_command_prefix") or "设置").strip()
        return text == settings or any(
            text == prefix or (text.startswith(prefix) and text[len(prefix):len(prefix) + 1].isspace())
            for prefix in prefixes if prefix)

    def _buttons_payload(self, job: dict) -> Optional[dict]:
        """Button bar under a delivered image: again / upscale / source links."""
        comfy_prefix = str(self.plugin_cfg.get("command_prefix") or "/comfy").strip()
        upscale_prefix = str(self.plugin_cfg.get("upscale_command_prefix") or "/upscale").strip()
        scale = min(2.0, max(1.1, self._float("upscale_scale", 1.5)))
        # permission.type=2 (everyone) MUST be explicit: the platform's default
        # is not "everyone" and members otherwise get a "no permission" toast.
        everyone = {"type": 2}
        # Action buttons are CALLBACK buttons (type=1): command buttons (type=2)
        # only fill the input box in groups, while callbacks fire immediately —
        # the bridge turns the click into "[按钮] 用户点击了：<data>" which
        # handle_command recognises and executes directly.
        row1 = [
            {"id": "b_again", "render_data": {"label": "再来一张", "style": 1},
             "action": {"type": 1, "data": f"{comfy_prefix} again", "permission": everyone}},
            {"id": "b_upscale", "render_data": {"label": f"{scale:g}倍放大", "style": 1},
             "action": {"type": 1, "data": upscale_prefix, "permission": everyone}},
        ]
        rows = [{"buttons": row1}] if self._commands_enabled else []
        urls = job.get("urls") or []
        if urls:
            row2 = []
            for index, url in enumerate(urls[:4]):
                label = "原图直链" if len(urls) == 1 else f"原图{index + 1}"
                row2.append({"id": f"b_link{index}", "render_data": {"label": label, "style": 0},
                             "action": {"type": 0, "data": url, "permission": everyone}})
            rows.append({"buttons": row2})
        return {"content": {"rows": rows}} if rows else None

    def _keyboard_marker(self, payload: dict):
        """Wrap a keyboard payload in the bridge's marker element (if loaded)."""
        import sys
        for mod in list(sys.modules.values()):
            file_path = str(getattr(mod, "__file__", "") or "").replace("\\", "/")
            if "qqbot-fullmsg-bridge" in file_path and file_path.endswith("rich_content.py"):
                try:
                    validated = mod.validate_keyboard(json.dumps(payload))
                    return mod.KeyboardMarker(validated)
                except Exception as exc:  # noqa: BLE001 - buttons are best-effort
                    logger.debug("[comfyui_studio] keyboard payload rejected: %s", exc)
                    return None
        return None

    async def _deliver_job(self, job: dict):
        """Directly send a finished job's images to its session."""
        if job["delivered"]:
            return
        async with self._deliver_lock:
            await self._deliver_job_locked(job)

    async def _deliver_job_locked(self, job: dict):
        if job["delivered"]:
            return
        try:
            if job["status"] == "failed":
                await self.ctx.send_message_chain(
                    job["session"],
                    MessageChain([Text(f"绘图失败了：{job['error']}")]),
                )
                job["delivered"] = True
                return
            if (job.get("source") != "llm" and self._send_mode == "markdown" and job["urls"]
                    and len(job["urls"]) == len(job["files"])):
                # Reference the LOCAL files: the bridge re-hosts them to QQ's
                # own COS via the official chunk upload, so rendering never
                # depends on the image bed being reachable. The bed copy stays
                # as the archive behind the 原图直链 button.
                local = [str(Path(p).resolve()) for p in job["files"]]
                md = "\n".join(self._md_lines(local, job["width"], job["height"]))
                chain = [Text(md)]
                if self.plugin_cfg.get("buttons_enabled", True):
                    payload = self._buttons_payload(job)
                    marker = self._keyboard_marker(payload) if payload else None
                    if marker is not None:
                        chain.append(marker)
                await self.ctx.send_message_chain(
                    job["session"], MessageChain(chain)
                )
            else:
                sent_files = job.setdefault("_delivered_files", [])
                for file_path in job["files"]:
                    if file_path in sent_files:
                        continue
                    await self.ctx.send_message_chain(
                        job["session"],
                        MessageChain([Image(image=str(file_path))]),
                    )
                    sent_files.append(file_path)
            job["delivered"] = True
            job.pop("delivery_error", None)
        except Exception as exc:  # noqa: BLE001 - delivery must never crash the job
            job["delivery_error"] = str(exc)
            logger.error("[comfyui_studio] direct send failed: %s", exc)

    async def _prepare_img2img_job(self, job: dict, client: ComfyClient, unet: str):
        store = self._input_store()
        data = await store.read(job["_source_image"])
        image = await asyncio.to_thread(store.decode, data)
        input_info = await store.save(data)
        job.update(input_id=input_info["id"], input_width=image.width, input_height=image.height)
        width, height = int(job["width"] or 0), int(job["height"] or 0)
        if width <= 0 and height <= 0:
            width, height = image.size
        elif width <= 0:
            width = round(height * image.width / image.height)
        elif height <= 0:
            height = round(width * image.height / image.width)
        width, height, count = self._clamp_size(width, height, job["count"])
        preset = job.get("preset") or {}
        backend = preset.get("img2img_backend") or self.plugin_cfg.get("img2img_backend") or "auto"
        if backend == "auto":
            lowered = unet.lower()
            if "qwen" in lowered and ("2.1" in lowered or "2_1" in lowered):
                backend = "qwen_image_2_1"
            elif "qwen" in lowered and "edit" in lowered:
                backend = "qwen_edit_plus" if any(s in lowered for s in ("plus", "2509", "2511")) else "qwen_edit"
            else:
                backend = "standard"
        if backend not in ("standard", "qwen_edit", "qwen_edit_plus", "qwen_image_2_1"):
            raise ImageInputError("图生图工作流类型无效，请检查模型预设")
        if backend == "qwen_image_2_1":
            width, height = max(64, width // 32 * 32), max(64, height // 32 * 32)
        references = job.get("_reference_images") or []
        if references and backend not in ("qwen_edit_plus", "qwen_image_2_1"):
            raise ImageInputError("额外参考图需要 Qwen Edit Plus 或 Qwen Image 2.1 模型预设")
        mask_data = await store.read(job["_mask_image"]) if job.get("_mask_image") else None
        prepared, mask = await asyncio.to_thread(store.prepare, data, width, height,
                                                job["resize_mode"], mask_data)
        if mask is not None:
            job["mask_id"] = (await store.save(mask_data))["id"]
        job.update(width=width, height=height, count=count, img2img_backend=backend)
        if job["denoise"] == 0:
            return {}, prepared
        required = {"LoadImage", "VAEEncode", "RepeatLatentBatch"}
        if mask is not None:
            required.update({"VAEEncodeForInpaint", "ImageToMask", "ImageCompositeMasked", "RepeatImageBatch"})
        required.update({"qwen_edit": {"TextEncodeQwenImageEdit"},
                         "qwen_edit_plus": {"TextEncodeQwenImageEditPlus"},
                         "qwen_image_2_1": {"TextEncodeQwenImage21"}}.get(backend, set()))
        info = await client.object_info()
        missing = sorted(required.difference(info))
        if missing:
            raise ImageInputError("ComfyUI 缺少图生图节点：" + "、".join(missing))
        uploaded = await client.upload_input_image(prepared, f"{job['id']}_source.png")
        mask_name = await client.upload_input_image(mask, f"{job['id']}_mask.png") if mask else None
        reference_names = []
        for index, source in enumerate(references):
            raw = await store.read(source)
            reference = await asyncio.to_thread(store.decode, raw)
            rw, rh, _ = self._clamp_size(reference.width, reference.height, 1)
            normalized, _ = await asyncio.to_thread(store.prepare, raw, rw, rh, "contain")
            reference_names.append(await client.upload_input_image(normalized, f"{job['id']}_ref{index + 1}.png"))
        return {"input_image": uploaded, "denoise": job["denoise"], "mask_image": mask_name,
                "img2img_backend": backend, "reference_images": reference_names}, prepared

    async def _execute_job(self, job: dict):
        """Run one generation job end-to-end and deliver if requested."""
        client = ComfyClient(
            str(self.plugin_cfg.get("comfy_url") or ""),
            timeout=self._float("request_timeout", 120.0),
        )
        try:
            job["status"] = "preparing"
            preset = job.get("preset") or {}
            unet = await client.resolve_unet(
                str(preset.get("unet") or self.plugin_cfg.get("unet_name") or "")
            )
            job["unet"] = unet
            available_loras = await client.list_loras()
            if job.get("replace_default_loras"):
                missing = [x["name"] for x in job.get("tool_loras") or [] if x["name"] not in available_loras]
                if missing:
                    raise ComfyClientError("个人配置中的 LoRA 已不可用，请重新设置：" + "、".join(missing))
            loras = self._resolve_loras(job.get("tool_loras"), available_loras)
            if loras:
                job["loras"] = [
                    f"{l['name']}@{l['strength_model']:g}" for l in loras
                ]
            img2img_args, prepared = {}, None
            if job.get("mode") == "img2img":
                img2img_args, prepared = await self._prepare_img2img_job(job, client, unet)
            if job.get("source") == "command":
                await self._notify_generation(None, job)
            seed = job.get("seed")
            if seed is None:
                seed = random.randrange(2**48)
            job["seed"] = seed
            graph = self._build_graph(
                unet=unet,
                clip=str(preset.get("clip") or self.plugin_cfg.get("clip_name") or ""),
                clip_type=str(preset.get("clip_type") or self.plugin_cfg.get("clip_type") or "stable_diffusion"),
                vae=str(preset.get("vae") or self.plugin_cfg.get("vae_name") or ""),
                weight_dtype=str(preset.get("weight_dtype") or self.plugin_cfg.get("weight_dtype") or "default"),
                positive=self._positive_prompt(job["prompt"], preset),
                negative=self._negative_prompt(preset),
                width=job["width"],
                height=job["height"],
                batch=job["count"],
                seed=seed,
                steps=int(preset.get("steps") or 0) or self._int("steps", 30),
                cfg_scale=self._preset_cfg(preset),
                sampler=(str(preset.get("sampler") or "").strip()
                         or str(self.plugin_cfg.get("sampler") or "euler")),
                scheduler=(str(preset.get("scheduler") or "").strip()
                           or str(self.plugin_cfg.get("scheduler") or "simple")),
                loras=loras,
                **img2img_args,
            )
            job["status"] = "running"
            if prepared is not None and job["denoise"] == 0:
                output_data = [prepared] * job["count"]
            else:
                prompt_id = await client.submit(graph, client_id=uuid.uuid4().hex)
                job["prompt_id"] = prompt_id
                result = await client.wait_result(
                    prompt_id, timeout=self._int("job_timeout_seconds", 300)
                )
                if not result.ok:
                    raise ComfyClientError(result.error)
                output_data = [await client.download_image(ref) for ref in result.images]

            out_dir = self._image_dir()
            paths: list[Path] = []
            for data in output_data:
                path = out_dir / f"{_now_str()}_{uuid.uuid4().hex[:8]}.png"
                path.write_bytes(data)
                paths.append(path)
            if not paths:
                raise ComfyClientError("ComfyUI 未返回任何输出图片")
            job["files"] = [str(p) for p in paths]

            if job.get("source") != "llm" and self.plugin_cfg.get("bed_enabled", True):
                urls: list[str] = []
                bed_error = ""
                for path in paths:
                    url, error = await self._upload_one(path)
                    if url:
                        urls.append(url)
                    elif not bed_error:
                        bed_error = error
                job["urls"] = urls
                if len(urls) != len(paths):
                    logger.warning(
                        "[comfyui_studio] bed upload failed, fallback to media: %s",
                        bed_error,
                    )
                    job["error"] = f"图床上传失败：{bed_error}"

            job["status"] = "done"
            logger.info(
                "[comfyui_studio] job %s done: %s image(s), %s uploaded",
                job["id"], len(job["files"]), len(job["urls"]),
            )
        except asyncio.CancelledError:
            job["status"] = "failed"
            job["error"] = "任务已取消"
            raise
        except Exception as exc:  # noqa: BLE001 - report any failure to the user
            job["status"] = "failed"
            job["error"] = str(exc)
            logger.error("[comfyui_studio] job %s failed: %s", job["id"], exc)
        finally:
            job["done"].set()
            if job["deliver_directly"]:
                await self._deliver_job(job)

    def _start_job(self, source: str, session: str, prompt: str, width: int,
                   height: int, count: int, deliver_directly: bool,
                   tool_loras: Optional[list] = None,
                   preset: Optional[dict] = None, *, source_image=None,
                   denoise=None, resize_mode=None, seed=None, mask_image=None,
                   reference_images=None, replace_default_loras=False) -> dict:
        options = self._img2img_options(denoise, resize_mode, seed) if source_image is not None else {}
        job = _new_job(source, session, prompt, width, height, count)
        job["deliver_directly"] = deliver_directly
        job["tool_loras"] = tool_loras
        job["preset"] = preset
        job["mode"] = "img2img" if source_image is not None else "txt2img"
        job["seed"] = seed
        job["replace_default_loras"] = replace_default_loras
        if source_image is not None:
            job.update(options)
            job["_source_image"] = source_image
            job["_mask_image"] = mask_image
            job["_reference_images"] = reference_images or []
        # Each job uses the settings that were active when it was submitted.
        worker = copy.copy(self)
        worker.plugin_cfg = copy.deepcopy(self.plugin_cfg)
        if replace_default_loras:
            worker.plugin_cfg["default_loras"] = ""
        task = asyncio.create_task(worker._execute_job(job))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    # ------------------------------------------------------------------
    # Dynamic tool description: append the current LoRA list so the model
    # can pick valid names without a trial-and-error round trip.
    # ------------------------------------------------------------------
    @on.llm_request()
    async def inject_lora_hint(self, event, request, tag_set=None):
        if not self.plugin_cfg.get("enabled", True):
            return
        current_images = self._remember_images(event)
        tool_set = getattr(request, "tool_set", None)
        if tool_set is None:
            return
        suffixes = []
        names = await self._get_lora_names()
        if names:
            suffixes.append(
                "当前可用的 LoRA 文件名（loras.name 必须精确使用其中之一，"
                "仅在用户明确要求对应画风/角色/效果时才传）：" + "；".join(names)
            )
        presets = self._get_presets()
        if presets:
            suffixes.append(
                "当前可用的模型预设（model 参数可填编号或名称，每套含配套编码器/VAE/"
                "采样参数/质量词）：" + "；".join(
                    f"#{i}. {p['name']}" for i, p in enumerate(presets, 1)
                )
            )
        if current_images:
            suffixes.append(f"当前消息有 {len(current_images)} 张可用图片；修改图片请调用 comfyui_img2img，image_index 从 1 开始。")
        suffix = "\n".join(suffixes)
        for tool in tool_set.tools:
            if getattr(tool, "name", "") == "comfyui_generate_image":
                tool.description = TOOL_DESCRIPTION + "\n" + suffix
            elif getattr(tool, "name", "") == "comfyui_img2img":
                tool.description = IMG2IMG_DESCRIPTION + "\n" + suffix

    # ------------------------------------------------------------------
    # LLM tool: the model decides prompt / resolution / count
    # ------------------------------------------------------------------
    @register.tool(
        name="comfyui_generate_image",
        description=TOOL_DESCRIPTION,
        params={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "画面内容描述，不含质量词；语言和写法跟随所选模型",
                },
                "width": {
                    "type": "integer",
                    "description": "图片宽度（像素，8 的倍数）",
                },
                "height": {
                    "type": "integer",
                    "description": "图片高度（像素，8 的倍数）",
                },
                "count": {
                    "type": "integer",
                    "description": "生成张数，默认 1",
                },
                "model": {
                    "type": "string",
                    "description": (
                        "可选的模型预设编号或名称（见系统提示的可用列表）。"
                        "根据画面风格选最合适的：动漫/插画类选动漫系预设，"
                        "写实/摄影/场景/带文字类选通用大模型预设；不传则用默认模型"
                    ),
                },
                "loras": {
                    "type": "array",
                    "description": (
                        "可选叠加的 LoRA 列表。仅当用户明确要求某种画风/角色/效果"
                        "且系统提供了对应 LoRA 时使用；name 必须精确使用系统提示的"
                        "可用文件名"
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "LoRA 文件名（用系统提供的可用列表里的名字）",
                            },
                            "strength_model": {
                                "type": "number",
                                "description": "模型强度，建议 0.4-1.2，默认 1",
                            },
                            "strength_clip": {
                                "type": "number",
                                "description": "文本编码强度，默认同 strength_model",
                            },
                        },
                        "required": ["name"],
                    },
                },
            },
            "required": ["prompt", "width", "height"],
        },
    )
    async def tool_generate_image(self, event, prompt: str, width: int = 1024,
                                  height: int = 1024, count: int = 1,
                                  loras: Optional[list] = None,
                                  model: Optional[str] = None):
        if not self.plugin_cfg.get("enabled", True):
            return "绘图功能当前已关闭。"

        preset = None
        if model:
            presets = self._get_presets()
            preset, ambiguous = self._resolve_preset(model, presets)
            if not preset:
                if ambiguous:
                    return (
                        f"模型「{model}」匹配到多个预设：{'、'.join(ambiguous)}。"
                        "请改用编号或完整名称重新调用。"
                    )
                listing = "\n".join(
                    f"#{i}. {p['name']}" for i, p in enumerate(presets, 1)
                ) or "（尚未配置任何模型预设）"
                return f"没有找到模型预设「{model}」。可用预设：\n{listing}"

        width, height, count = self._clamp_size(width, height, count)
        # AI calls always deliver local image files through the adapter media path.
        job = self._start_job(
            source="llm",
            session=event.sid,
            prompt=prompt,
            width=width,
            height=height,
            count=count,
            deliver_directly=True,
            tool_loras=loras,
            preset=preset,
        )

        return await self._wait_for_job(job)

    @register.tool(name="comfyui_img2img", description=IMG2IMG_DESCRIPTION, params=IMG2IMG_PARAMS)
    async def tool_img2img(self, event, prompt: str, source_image: Optional[str] = None,
                          image_index: int = 1, denoise: Optional[float] = None,
                          width: int = 0, height: int = 0, count: int = 1,
                          model: Optional[str] = None, loras: Optional[list] = None,
                          resize_mode: Optional[str] = None, seed: Optional[int] = None,
                          mask_image: Optional[str] = None,
                          reference_images: Optional[list] = None,
                          mask_index: Optional[int] = None,
                          reference_indices: Optional[list] = None):
        if not self.plugin_cfg.get("enabled", True):
            return "绘图功能当前已关闭。"
        try:
            if not str(prompt or "").strip():
                raise ImageInputError("请说明希望怎样修改图片")
            options = self._img2img_options(denoise, resize_mode, seed)
            preset = self._select_model(model)
            source = await self._select_source(event, source_image, image_index)
            mask, references = self._optional_image_sources(mask_image, reference_images)
            if mask_index is not None:
                if mask is not None:
                    raise ImageInputError("蒙版请只指定链接或图片序号中的一种")
                mask = await self._select_source(event, image_index=mask_index)
            if reference_indices is not None and not isinstance(reference_indices, list):
                raise ImageInputError("参考图序号必须是列表")
            if len(references) + len(reference_indices or []) > 2:
                raise ImageInputError("最多可添加两张额外参考图")
            for index in reference_indices or []:
                references.append(await self._select_source(event, image_index=index))
            width, height = self._requested_image_size(width, height)
            count = max(1, min(self._int("max_batch", 4), int(count or 1)))
            job = self._start_job("llm", event.sid, prompt.strip(), width, height, count,
                                  True, tool_loras=loras, preset=preset, source_image=source,
                                  mask_image=mask, reference_images=references, **options)
        except (ImageInputError, TypeError, ValueError) as exc:
            return f"图生图未开始：{exc}"
        return await self._wait_for_job(job)

    @staticmethod
    def _requested_image_size(width, height):
        try:
            width, height = int(width or 0), int(height or 0)
        except (TypeError, ValueError) as exc:
            raise ImageInputError("宽高必须是非负整数，0 表示跟随原图") from exc
        if width < 0 or height < 0:
            raise ImageInputError("宽高不能为负数，0 表示跟随原图")
        return width, height

    async def _wait_for_job(self, job):
        sync_wait = max(5.0, min(55.0, self._int("sync_wait_seconds", 45)))
        try:
            await asyncio.wait_for(job["done"].wait(), timeout=sync_wait)
        except asyncio.TimeoutError:
            if not job["done"].is_set():
                # Hand delivery over to the background task; tell the LLM to wait.
                job["deliver_directly"] = True
                return (
                    "任务已提交，完成后会自动发送图片文件。"
                    "不要发送进度通知、复述生成参数或重复调用本工具。"
                )
        except asyncio.CancelledError:
            job["deliver_directly"] = True
            if job["done"].is_set():
                await asyncio.shield(self._deliver_job(job))
            raise

        # Completion is signalled before background delivery finishes. Await the
        # same locked delivery before telling the LLM whether to send an image.
        if job.get("deliver_directly"):
            await self._deliver_job(job)
        return self._tool_result_for_done_job(job)

    def _tool_result_for_done_job(self, job: dict):
        """Build the tool return value for a finished job."""
        if job["status"] == "failed":
            return f"绘图失败：{job['error']}。请告知用户这次没有画出来。"

        if job.get("source") == "llm":
            if job.get("delivered"):
                return "图片文件已直接发到当前会话。不要重复发图、发链接、发送 Markdown、进度通知或复述参数。"
            return ("图片已生成，但图片文件发送未完成：" + str(job.get("delivery_error") or "发送失败")
                    + "。请告知用户发送失败，不要声称已发送，也不要自动重新生成。")

        width, height, count = job["width"], job["height"], len(job["files"])
        preset_name = (job.get("preset") or {}).get("name") or ""
        preset_note = f"（模型: {preset_name}）" if preset_name else ""
        lora_note = ""
        if job.get("loras"):
            lora_note = "（LoRA: " + ", ".join(job["loras"]) + "）"
        mode_note = f"图生图，重绘幅度 {job['denoise']:g}，" if job.get("mode") == "img2img" else ""
        summary = f"图片已生成（{mode_note}{width}x{height}，共 {count} 张）{preset_note}{lora_note}。"

        if job.get("delivered"):
            # Buttons-enabled delivery: the plugin already sent the image with
            # its button bar; the LLM only adds a short comment.
            return (summary + " 图片已直接发到当前会话（带快捷按钮）。"
                    "请只配一句简短自然的评论，不要重复发图、发链接或复述参数。")

        if (self._send_mode == "markdown" and job["urls"]
                and len(job["urls"]) == len(job["files"])):
            # Local files again: the bridge re-hosts them to QQ COS reliably;
            # bed URLs stay behind the 原图直链 button.
            md_block = "\n".join(
                self._md_lines([str(Path(p).resolve()) for p in job["files"]], width, height)
            )
            return (
                f"{summary}\n{md_block}\n"
                "请立刻用 <markdown> 标签把上面的图片行原样发给用户"
                "（保持 ![img #宽px #高px](链接) 的格式一字不改），"
                "并配一句简短自然的话。"
            )

        # Media mode (or bed upload failed): hand local files to the core
        # <file> delivery flow.
        from core.agent.tool import ToolResult

        return ToolResult(
            text=summary + " 请用 <file> 标签把这些图片发给用户。",
            attachments=[Image(image=p) for p in job["files"]],
        )

    # ------------------------------------------------------------------
    # /comfy direct command (bypasses the LLM)
    # ------------------------------------------------------------------
    async def _reply(self, event: KiraMessageEvent, text: str):
        await self.ctx.send_message_chain(
            event.session.sid, MessageChain([Text(text)])
        )

    def _make_settings_client(self):
        return ComfyClient(str(self.plugin_cfg.get("comfy_url") or ""), timeout=self._PROBE_TIMEOUT)

    async def _send_settings_page(self, event, text, payload, choices, draft):
        marker = self._keyboard_marker(payload)
        if marker is None:
            lines = []
            for label, action, value in choices:
                lines.append(f"{label}：/__comfy_settings {draft['token']} {draft['revision']} {action} {value}")
            text += "\n按钮不可用，可复制对应选项的指令：\n" + "\n".join(lines)
        chain = [Text("# 生图设置\n" + text)]
        if marker is not None:
            chain.append(marker)
        await self.ctx.send_message_chain(self._event_owner(event)[0], MessageChain(chain))

    @register.tool(name="comfyui_user_settings",
                   description="打开当前用户的生图设置按钮向导，或查看已保存配置。用户说设置绘图参数时调用；选项由用户依次点击，保存后 /comfy 提示词 自动使用。",
                   params={"type": "object", "properties": {"action": {"type": "string", "enum": ["start", "show"]}}})
    async def tool_user_settings(self, event, action="start"):
        if not self.plugin_cfg.get("enabled", True):
            return "绘图功能当前已关闭。"
        try:
            if action == "show":
                profile = self._user_profile(event)
                return self._profile_summary(profile) if profile else "当前用户尚未保存配置，请发送设置指令。"
            if action != "start":
                return "仅支持 start 或 show。"
            if not self._commands_enabled:
                return "指令式功能已关闭，无法打开按钮设置向导。管理员可在插件配置中开启「启用所有指令式功能」。"
            await self._start_settings(event)
            return "已发送第一步底模选择，请让用户点击设置按钮继续；不要代替用户选择或开始生成。"
        except (SettingsError, OSError, ValueError) as exc:
            return f"设置未开始：{exc}"

    async def _notify_generation(self, event, job):
        preset = job.get("preset") or {}
        model = preset.get("name") or job.get("unet") or str(self.plugin_cfg.get("unet_name") or "自动选择可用底模")
        dimensions = f"{job['width']} × {job['height']}" if job["width"] and job["height"] else "跟随原图（受尺寸上限限制）"
        lora_note = "、".join(job.get("loras") or []) or "不使用"
        mode = "图生图" if job.get("mode") == "img2img" else "文生图"
        await self.ctx.send_message_chain(job["session"], MessageChain([Text(
            f"正在生成 · {mode}\n底模：{model}\n分辨率：{dimensions}\n张数：{job['count']}\nLoRA：{lora_note}\n完成后会发到当前会话。")]))

    @on.im_message(priority=Priority.HIGH)
    async def handle_command(self, event: KiraMessageEvent):
        if not self.plugin_cfg.get("enabled", True):
            return

        message = event.message
        text = "".join(
            getattr(ele, "text", "") for ele in (message.chain or [])
            if isinstance(ele, Text)
        ).strip()
        # Bridge button clicks arrive as synthetic notice events whose text
        # embeds the button data after "点击了：" and may also carry the
        # clicker's nickname ("[按钮] 萧洋 点击了：/comfy again"). They must run
        # even though is_notice=True; stop() then keeps the LLM out of it.
        click_match = re.match(r"^\[按钮\].*?点击了：\s*(.+)$", text)
        clicked = click_match is not None
        if clicked:
            text = click_match.group(1).strip()
        if event.is_notice and not clicked:
            return

        sender_id = getattr(message.sender, "user_id", "")
        if sender_id and sender_id == getattr(message, "self_id", ""):
            return  # ignore the bot's own messages
        if self.plugin_cfg.get("img2img_enabled", True):
            self._remember_images(event)
        # Consume known disabled commands before any callback or LLM can act on them.
        if not self._commands_enabled:
            if self._is_plugin_command(text):
                event.stop()
                await self._reply(event, "指令式功能已关闭，请先在插件配置中开启「启用所有指令式功能」。")
            return
        if text.startswith("/__comfy_settings "):
            event.stop()
            try:
                await self._settings_action(event, text)
            except (SettingsError, ValueError, IndexError, OSError, ComfyClientError) as exc:
                await self._reply(event, f"设置未完成：{exc}。可重新发送设置指令继续。")
            return
        prefix = str(self.plugin_cfg.get("command_prefix") or "/comfy").strip()
        lora_prefix = str(self.plugin_cfg.get("lora_command_prefix") or "/lora").strip()
        model_prefix = str(self.plugin_cfg.get("model_command_prefix") or "/model").strip()
        img_prefix = str(self.plugin_cfg.get("img2img_command_prefix") or "/img2img").strip()
        upscale_prefix = str(self.plugin_cfg.get("upscale_command_prefix") or "/upscale").strip()

        def matches(command):
            return text == command or (text.startswith(command)
                                       and text[len(command):len(command) + 1].isspace())

        settings_prefix = str(self.plugin_cfg.get("settings_command_prefix") or "设置").strip()
        if text in (settings_prefix, f"{prefix} 设置", f"{prefix} setup", f"{prefix} settings"):
            event.stop()
            try:
                await self._start_settings(event)
            except (SettingsError, OSError, ValueError) as exc:
                await self._reply(event, f"设置未开始：{exc}")
            return

        if matches(upscale_prefix):
            await self._handle_upscale_command(event, upscale_prefix)
            event.stop()
            return

        if matches(img_prefix):
            await self._handle_img2img_command(event, text[len(img_prefix):].strip())
            event.stop()
            return

        if matches(lora_prefix):
            await self._handle_lora_command(event, text[len(lora_prefix):].strip(),
                                            prefix, lora_prefix)
            event.stop()
            return

        if matches(model_prefix):
            await self._handle_model_command(event, prefix, lora_prefix, model_prefix)
            event.stop()
            return

        if not matches(prefix):
            return

        rest = text[len(prefix):].strip()
        if rest.startswith("--img2img ") or rest.startswith("img2img "):
            await self._handle_img2img_command(event, rest.split(" ", 1)[1].strip())
            event.stop()
            return
        force_text = rest.startswith("--txt2img ")
        if force_text:
            rest = rest[len("--txt2img "):].strip()
        if rest.lower() == "again":
            await self._handle_again_command(event, prefix, lora_prefix, model_prefix,
                                             upscale_prefix)
            event.stop()
            return
        if rest.lower() in ("", "help", "用法", "帮助"):
            presets = self._get_presets()
            preset_lines = ""
            if presets:
                preset_lines = "\n可用模型预设：\n" + "\n".join(
                    f"#{i}. {p['name']} → {p['unet']}"
                    for i, p in enumerate(presets, 1)
                )
            await self._reply(event, self._usage_text(prefix, lora_prefix, model_prefix) + preset_lines)
            event.stop()
            return

        match = _CMD_RE.match(rest)
        try:
            profile = self._user_profile(event)
        except (SettingsError, ValueError, OSError) as exc:
            await self._reply(event, f"读取个人配置失败：{exc}")
            event.stop()
            return
        if not match:
            # A bare prompt uses the user's saved profile. Explicit legacy
            # dimensions and model/LoRA arguments still override their fields.
            try:
                preset = self._profile_model(profile)
            except SettingsError as exc:
                await self._reply(event, str(exc))
                event.stop()
                return
            defaults = profile or {"width": 1024, "height": 1024, "count": 1, "loras": None}
            dimensions = f"{defaults['width']}x{defaults['height']} {defaults['count']} "
            match = _CMD_RE.match(dimensions + rest)
        if not match:
            await self._reply(event, "指令格式不对哦。\n" + self._usage_text(prefix, lora_prefix, model_prefix))
            event.stop()
            return

        width = int(match.group("width"))
        height = int(match.group("height"))
        count = int(match.group("count") or (profile or {}).get("count", 1))
        prompt = match.group("prompt").strip()
        width, height, count = self._clamp_size(width, height, count)
        if not prompt:
            await self._reply(event, "提示词不能为空。\n" + self._usage_text(prefix, lora_prefix, model_prefix))
            event.stop()
            return

        try:
            preset = self._profile_model(profile)
        except SettingsError as exc:
            if not match.group("model"):
                await self._reply(event, str(exc)); event.stop(); return
            preset = None
        model_token = match.group("model")
        if model_token:
            presets = self._get_presets()
            if not presets:
                await self._reply(
                    event,
                    f"没有配置模型预设（#{model_token} 无法识别）。"
                    "请在 WebUI 侧边栏「ComfyUI 绘图 → 模型预设」里添加，"
                    "或省略 #参数使用默认模型。",
                )
                event.stop()
                return
            preset, ambiguous = self._resolve_preset(model_token, presets)
            if not preset:
                listing = "\n".join(
                    f"#{i}. {p['name']}" for i, p in enumerate(presets, 1)
                )
                hint = f"「{model_token}」匹配到多个：{'、'.join(ambiguous)}" if ambiguous \
                    else f"「{model_token}」没有匹配到模型预设"
                await self._reply(event, hint + "\n可用预设：\n" + listing)
                event.stop()
                return

        tool_loras = copy.deepcopy(profile.get("loras")) if profile else None
        replace_defaults = profile is not None and profile.get("loras") is not None
        lora_segment = match.group("loras")
        if lora_segment:
            try:
                client = ComfyClient(
                    str(self.plugin_cfg.get("comfy_url") or ""),
                    timeout=self._PROBE_TIMEOUT,
                )
                available = await client.list_loras()
            except Exception as exc:  # noqa: BLE001 - report cleanly
                await self._reply(event, f"读取 LoRA 列表失败，无法校验名称：{exc}")
                event.stop()
                return
            tool_loras, errors = self._parse_lora_segment(lora_segment, available)
            replace_defaults = profile is not None
            if errors:
                numbered = "\n".join(
                    f"{i}. {name}" for i, name in enumerate(available, 1)
                ) or "（服务器无 LoRA）"
                await self._reply(
                    event,
                    "LoRA 参数有问题，本次未开始生成：\n- " + "\n- ".join(errors)
                    + "\n可用 LoRA（可用序号代替名称）：\n" + numbered
                    + "\n" + self._usage_text(prefix, lora_prefix, model_prefix),
                )
                event.stop()
                return

        source_image = None
        if (not force_text and self.plugin_cfg.get("img2img_enabled", True)
                and self.plugin_cfg.get("img2img_auto_on_attachment", True)
                and (self._chain_images(message.chain) or any(
                    getattr(getattr(element, "type", None), "value", "") == "reply"
                    for element in message.chain or []))):
            try:
                self._img2img_options()
                source_image = await self._select_source(event)
            except ImageInputError as exc:
                await self._reply(event, f"图生图未开始：{exc}；如需忽略引用图片，请使用 --txt2img。")
                event.stop()
                return
        job = self._start_job(
            source="command",
            session=event.session.sid,
            prompt=prompt,
            width=width,
            height=height,
            count=count,
            deliver_directly=True,
            tool_loras=tool_loras,
            preset=preset,
            source_image=source_image,
            replace_default_loras=replace_defaults,
        )
        logger.info(
            "[comfyui_studio] /comfy job %s queued: %dx%d x%d preset=%s loras=%s",
            job["id"], width, height, count,
            (preset or {}).get("name", "-"), job.get("loras") or tool_loras or "-",
        )
        event.stop()

    @staticmethod
    def _parse_img2img_command(text: str) -> dict:
        result = {"width": 0, "height": 0, "count": 1}
        remaining = text.strip()
        option_names = {"d": "denoise", "denoise": "denoise", "s": "seed", "seed": "seed",
                        "i": "image_index", "image-index": "image_index", "fit": "resize_mode",
                        "source": "source_image", "mask": "mask_image", "ref": "reference_images"}
        option_names.update({"mask-index": "mask_index", "ref-index": "reference_indices"})
        seen = set()
        while remaining:
            option = re.match(r'''^--?([a-z-]+)(?:=|\s+)("[^"]*"|'[^']*'|\S+)(?:\s+|$)''', remaining)
            if option:
                key = option_names.get(option[1])
                if not key:
                    raise ImageInputError(f"未知图生图参数：{option[1]}")
                value = option[2]
                if value[:1] in ("'", '"'):
                    value = value[1:-1]
                if key in ("seed", "image_index", "mask_index", "reference_indices"):
                    value = int(value)
                elif key == "denoise":
                    value = float(value)
                if key in ("reference_images", "reference_indices"):
                    result.setdefault(key, []).append(value)
                else:
                    result[key] = value
                remaining = remaining[option.end():].lstrip()
                continue
            if remaining.startswith("-"):
                raise ImageInputError("参数格式不正确；示例：-d 0.6 --seed 123 --fit crop")
            model = re.match(r"^#([^\s]+)(?:\s+|$)", remaining)
            size = re.match(r"^(auto|\d{2,5}[xX×*]\d{2,5})(?:\s+|$)", remaining)
            count = re.match(r"^(\d{1,2})(?:\s+|$)", remaining)
            loras = re.match(r"^\[([^\]]*:[^\]]*)\](?:\s+|$)", remaining)
            if model and "model" not in seen:
                result["model"] = model[1]; seen.add("model"); match = model
            elif size and "size" not in seen:
                if size[1] != "auto":
                    result["width"], result["height"] = map(int, re.split(r"[xX×*]", size[1]))
                seen.add("size"); match = size
            elif count and "count" not in seen:
                result["count"] = int(count[1]); seen.add("count"); match = count
            elif loras and "loras" not in seen:
                result["lora_segment"] = loras[1]; seen.add("loras"); match = loras
            else:
                break
            remaining = remaining[match.end():].lstrip()
        if not remaining:
            raise ImageInputError("请在参数后写出希望怎样修改图片")
        result["prompt"] = remaining
        return result

    async def _handle_img2img_command(self, event, text):
        prefix = str(self.plugin_cfg.get("img2img_command_prefix") or "/img2img")
        if text.lower() in ("", "help", "帮助", "用法"):
            await self._reply(event,
                f"发送或引用图片后输入：{prefix} [#模型] [auto或宽x高] [张数] [-d 0.6] 提示词\n"
                f"示例：{prefix} #1 auto -d 0.45 改成水彩插画\n"
                "可选：-i 2 选择第2张图；--seed 123 固定种子；--fit crop 居中裁切；"
                "--source 图片链接；--mask 蒙版链接或 --mask-index 2；"
                "--ref 参考图链接或 --ref-index 3；[LoRA:强度]。\n"
                "蒙版白色区域修改、黑色区域保留。未指定尺寸时使用原图比例，超过上限自动缩小。")
            return
        try:
            values = self._parse_img2img_command(text)
            options = self._img2img_options(values.get("denoise"), values.get("resize_mode"), values.get("seed"))
            preset = self._select_model(values.get("model"))
            source = await self._select_source(event, values.get("source_image"), values.get("image_index", 1))
            mask, references = self._optional_image_sources(values.get("mask_image"), values.get("reference_images"))
            if values.get("mask_index") is not None:
                if mask is not None:
                    raise ImageInputError("蒙版请只指定链接或图片序号中的一种")
                mask = await self._select_source(event, image_index=values["mask_index"])
            if len(references) + len(values.get("reference_indices", [])) > 2:
                raise ImageInputError("最多可添加两张额外参考图")
            for index in values.get("reference_indices", []):
                references.append(await self._select_source(event, image_index=index))
            loras = None
            if values.get("lora_segment"):
                client = ComfyClient(str(self.plugin_cfg.get("comfy_url") or ""), timeout=self._PROBE_TIMEOUT)
                loras, errors = self._parse_lora_segment(values["lora_segment"], await client.list_loras())
                if errors:
                    raise ImageInputError("LoRA 参数无效：" + "；".join(errors))
            job = self._start_job("command", event.session.sid, values["prompt"], values["width"], values["height"],
                                  max(1, min(self._int("max_batch", 4), values["count"])), True,
                                  tool_loras=loras, preset=preset, source_image=source,
                                  mask_image=mask, reference_images=references, **options)
        except Exception as exc:
            await self._reply(event, f"图生图未开始：{exc}")

    def _last_done_job(self, session: str) -> Optional[dict]:
        """Most recent finished job with downloadable output in this session."""
        for job in reversed(_JOBS):
            if (job.get("session") == session and job.get("status") == "done"
                    and job.get("files")):
                return job
        return None

    async def _handle_again_command(self, event: KiraMessageEvent, prefix: str,
                                    lora_prefix: str, model_prefix: str,
                                    upscale_prefix: str):
        """``/comfy again`` - rerun the latest job in this session (new seed)."""
        job = self._last_done_job(event.session.sid)
        if not job:
            await self._reply(event, "这个会话还没有画过图，先用 /comfy 画一张吧。")
            return
        kwargs = {}
        if job.get("mode") == "img2img":
            kwargs = {"source_image": job.get("_source_image"),
                      "denoise": job.get("denoise"),
                      "resize_mode": job.get("resize_mode"),
                      "mask_image": job.get("_mask_image"),
                      "reference_images": job.get("_reference_images")}
            if not self.plugin_cfg.get("img2img_enabled", True):
                await self._reply(event, "图生图已关闭，请先在配置页启用后再重绘。")
                return
        preset = job.get("preset")
        width, height, count = self._clamp_size(job["width"], job["height"], job["count"])
        new_job = self._start_job(
            "command", event.session.sid, job["prompt"], width, height, count,
            True, tool_loras=job.get("tool_loras"), preset=preset,
            replace_default_loras=job.get("replace_default_loras", False), **kwargs)
        model_note = f"（模型: {preset['name']}）" if preset and preset.get("name") else ""
        logger.info("[comfyui_studio] again job %s reruns %s", new_job["id"], job["id"])

    async def _handle_upscale_command(self, event: KiraMessageEvent,
                                      upscale_prefix: str):
        """``/upscale`` - hires-fix the latest output: pixel upscale then re-sample."""
        if not self.plugin_cfg.get("img2img_enabled", True):
            await self._reply(event, "放大需要图生图，请先在配置页启用图生图。")
            return
        job = self._last_done_job(event.session.sid)
        if not job:
            await self._reply(event, "这个会话还没有可放大的作品，先用 /comfy 画一张吧。")
            return
        scale = min(2.0, max(1.1, self._float("upscale_scale", 1.5)))
        denoise = min(0.95, max(0.1, self._float("upscale_denoise", 0.8)))
        width = max(64, (round(job["width"] * scale) // 8) * 8)
        height = max(64, (round(job["height"] * scale) // 8) * 8)
        width, height, _ = self._clamp_size(width, height, 1)
        if width <= job["width"] and height <= job["height"]:
            await self._reply(event, "这张图已达到当前生成尺寸上限；请先在配置页调高最大生成像素或单边上限，再进行放大。")
            return
        source = {"kind": "trusted_path", "value": job["files"][0]}
        preset = job.get("preset")
        new_job = self._start_job(
            "command", event.session.sid, job["prompt"] or "", width, height, 1,
            True, tool_loras=job.get("tool_loras"), preset=preset,
            source_image=source, denoise=denoise, resize_mode="stretch",
            replace_default_loras=job.get("replace_default_loras", False))
        logger.info("[comfyui_studio] upscale job %s from %s (%dx%d -> %dx%d, denoise %g)",
                    new_job["id"], job["id"], job["width"], job["height"],
                    width, height, denoise)

    async def _handle_model_command(self, event: KiraMessageEvent, prefix: str,
                                    lora_prefix: str, model_prefix: str):
        """``/model`` - list configured model presets."""
        presets = self._get_presets()
        if not presets:
            await self._reply(
                event,
                "还没有配置模型预设。\n请在 WebUI 侧边栏「ComfyUI 绘图 → 模型预设」"
                "里添加（名称 + 绘图模型 + 配套文本编码器 + VAE）。",
            )
            return
        lines = [f"已配置 {len(presets)} 个模型预设："]
        lines.extend(
            f"#{i}. {p['name']} → {p['unet']}（{p.get('steps') or self._int('steps', 30)} 步）"
            for i, p in enumerate(presets, 1)
        )
        lines.append(f"用法：{prefix} #编号或名称 宽x高 [张数] 提示词（配套模型与步数自动带上）")
        await self._reply(event, "\n".join(lines))

    async def _handle_lora_command(self, event: KiraMessageEvent, keyword: str,
                                   prefix: str, lora_prefix: str):
        """``/lora [keyword]`` - list LoRAs available on the server."""
        try:
            client = ComfyClient(
                str(self.plugin_cfg.get("comfy_url") or ""),
                timeout=self._PROBE_TIMEOUT,
            )
            available = await client.list_loras()
        except Exception as exc:  # noqa: BLE001 - report cleanly
            await self._reply(event, f"读取 LoRA 列表失败：{exc}")
            return
        if keyword:
            kw = keyword.lower()
            available = [a for a in available if kw in a.lower()]
        if not available:
            await self._reply(event, "没有匹配的 LoRA。")
            return
        lines = [f"服务器上共有 {len(available)} 个 LoRA："]
        lines.extend(f"{i}. {name}" for i, name in enumerate(available, 1))
        lines.append(
            f"用法：{prefix} 832x832 1 [名称:强度,名称:强度] 提示词"
            f"（多条用逗号隔开，名称可只写一部分；查看全部用 {lora_prefix}）"
        )
        await self._reply(event, "\n".join(lines))

    def _usage_text(self, prefix: str, lora_prefix: str = "/lora",
                    model_prefix: str = "/model") -> str:
        return (
            f"直连绘图指令用法：\n"
            f"个人设置：{self.plugin_cfg.get('settings_command_prefix') or '设置'} 或 {prefix} 设置，按按钮逐步选择；保存后发送 {prefix} 提示词。\n"
            f"{prefix} [#模型编号或名称] 宽x高 [张数] [LoRA:强度,LoRA:强度] 提示词\n"
            f"示例：{prefix} #1 832x1216 2 [1:0.8,age slider:0.5] 1girl, silver hair\n"
            f"#模型 切换整套预设（绘图模型+配套编码器+VAE，编号/名称见 {model_prefix}；"
            f"省略则用默认配置）。\n"
            f"LoRA 可用序号（{lora_prefix} 列表里的序号）或名称（可只写一部分，自动匹配），"
            f"多条用逗号隔开，强度必填；省略 [..] 则只用默认 LoRA。\n"
            f"查看模型预设：{model_prefix}　查看 LoRA 列表：{lora_prefix} [关键词]\n"
            f"提示词建议用英文标签（danbooru 风格），分辨率与张数会自动按管理员"
            f"配置的上限钳制。"
            f"\n图生图：{self.plugin_cfg.get('img2img_command_prefix') or '/img2img'} [#模型] [auto或宽x高] [-d 0.6] 提示词。"
            f"附带或引用原图即可；{prefix} 带图也可自动改图，--txt2img 可强制文生图。"
            f"\n{prefix} again 重画最近一张（新随机种子）；"
            f"{self.plugin_cfg.get('upscale_command_prefix') or '/upscale'} 把最近一张放大重采样"
            f"（倍数与重绘幅度见配置页）。生成图底部带快捷按钮（可关）。"
        )

    def _get_presets(self) -> list[dict]:
        """Normalized model presets from config (order = the #number)."""
        raw = self.plugin_cfg.get("model_presets")
        presets = raw if isinstance(raw, list) else []
        out = []
        for item in presets:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            unet = str(item.get("unet") or "").strip()
            if not name or not unet:
                continue
            out.append({
                "name": name,
                "unet": unet,
                "clip": str(item.get("clip") or "").strip(),
                "clip_type": (str(item.get("clip_type") or "").strip()
                              or str(self.plugin_cfg.get("clip_type") or "stable_diffusion")),
                "vae": str(item.get("vae") or "").strip(),
                "weight_dtype": (str(item.get("weight_dtype") or "").strip()
                                 or str(self.plugin_cfg.get("weight_dtype") or "default")),
                "sampler": str(item.get("sampler") or "").strip(),
                "scheduler": str(item.get("scheduler") or "").strip(),
                "steps": self._preset_steps(item.get("steps")),
                "cfg": self._preset_cfg_raw(item.get("cfg")),
                "positive_quality": str(item.get("positive_quality") or "").strip(),
                "negative_quality": str(item.get("negative_quality") or "").strip(),
                "img2img_backend": str(item.get("img2img_backend") or "").strip(),
            })
        return out

    @staticmethod
    def _preset_cfg_raw(value) -> float:
        """Per-preset CFG as stored; 0 means inherit the global value."""
        try:
            cfg = float(value)
        except (TypeError, ValueError):
            return 0.0
        return cfg if cfg > 0 else 0.0

    def _preset_steps(self, value) -> int:
        """Per-preset steps; 0 means inherit the global default."""
        try:
            steps = int(value)
        except (TypeError, ValueError):
            return 0
        return steps if steps > 0 else 0

    @staticmethod
    def _resolve_preset(query: str, presets: list[dict]) -> tuple[Optional[dict], list[str]]:
        """Match a preset by 1-based number or fuzzy name; (preset, candidates)."""
        q = query.strip().lstrip("#").strip()
        if not q or not presets:
            return None, []
        if q.isdigit():
            index = int(q)
            if 1 <= index <= len(presets):
                return presets[index - 1], []
            return None, []
        lowered = [(p, p["name"].lower()) for p in presets]
        for preset, name in lowered:
            if name == q.lower():
                return preset, []
        hits = [p for p, name in lowered if q.lower() in name]
        if len(hits) == 1:
            return hits[0], []
        return None, [p["name"] for p in hits[:5]]

    @staticmethod
    def _match_lora_name(query: str, available: list[str]) -> tuple[Optional[str], list[str]]:
        """Fuzzy-match a LoRA name; returns (resolved, ambiguous_candidates)."""
        q = query.strip()
        if not q or not available:
            return None, []
        lowered = [(a, a.lower()) for a in available]
        for a, al in lowered:  # exact (with or without case)
            if al == q.lower():
                return a, []
        for a, al in lowered:  # missing extension
            if al == q.lower() + ".safetensors":
                return a, []
        hits = [a for a, al in lowered if q.lower() in al]
        if len(hits) == 1:
            return hits[0], []
        return None, hits[:5]

    def _parse_lora_segment(self, segment: str, available: list[str]):
        """Parse ``名称或序号:强度,名称:强度``; returns (loras, errors).

        Entries are separated by comma (half/full width) or ``|``. A
        pure-number name refers to the 1-based index of the ``/lora``
        listing (i.e. the server's LoRA order).
        """
        loras, errors = [], []
        for part in re.split(r"[,，|]", segment):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                errors.append(f"「{part}」缺少强度（格式 名称或序号:强度）")
                continue
            name, _, suffix = part.rpartition(":")
            name, suffix = name.strip(), suffix.strip()
            try:
                strength = self._clamp_strength(suffix)
            except (TypeError, ValueError):
                errors.append(f"「{part}」强度无效")
                continue
            if name.isdigit():
                index = int(name)
                if not available:
                    errors.append("服务器上没有 LoRA")
                    continue
                if not 1 <= index <= len(available):
                    errors.append(f"序号 {index} 超出范围（1-{len(available)}）")
                    continue
                loras.append({
                    "name": available[index - 1],
                    "strength_model": strength,
                    "strength_clip": strength,
                })
                continue
            resolved, ambiguous = self._match_lora_name(name, available)
            if resolved:
                loras.append({
                    "name": resolved,
                    "strength_model": strength,
                    "strength_clip": strength,
                })
            elif ambiguous:
                errors.append(f"「{name}」匹配到多个：{'、'.join(ambiguous)}")
            else:
                errors.append(f"「{name}」没有匹配到任何 LoRA")
        return loras, errors

    # ------------------------------------------------------------------
    # WebUI sidebar page + helper APIs
    # ------------------------------------------------------------------
    @register.page(
        "/index",
        menu=PageMenu(label={"zh": "ComfyUI 绘图", "en": "ComfyUI Draw"},
                      icon="Picture", order=60),
    )
    def studio_page(self) -> PluginPage:
        return PluginPage.from_folder("./web")

    #: Short timeout for interactive sidebar probes so the page never hangs.
    _PROBE_TIMEOUT = 15.0

    @register.api("GET", "/comfy-info")
    async def api_comfy_info(self):
        """Proxy ComfyUI node lists for the sidebar dropdowns."""
        client = ComfyClient(
            str(self.plugin_cfg.get("comfy_url") or ""),
            timeout=self._PROBE_TIMEOUT,
        )
        try:
            info, queue = await asyncio.gather(client.object_info(), client.queue_status())
            options = lambda node, key: client._required_options(info, node, key)
            unets = options("UNETLoader", "unet_name")
            clips = options("CLIPLoader", "clip_name")
            vaes = options("VAELoader", "vae_name")
            loras = options("LoraLoader", "lora_name")
            samplers = options("KSampler", "sampler_name")
            schedulers = options("KSampler", "scheduler")
            clip_types = options("CLIPLoader", "type")
            weight_dtypes = options("UNETLoader", "weight_dtype")
            return {
                "ok": True,
                "comfy_url": self.plugin_cfg.get("comfy_url"),
                "unets": unets,
                "clips": clips,
                "vaes": vaes,
                "loras": loras,
                "samplers": samplers,
                "schedulers": schedulers,
                "clip_types": clip_types,
                "weight_dtypes": weight_dtypes,
                "queue_running": len(queue.get("queue_running", []) or []),
                "queue_pending": len(queue.get("queue_pending", []) or []),
                "img2img_ready": all(node in info for node in ("LoadImage", "VAEEncode", "RepeatLatentBatch")),
                "img2img_backends": [name for name, node in (
                    ("standard", "VAEEncode"), ("qwen_edit", "TextEncodeQwenImageEdit"),
                    ("qwen_edit_plus", "TextEncodeQwenImageEditPlus"),
                    ("qwen_image_2_1", "TextEncodeQwenImage21")) if node in info],
                "inpaint_ready": all(node in info for node in ("VAEEncodeForInpaint", "ImageToMask", "ImageCompositeMasked", "RepeatImageBatch")),
            }
        except Exception as exc:  # noqa: BLE001 - surface the error to the page
            return {"ok": False, "error": str(exc)}

    @register.api("POST", "/test-comfy")
    async def api_test_comfy(self):
        """Connection test: system_stats + checkpoint resolution."""
        client = ComfyClient(
            str(self.plugin_cfg.get("comfy_url") or ""),
            timeout=self._PROBE_TIMEOUT,
        )
        try:
            stats = await client.system_stats()
            unet = await client.resolve_unet(
                str(self.plugin_cfg.get("unet_name") or "")
            )
            return {
                "ok": True,
                "system": stats.get("system", {}),
                "resolved_model": unet,
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @register.api("POST", "/test-bed")
    async def api_test_bed(self):
        """Upload a tiny test PNG using the current bed settings."""
        try:
            url, error = await test_upload(
                base_url=str(self.plugin_cfg.get("bed_base_url") or ""),
                upload_path=str(self.plugin_cfg.get("bed_upload_path") or "/upload"),
                file_field=str(self.plugin_cfg.get("bed_file_field") or "files"),
                token=str(self.plugin_cfg.get("bed_token") or ""),
                timeout=self._PROBE_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        if url:
            return {"ok": True, "url": url}
        return {"ok": False, "error": error}

    @register.api("GET", "/jobs")
    async def api_jobs(self):
        """Recent generation jobs for the sidebar page."""
        return {"ok": True, "jobs": [_job_public(j) for j in reversed(_JOBS)]}

    @register.api("GET", "/user-settings")
    async def api_user_settings(self):
        try:
            profiles = list(self._profile_store.profiles.values())
            return {"ok": True, "profiles": sorted(profiles, key=lambda x: x.get("updated_at", ""), reverse=True)}
        except (SettingsError, ValueError, OSError) as exc:
            raise HTTPException(500, f"读取用户配置失败：{exc}") from exc

    @register.api("PUT", "/user-settings/{profile_id}")
    async def api_update_user_settings(self, profile_id: str, request: Request):
        raw = bytearray()
        async for part in request.stream():
            raw.extend(part)
            if len(raw) > 16384:
                raise HTTPException(413, "配置内容过大")
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("配置必须是对象")
            profile = await self._admin_save_profile(profile_id, payload)
            return {"ok": True, "profile": profile}
        except (SettingsError, ValueError, TypeError, ComfyClientError) as exc:
            raise HTTPException(409 if "其他地方更新" in str(exc) else 400, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(500, "用户配置保存失败，请检查服务器存储") from exc

    @register.api("GET", "/output-image/{job_id}/{index}")
    async def api_output_image(self, job_id: str, index: int):
        job = next((j for j in _JOBS if j["id"] == job_id), None)
        if job is None or not 0 <= index < len(job["files"]):
            raise HTTPException(404, "图片不存在或任务记录已清理")
        path = Path(job["files"][index])
        if not path.is_file():
            raise HTTPException(404, "图片文件不存在")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})
