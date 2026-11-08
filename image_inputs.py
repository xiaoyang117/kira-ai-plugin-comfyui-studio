"""Bounded image input storage and deterministic image/mask preparation."""

import asyncio
import base64
import binascii
from io import BytesIO
from pathlib import Path
import re
import uuid

from PIL import Image, ImageOps, UnidentifiedImageError


class ImageInputError(ValueError):
    pass


class ImageInputStore:
    def __init__(self, root: Path, *, max_bytes: int = 20 * 1024 * 1024,
                 max_pixels: int = 40000000, downloader=None, timeout: float = 30):
        self.root = root
        self.max_bytes = max_bytes
        self.max_pixels = max_pixels
        self.downloader = downloader
        self.timeout = timeout

    def path(self, input_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", str(input_id)):
            raise ImageInputError("图片编号无效，请重新上传图片")
        path = self.root / (input_id + ".png")
        if not path.is_file():
            raise ImageInputError("原图不存在或已清理，请重新上传图片")
        return path

    def decode(self, data: bytes) -> Image.Image:
        if not data or len(data) > self.max_bytes:
            raise ImageInputError(f"图片为空或超过 {self.max_bytes // 1048576} MB 限制")
        try:
            with Image.open(BytesIO(data)) as image:
                if image.width * image.height > self.max_pixels:
                    raise ImageInputError(f"原图像素过大，上限为 {self.max_pixels:,} 像素")
                image.seek(0)
                image = ImageOps.exif_transpose(image)
                image.load()
                rgba = image.convert("RGBA")
                result = Image.new("RGB", rgba.size, "white")
                result.paste(rgba, mask=rgba.getchannel("A"))
                return result
        except ImageInputError:
            raise
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ImageInputError("无法读取图片，请使用有效的 PNG、JPEG 或 WebP 图片") from exc

    def _save(self, data: bytes) -> dict:
        image = self.decode(data)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        normalized = buffer.getvalue()
        if len(normalized) > self.max_bytes:
            raise ImageInputError("图片转换后体积过大，请缩小图片后重试")
        self.root.mkdir(parents=True, exist_ok=True)
        input_id = uuid.uuid4().hex
        (self.root / (input_id + ".png")).write_bytes(normalized)
        return {"id": input_id, "width": image.width, "height": image.height}

    async def save(self, data: bytes) -> dict:
        return await asyncio.to_thread(self._save, data)

    async def read(self, source) -> bytes:
        trusted = isinstance(source, dict) and source.get("kind") == "trusted_path"
        value = source.get("value", "") if isinstance(source, dict) else str(source or "")
        if trusted:
            path = Path(value)
            if not path.is_file() or path.stat().st_size > self.max_bytes:
                raise ImageInputError("聊天图片已失效或体积过大，请重新发送图片")
            data = await asyncio.to_thread(path.read_bytes)
        elif value.startswith(("input:", "upload:")) or re.fullmatch(r"[0-9a-f]{32}", value):
            path = self.path(value.split(":", 1)[-1])
            data = await asyncio.to_thread(path.read_bytes)
        elif value.startswith(("http://", "https://")):
            if self.downloader is None:
                raise ImageInputError("图片下载服务不可用")
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / (uuid.uuid4().hex + ".download")
            try:
                await asyncio.wait_for(self.downloader(value, str(path), timeout=self.timeout,
                                                       max_bytes=self.max_bytes), self.timeout + 2)
                data = await asyncio.to_thread(path.read_bytes)
            except Exception as exc:
                raise ImageInputError("下载原图失败，请重新发送图片或检查图片链接") from exc
            finally:
                path.unlink(missing_ok=True)
        else:
            if value.startswith("data:"):
                header, sep, value = value.partition(",")
                if not sep or not header.startswith("data:image/") or ";base64" not in header:
                    raise ImageInputError("图片数据格式无效，需要 base64 图片")
            elif value.startswith("base64://"):
                value = value[9:]
            elif not (isinstance(source, dict) and source.get("kind") == "base64"):
                raise ImageInputError("请提供图片链接、上传编号，或在消息中附带图片")
            if len(value) > ((self.max_bytes + 2) // 3) * 4 + 4:
                raise ImageInputError("图片超过上传体积限制")
            try:
                data = base64.b64decode(value, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ImageInputError("图片 base64 数据无效") from exc
        if not data or len(data) > self.max_bytes:
            raise ImageInputError("图片为空或超过上传体积限制")
        return data

    @staticmethod
    def resize(image: Image.Image, size: tuple[int, int], mode: str, *, mask=False):
        method = Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS
        if mode == "stretch":
            return image.resize(size, method)
        if mode == "crop":
            return ImageOps.fit(image, size, method=method)
        if mode != "contain":
            raise ImageInputError("尺寸适配方式必须是 contain、crop 或 stretch")
        fitted = ImageOps.contain(image, size, method=method)
        canvas = Image.new(image.mode, size, 0 if mask else "white")
        canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
        return canvas

    def prepare(self, data: bytes, width: int, height: int, mode: str,
                mask_data: bytes | None = None) -> tuple[bytes, bytes | None]:
        image = self.decode(data)
        original_size = image.size
        image = self.resize(image, (width, height), mode)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        mask_bytes = None
        if mask_data is not None:
            mask = self.decode(mask_data).convert("L")
            if mask.size != original_size:
                raise ImageInputError("蒙版尺寸必须与原图一致；白色区域重绘，黑色区域保留")
            mask = self.resize(mask, (width, height), mode, mask=True)
            mask_buffer = BytesIO()
            mask.save(mask_buffer, format="PNG")
            mask_bytes = mask_buffer.getvalue()
        return buffer.getvalue(), mask_bytes
