"""Persistent per-user profiles and the chat configuration wizard."""

import asyncio
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import time


class SettingsError(ValueError):
    pass


class ProfileStore:
    def __init__(self, root):
        self.path = Path(root) / "user_settings.json"
        self.lock = asyncio.Lock()
        self.drafts = {}
        self.profiles = {}
        if self.path.exists():
            # A corrupt file must remain recoverable rather than being overwritten.
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if value.get("version") != 1 or not isinstance(value.get("profiles"), dict):
                raise SettingsError("用户配置文件格式异常，请检查备份后恢复")
            self.profiles = value["profiles"]

    @staticmethod
    def profile_id(adapter, user):
        return hashlib.sha256((adapter + "\0" + user).encode()).hexdigest()[:32]

    async def save(self, key, profile, expected_revision):
        async with self.lock:
            current = self.profiles.get(key, {})
            if current.get("revision", 0) != expected_revision:
                raise SettingsError("配置已在其他地方更新，请重新读取或重新发送设置指令")
            value = copy.deepcopy(profile)
            value.update(id=key, revision=expected_revision + 1,
                         updated_at=datetime.now(timezone.utc).isoformat())
            profiles = {**self.profiles, key: value}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".json.new")
            temporary.write_text(json.dumps({"version": 1, "profiles": profiles}, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.path)
            self.profiles = profiles
            return copy.deepcopy(value)


_STORES = {}


class ChatSettingsMixin:
    @property
    def _profile_store(self):
        key = str(self._data_dir.resolve())
        if key not in _STORES:
            _STORES[key] = ProfileStore(self._data_dir)
        return _STORES[key]

    def _settings_owner(self, event):
        sid, user = self._event_owner(event)
        session = getattr(event, "session", None)
        adapter = str(getattr(session, "adapter_name", "") or sid.split(":", 1)[0])
        sender = getattr(getattr(event, "message", None), "sender", None)
        if sender is None:
            messages = getattr(event, "messages", None) or []
            sender = getattr(messages[-1], "sender", None) if messages else None
        if not user or not adapter:
            raise SettingsError("无法识别当前用户，请使用本人账号发送设置指令")
        return {"adapter": adapter, "user_id": user,
                "nickname": str(getattr(sender, "nickname", "") or user)}

    def _user_profile(self, event):
        owner = self._settings_owner(event)
        key = ProfileStore.profile_id(owner["adapter"], owner["user_id"])
        return copy.deepcopy(self._profile_store.profiles.get(key))

    def _profile_model(self, profile):
        if not profile or profile.get("model") is None:
            return None
        model = profile["model"]
        for preset in self._get_presets():
            if preset["name"] == model.get("name") and preset["unet"] == model.get("unet"):
                return preset
        raise SettingsError("保存的底模预设已被修改或移除，请重新设置底模")

    def _validate_profile(self, values):
        result = {}
        model = values.get("model")
        if model is not None and (not isinstance(model, dict) or not isinstance(model.get("name"), str) or not isinstance(model.get("unet"), str)):
            raise SettingsError("请选择有效的模型预设")
        result["model"] = None if model is None else {"name": model["name"], "unet": model["unet"]}
        self._profile_model(result)
        for key in ("width", "height", "count"):
            value = values.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise SettingsError("宽度、高度和张数必须是正整数")
        size = self._clamp_size(values["width"], values["height"], values["count"])
        if tuple(values[k] for k in ("width", "height", "count")) != size:
            raise SettingsError("尺寸或张数超过当前上限，或尺寸不是 8 的倍数，请调整后再保存")
        result.update(zip(("width", "height", "count"), size))
        loras = values.get("loras")
        if loras is not None:
            if not isinstance(loras, list) or len(loras) > 8:
                raise SettingsError("最多可保存 8 个 LoRA")
            cleaned, seen = [], set()
            for item in loras:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip() or item["name"] in seen:
                    raise SettingsError("LoRA 名称不能为空或重复")
                name = item["name"].strip()
                if name in seen:
                    raise SettingsError("LoRA 名称不能重复")
                try:
                    sm = float(item.get("strength_model", 1))
                    sc = float(item.get("strength_clip", sm))
                    if not (-1 <= sm <= 2 and -1 <= sc <= 2):
                        raise ValueError()
                except (TypeError, ValueError):
                    raise SettingsError("LoRA 强度必须在 -1~2 之间")
                seen.add(name)
                cleaned.append({"name": name, "strength_model": sm, "strength_clip": sc})
            loras = cleaned
        result["loras"] = loras
        return result

    def _profile_summary(self, profile):
        model = (profile.get("model") or {}).get("name") or str(self.plugin_cfg.get("unet_name") or "全局默认底模")
        loras = profile.get("loras")
        note = "沿用全局默认" if loras is None else "、".join(f"{x['name']}:{x['strength_model']:g}" for x in loras) or "不使用 LoRA"
        return f"底模：{model}\n分辨率：{profile['width']} × {profile['height']}\n张数：{profile['count']}\nLoRA：{note}"

    async def _start_settings(self, event):
        owner = self._settings_owner(event)
        key = ProfileStore.profile_id(owner["adapter"], owner["user_id"])
        store = self._profile_store
        # New drafts invalidate previous buttons but keep the saved profile active.
        draft = {"owner": owner, "key": key, "session": self._event_owner(event)[0],
                 "token": secrets.token_hex(4), "revision": 0, "expires": time.monotonic() + 1800,
                 "base_revision": store.profiles.get(key, {}).get("revision", 0),
                 "stage": "model", "page": 0, "values": {},
                 "models": [None] + [{"name": p["name"], "unet": p["unet"]} for p in self._get_presets()]}
        for old_key in list(store.drafts):
            if store.drafts[old_key]["expires"] < time.monotonic():
                del store.drafts[old_key]
        store.drafts[key] = draft
        await self._show_settings(event, draft)

    async def _show_settings(self, event, draft):
        stage, values = draft["stage"], draft["values"]
        choices, controls = [], []
        if stage == "model":
            title = "1/5 · 选择底模"
            options = [(p["name"] if p else "使用全局默认底模", i) for i, p in enumerate(draft["models"])]
        elif stage == "size":
            title = "2/5 · 选择分辨率"
            sizes = []
            for w, h in [(1024,1024),(832,1216),(1216,832),(896,1152),(1152,896),(768,1024),(1024,768),(512,512)]:
                size = self._clamp_size(w, h, 1)[:2]
                if size not in sizes:
                    sizes.append(size)
            draft["sizes"] = sizes
            options = [(f"{w} × {h}", i) for i, (w,h) in enumerate(sizes)]
        elif stage == "count":
            title = "3/5 · 选择张数"
            options = [(f"{n} 张", n) for n in range(1, max(1, min(100, self._int("max_batch", 4))) + 1)]
        elif stage == "lora":
            title = "4/5 · 选择 LoRA"
            options = [(Path(name).stem, i) for i, name in enumerate(draft["available_loras"])]
            controls = [("使用默认 LoRA", "defaults", "0"), ("不使用 LoRA", "none", "0"), ("选好了", "finish", "0")]
        elif stage == "strength":
            title = "4/5 · 选择 LoRA 强度"
            options = [(str(s), s) for s in (0.4,0.6,0.8,1.0,1.2,1.5,2.0)]
        else:
            title = "5/5 · 确认并保存"
            options = []
            controls = [("保存配置", "save", "0"), ("重新选择", "restart", "0")]
        page_size = 15 if stage == "lora" else 20
        pages = max(1, (len(options) + page_size - 1) // page_size)
        draft["page"] = min(max(0, draft["page"]), pages - 1)
        start = draft["page"] * page_size
        for label, value in options[start:start + page_size]:
            if stage == "lora" and any(x["name"] == draft["available_loras"][value] for x in values.get("loras", [])):
                label = "✓ " + label
            choices.append((label[:30], "pick", str(value)))
        if pages > 1:
            navigation = []
            if draft["page"] > 0:
                navigation.append(("上一页", "page", str(draft["page"] - 1)))
            if draft["page"] < pages - 1:
                navigation.append(("下一页", "page", str(draft["page"] + 1)))
            choices.extend(navigation)
            title += f" · 第 {draft['page'] + 1}/{pages} 页"
        choices.extend(controls)
        choices.append(("取消设置", "cancel", "0"))
        draft["allowed"] = {(action, value) for _, action, value in choices}
        text = f"{draft['owner']['nickname']} 的生图设置\n{title}"
        if stage == "lora":
            selected = values.get("loras", [])
            text += "\n已选择：" + ("、".join(f"{x['name']}:{x['strength_model']:g}" for x in selected) or "无")
            text += "\n选择一个 LoRA 后单独选择强度，可继续添加；再次点已选项可移除。"
            if not options:
                text += "\n当前没有可用 LoRA，可选择不使用或沿用默认。"
        elif stage == "strength":
            text += "\n" + draft["pending_lora"]
        elif stage == "confirm":
            text += "\n" + self._profile_summary(values) + "\n保存后覆盖你上次的配置。"
        rows = []
        buttons = []
        for i, (label, action, value) in enumerate(choices):
            command = f"/__comfy_settings {draft['token']} {draft['revision']} {action} {value}"
            buttons.append({"id": f"s{i}", "render_data": {"label": label, "style": 1},
                            "action": {"type": 1, "data": command,
                                       "permission": {"type": 0, "specify_user_ids": [draft["owner"]["user_id"]]}}})
        for i in range(0, len(buttons), 5):
            rows.append({"buttons": buttons[i:i+5]})
        if len(rows) > 5:
            raise SettingsError("选项数量超出按钮限制")
        await self._send_settings_page(event, text, {"content": {"rows": rows}}, choices, draft)

    async def _settings_action(self, event, text):
        owner = self._settings_owner(event)
        key = ProfileStore.profile_id(owner["adapter"], owner["user_id"])
        store = self._profile_store
        draft = store.drafts.get(key)
        parts = text.split()
        if (len(parts) != 5 or not draft or parts[1] != draft["token"]
                or parts[2] != str(draft["revision"]) or draft["expires"] < time.monotonic()
                or draft["session"] != self._event_owner(event)[0]):
            raise SettingsError("这个按钮已失效或属于其他用户，请使用自己最新一条设置消息")
        # Claim the transition synchronously before any network await.
        if (parts[3], parts[4]) not in draft.get("allowed", set()):
            raise SettingsError("选项不属于当前步骤，请使用最新设置消息")
        draft["revision"] += 1
        draft["allowed"] = set()
        action, value = parts[3:]
        stage = draft["stage"]
        if action == "cancel":
            del store.drafts[key]
            await self._reply(event, "已取消本次设置，之前保存的配置继续生效。")
            return
        if action == "restart":
            await self._start_settings(event)
            return
        if action == "page":
            draft["page"] = int(value)
        elif stage == "model" and action == "pick":
            draft["values"]["model"] = draft["models"][int(value)]
            draft.update(stage="size", page=0)
        elif stage == "size" and action == "pick":
            draft["values"].update(zip(("width", "height"), draft["sizes"][int(value)]))
            draft.update(stage="count", page=0)
        elif stage == "count" and action == "pick":
            count = int(value)
            if not 1 <= count <= min(100, self._int("max_batch", 4)):
                raise SettingsError("张数超出上限")
            draft["values"].update(count=count, loras=[])
            try:
                draft["available_loras"] = await self._make_settings_client().list_loras()
            except Exception:
                await self._reply(event, "暂时无法读取 LoRA 列表，请稍后重新点击张数重试。")
                await self._show_settings(event, draft)
                return
            draft.update(stage="lora", page=0)
        elif stage == "lora" and action == "pick":
            name = draft["available_loras"][int(value)]
            selected = draft["values"]["loras"]
            existing = next((x for x in selected if x["name"] == name), None)
            if existing:
                selected.remove(existing)
            elif len(selected) >= 8:
                raise SettingsError("最多选择 8 个 LoRA，请先移除一个或点击选好了")
            else:
                draft.update(pending_lora=name, stage="strength", page=0)
        elif stage == "strength" and action == "pick":
            strength = float(value)
            if strength not in (0.4,0.6,0.8,1.0,1.2,1.5,2.0):
                raise SettingsError("强度选项无效")
            draft["values"]["loras"].append({"name": draft["pending_lora"], "strength_model": strength, "strength_clip": strength})
            draft.update(stage="lora", page=0)
        elif stage == "lora" and action in ("defaults", "none", "finish"):
            if action != "finish":
                draft["values"]["loras"] = None if action == "defaults" else []
            draft.update(stage="confirm", page=0)
        elif stage == "confirm" and action == "save":
            values = self._validate_profile(draft["values"])
            saved = await store.save(key, {**owner, **values}, draft["base_revision"])
            del store.drafts[key]
            prefix = str(self.plugin_cfg.get("command_prefix") or "/comfy")
            await self._reply(event, "已保存你的生图配置\n" + self._profile_summary(saved) + f"\n接下来发送：{prefix} 提示词")
            return
        else:
            raise SettingsError("设置步骤已变化，请使用最新一条消息里的按钮")
        await self._show_settings(event, draft)

    async def _admin_save_profile(self, key, payload):
        current = self._profile_store.profiles.get(key)
        if current is None:
            raise SettingsError("用户配置不存在，请刷新列表")
        values = self._validate_profile(payload)
        if values["loras"]:
            available = await self._make_settings_client().list_loras()
            if any(x["name"] not in available for x in values["loras"]):
                raise SettingsError("所选 LoRA 已不存在，请刷新模型列表")
        revision = payload.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise SettingsError("缺少配置版本，请刷新列表")
        return await self._profile_store.save(key, {**current, **values}, revision)
