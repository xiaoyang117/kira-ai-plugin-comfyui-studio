# Changelog

## 0.0.1 — First public release

- First public GitHub and plugin-store release for KiraAI >=3.0.0.
- Text-to-image, image-to-image, masked editing, Qwen editing, LoRA and upscaling.
- Sequential chat configuration, paginated LoRA selection and persistent per-user profiles, with administrator editing.
- All command features default to disabled; AI generation stays available.
- AI jobs send local image files without progress notices, Markdown, buttons or image-bed uploads.
- Public defaults use localhost and leave optional image-bed uploads disabled.
- Verified with 52 regression tests, configuration-page interaction checks and user testing on KiraAI 3.0.

## Internal development history

The versions below were internal development labels before public version 0.0.1, not published release numbers.

## 1.6.2 — 2026-11-09

- Send AI-generated images directly as local image files for both text-to-image and image-to-image calls, without progress notices, image-bed uploads, Markdown or action buttons.
- Keep command generation notices and configured delivery unchanged.
- Track successfully delivered files so a partial media-send retry does not resend the successful part; report delivery failures truthfully to the AI.

## 1.6.1 — 2026-11-09

- Add a default-off master switch for every command feature, including the settings wizard, custom prefixes and old callback buttons.
- Consume disabled plugin commands before LLM processing; preserve natural-language AI image generation and configuration APIs.
- Hide command action buttons while disabled, retain source links, and prevent AI from opening an unusable settings wizard.
- Require explicit opt-in after upgrades instead of inheriting the legacy default-on command flag.

## 1.6.0 — 2026-11-08

- Add a per-user chat setup wizard: model, resolution, batch, paginated LoRA selection, per-LoRA strength and confirmation; each transition sends a separate keyboard message within the 5×5 limit.
- Bind wizard buttons to the sender/session and invalidate consumed, cancelled, restarted and expired steps.
- Persist profiles atomically by adapter and user; bare `/comfy prompt` reuses saved preferences, with explicit command arguments supported.
- Add a native `comfyui_user_settings` tool and a configurable setup command.
- Add authenticated profile listing/editing APIs and a searchable, paginated sidebar administration tab with independent saves and optimistic revisions.
- Emit one consistent generation notice with the actual model, dimensions, batch and resolved LoRAs for command and AI generation paths.

## 1.5.1 — 2026-11-08

- Reorganize the configuration-only page into connection, defaults, image-editing settings, presets, LoRA, delivery and task history; fix unclosed nested panels.
- Keep retry, upscale and callback-button behavior from 1.5.0; move message buttons to delivery settings and synchronize Chinese/English configuration labels.
- Await locked direct delivery before returning an LLM result, preventing duplicate images during concurrent completion.
- Preserve masks and extra reference images when retrying an image-editing task.
- Explain disabled image editing or generation-size limits before attempting upscale.
- Refresh documentation to match chat-only generation and configuration-only WebUI.

## 1.5.0 — 2026-11-08

- Fix a crash that killed every chat command: the im_message hook used `event.sid`, which only exists on batch events; non-@ group messages raised AttributeError before command matching (commands stopped activating).
- Add a text-only button bar under every delivered image (再来一张 / 倍数放大 / 原图直链), toggleable via `buttons_enabled`.
- Buttons use the bridge keyboard payload with graceful degradation: delivery falls back to plain markdown when the bridge is absent.
- Add `/comfy again` to rerun the latest job in the session with a fresh seed.
- Add `/upscale` hires-fix: pixel upscale by `upscale_scale` (default 1.5x) through the img2img pipeline at `upscale_denoise` (default 0.8), reusing the original prompt/preset/LoRA.
- Deliver markdown images by LOCAL file path: the bridge re-hosts them to QQ's COS via the official chunk upload, so rendering no longer depends on the image bed being reachable (a slow bed previously timed out the bridge's download and broke images). The bed copy remains as the archive behind the source-link button.
- Fix button clicks not firing: the bridge embeds the clicker's nickname in the synthetic event text ("[按钮] <昵称> 点击了：<data>"), which the prefix match missed — clicks leaked to the LLM, which regenerated with different params. Now a regex extracts the data from both formats, matches the command, and stops the event so the LLM stays out.
- Wire the `comfyui_img2img` LLM tool to buttons-enabled delivery too (it previously returned a plain markdown block for the LLM to re-send, without buttons), and guard `_deliver_job` with a lock against double delivery when the tool wait races job completion.
- Action buttons are now CALLBACK buttons: clicks fire immediately in groups (command buttons only filled the input box there). The bridge's synthetic click event ("[按钮] 用户点击了：…") is recognised and executed directly, without @ or manual send.


## 1.4.1 — 2026-11-08

- The web page is configuration-only again: removed the image-editing workspace UI and its page-driven generation/upload APIs (`/generate`, `/inputs`, `/input-image`), keeping `/output-image` for job-history thumbnails.
- Image-to-image remains available through `/img2img`, `/comfy` with attached images, and the `comfyui_img2img` LLM tool; the removed workspace settings live in the "连接与限制" tab.
- Dropped the two regression tests that targeted the removed page APIs; command/tool paths keep their coverage.


## 1.4.0 — 2026-11-08

- Add the native `comfyui_img2img` LLM tool, current/quoted/recent image selection, explicit source URLs and upload identifiers.
- Add image-to-image, mask-based inpainting, repeatable seeds, aspect-aware sizing, batches, preset overrides and LoRA support.
- Support generic VAE workflows, Qwen Edit, Qwen Edit Plus and Qwen Image 2.1 native image conditioning, including up to two extra reference images for multi-image backends.
- Add bounded ComfyUI image uploads, EXIF orientation normalization, image validation, and per-sender/session image history isolation.
- Add a full image-editing workspace with drag-and-drop, source/mask/reference uploads, progress, authenticated output previews and iterative editing.
- Add authenticated input, output and generation APIs; exclude source URLs and credentials from job history.
- Add `/img2img`, `/comfy --img2img`, automatic attached-image handling and an explicit `--txt2img` override.
- Preserve all existing text-to-image and message-delivery flows; zero denoise performs resizing without sampling.
- Add Chinese and English configuration labels and hints for image-editing settings.

## 1.3.0 — 2026-11-08

### Fixes

- Build workflow inputs as typed JSON values, preserving quotes, newlines, backslashes and literal placeholder text in prompts.
- Fall back to media delivery for the entire batch when any image-bed upload fails.
- Report missing configured diffusion models instead of silently selecting another model.
- Keep active jobs and their original settings during configuration saves; disabling or explicitly reloading the plugin still cancels active jobs.
- Hand off image delivery when an LLM tool call is cancelled by the framework.
- Keep aspect ratio when limiting the maximum side and enforce small pixel budgets.
- Exclude unavailable LoRAs even when the server's list is empty.
- Avoid stale or growing tool descriptions and accidental command prefix matches.
- Record failed delivery accurately in job history.

### Workspace

- New responsive green-and-neutral workspace with connection status and dark theme support.
- Track unsaved changes, support Ctrl/Cmd+S, and prevent saving after configuration loading fails.
- Preserve model selections when refreshing lists, retain explicit inheritance options in presets, and validate incomplete presets before saving.
- Record edits to preset sampling parameters and preserve zero LoRA strength.
- Show image thumbnails, readable job states, delivery errors and automatic job refresh while the history tab is visible.
- Read model metadata once instead of repeating node-schema requests.

### Validation

Run `python -m unittest discover -s tests -v` from the plugin directory. These tests run offline and do not submit generation jobs or send chat messages.

## Public package metadata

- Set the repository URL and the tested minimum KiraAI version to 3.0.0.
- Use localhost as the default ComfyUI endpoint; disable optional image-bed uploads by default and leave the service URL empty. Existing saved settings are unaffected.
- Refresh installation and AI delivery documentation.
