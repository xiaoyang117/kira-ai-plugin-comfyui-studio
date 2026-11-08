"""Offline regression tests; no ComfyUI jobs or chat messages are sent."""
import ast
import asyncio
import base64
from io import BytesIO
import importlib.util
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from PIL import Image as PILImage


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("studio_test_client", ROOT / "comfy_client.py")
client_module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = client_module
spec.loader.exec_module(client_module)
input_spec = importlib.util.spec_from_file_location("studio_test_inputs", ROOT / "image_inputs.py")
input_module = importlib.util.module_from_spec(input_spec)
input_spec.loader.exec_module(input_module)
settings_spec = importlib.util.spec_from_file_location("studio_test_settings", ROOT / "chat_settings.py")
settings_module = importlib.util.module_from_spec(settings_spec)
settings_spec.loader.exec_module(settings_module)


class BasePlugin:
    def __init__(self, ctx, cfg):
        self.ctx, self.plugin_cfg = ctx, cfg


class Decorators:
    def __getattr__(self, name):
        return lambda *a, **kw: lambda fn: fn


class Element:
    def __init__(self, text=None, image=None):
        self.text, self.image = text, image


class ImageElement(Element):
    def __init__(self, image):
        super().__init__(image=image)
        self.file = image
        self.file_type = "url"


class TextElement(Element):
    pass


tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
tree.body = [n for n in tree.body if not (
    isinstance(n, ast.ImportFrom) and (n.level or (n.module or "").startswith("core")))]
ns = dict(__file__=str(ROOT / "main.py"), BasePlugin=BasePlugin,
          PluginContext=object, PluginPage=object, PageMenu=lambda **kw: None,
          Priority=SimpleNamespace(HIGH=1), logger=logging.getLogger("test"),
          on=Decorators(), register=Decorators(), Image=ImageElement, Text=TextElement,
          MessageChain=list, KiraMessageEvent=object,
          ComfyClient=client_module.ComfyClient,
          ComfyClientError=client_module.ComfyClientError,
          upload_image=AsyncMock(), test_upload=AsyncMock(),
          ImageInputStore=input_module.ImageInputStore, ImageInputError=input_module.ImageInputError,
          ChatSettingsMixin=settings_module.ChatSettingsMixin, SettingsError=settings_module.SettingsError,
          download_file=AsyncMock(), get_data_path=lambda: Path(tempfile.gettempdir()))
exec(compile(tree, str(ROOT / "main.py"), "exec"), ns)
Plugin = ns["ComfyStudioPlugin"]


class Regressions(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ctx = SimpleNamespace(get_plugin_data_dir=lambda: self.temp.name,
                                   send_message_chain=AsyncMock(), plugin_mgr=None)
        self.plugin = Plugin(self.ctx, {"comfy_url": "http://test", "send_mode": "markdown", "command_features_enabled": True})
        ns["_JOBS"].clear()
        ns["_TASKS"].clear()
        ns["_RECENT_INPUTS"].clear()

    async def asyncTearDown(self):
        self.ctx.plugin_mgr = None
        await self.plugin.terminate()
        self.temp.cleanup()

    def graph(self, prompt):
        return self.plugin._build_graph("model", "clip", "stable_diffusion", "vae", "default",
            prompt, "negative", 1024, 768, 2, 123, 20, 4.0, "euler", "simple")

    def test_prompt_round_trip(self):
        for prompt in ['sign saying "hello"', 'line one\nline two', r'C:\models\new',
                       '__CFG__ __NEGATIVE__', '中文与 emoji 🎨']:
            graph = self.graph(prompt)
            self.assertEqual(graph['4']['inputs']['text'], prompt)
            self.assertEqual(graph['6']['inputs']['width'], 1024)
            self.assertEqual(graph['7']['inputs']['cfg'], 4.0)

    def test_lora_wiring(self):
        graph = self.graph('cat')
        self.plugin._apply_loras(graph, [dict(name='a', strength_model=0, strength_clip=0),
                                        dict(name='b', strength_model=1, strength_clip=1)])
        self.assertEqual(graph['11']['inputs']['model'], ['10', 0])
        self.assertEqual(graph['7']['inputs']['model'], ['11', 0])
        self.assertEqual(graph['4']['inputs']['clip'], ['11', 1])

    def test_size_keeps_aspect_and_limits(self):
        w, h, n = self.plugin._clamp_size(3000, 1000, 9)
        self.assertEqual((w, h, n), (1536, 512, 4))
        for size in [(64, 99999), (99999, 99999), (0, 0), (-1, -1)]:
            w, h, _ = self.plugin._clamp_size(*size, 1)
            self.assertLessEqual(w*h, 1048576)
            self.assertEqual(w % 8 + h % 8, 0)
        self.plugin.plugin_cfg['max_pixels'] = 4096
        self.assertEqual(self.plugin._clamp_size(99999, 64, 1), (64, 64, 1))

    async def test_missing_model_does_not_fall_back(self):
        client = client_module.ComfyClient('http://test')
        client.list_unets = AsyncMock(return_value=['first'])
        with self.assertRaises(client_module.ComfyClientError):
            await client.resolve_unet('missing')
        self.assertEqual(await client.resolve_unet(''), 'first')

    async def test_partial_upload_sends_every_file(self):
        job = dict(status='done', delivered=False, session='test', urls=['https://test/one'],
                   files=['one.png', 'two.png'], width=1024, height=768)
        await self.plugin._deliver_job(job)
        sent = self.ctx.send_message_chain.call_args_list
        self.assertEqual([c.args[1][0].image for c in sent], ['one.png', 'two.png'])
        self.assertTrue(job['delivered'])

    async def test_send_failure_is_not_reported_as_delivered(self):
        self.ctx.send_message_chain.side_effect = RuntimeError('network unavailable')
        job = dict(status='done', delivered=False, session='test', urls=[], files=['one.png'])
        await self.plugin._deliver_job(job)
        self.assertFalse(job['delivered'])
        self.assertIn('network unavailable', job['delivery_error'])

    async def test_save_preserves_job_and_original_settings(self):
        entered, release = asyncio.Event(), asyncio.Event()
        observed = []
        async def execute(worker, job):
            entered.set(); await release.wait()
            observed.append(worker.plugin_cfg['comfy_url'])
        with patch.object(Plugin, '_execute_job', execute):
            self.plugin._start_job('test', 'sid', 'cat', 64, 64, 1, False)
            await entered.wait()
            new_cfg = {**self.plugin.plugin_cfg, 'comfy_url': 'http://new'}
            self.ctx.plugin_mgr = SimpleNamespace(plugin_configs={'comfyui_studio': new_cfg},
                                                 is_plugin_enabled=lambda key: True)
            await self.plugin.terminate()
            replacement = Plugin(self.ctx, new_cfg)
            self.assertEqual(len(replacement._tasks), 1)
            release.set()
            await asyncio.gather(*replacement._tasks)
        self.assertEqual(observed, ['http://test'])

    async def test_disable_cancels_job(self):
        task = asyncio.create_task(asyncio.sleep(100))
        self.plugin._tasks.add(task)
        self.ctx.plugin_mgr = SimpleNamespace(plugin_configs={'comfyui_studio': {'enabled': False}},
                                             is_plugin_enabled=lambda key: False)
        await self.plugin.terminate()
        self.assertTrue(task.cancelled())

    def test_no_available_loras_means_none_applied(self):
        self.plugin.plugin_cfg['default_loras'] = 'missing:1'
        self.assertEqual(self.plugin._resolve_loras(None, []), [])

    async def test_command_master_switch_defaults_closed_even_for_legacy_configs(self):
        self.plugin.plugin_cfg.pop('command_features_enabled')
        self.plugin.plugin_cfg['command_enabled'] = True
        event = self.event([TextElement('/comfy cat')])
        with patch.object(self.plugin, '_start_job') as start:
            await self.plugin.handle_command(event)
        start.assert_not_called()
        event.stop.assert_called_once()
        self.assertIn('已关闭', self.ctx.send_message_chain.call_args.args[1][0].text)

    async def test_command_master_switch_blocks_commands_and_old_callbacks(self):
        self.plugin.plugin_cfg['command_features_enabled'] = False
        commands = ['/comfy cat', '/comfy 512x512 cat', '/comfy --img2img cat',
                    '/img2img cat', '/comfy again', '/upscale', '/model', '/lora cat',
                    '设置', '/comfy 设置', '/comfy setup', '/comfy settings',
                    '/__comfy_settings token 0 pick 1',
                    '[按钮] Alice 点击了：/__comfy_settings token 0 save 0',
                    '[按钮] 用户点击了：/comfy again', '[按钮] Alice 点击了：/upscale']
        with patch.object(self.plugin, '_start_job') as start, \
             patch.object(self.plugin, '_settings_action', AsyncMock()) as action, \
             patch.object(self.plugin, '_start_settings', AsyncMock()) as setup:
            for text in commands:
                with self.subTest(command=text):
                    event = self.event([TextElement(text)])
                    event.is_notice = text.startswith('[按钮]')
                    await self.plugin.handle_command(event)
                    event.stop.assert_called_once()
            start.assert_not_called(); action.assert_not_awaited(); setup.assert_not_awaited()
        self.assertFalse(ns['_JOBS'])
        self.assertFalse(self.plugin._profile_store.profiles)

    async def test_closed_commands_honor_custom_prefixes_and_leave_normal_chat(self):
        self.plugin.plugin_cfg.update(command_features_enabled=False, command_prefix='画图',
            img2img_command_prefix='改图', upscale_command_prefix='放大',
            model_command_prefix='底模', lora_command_prefix='风格', settings_command_prefix='绘图设置')
        for text in ['画图 cat', '改图 cat', '放大', '底模', '风格', '绘图设置', '[按钮] 用户点击了：画图 again']:
            event = self.event([TextElement(text)])
            await self.plugin.handle_command(event)
            event.stop.assert_called_once()
        for text in ['请帮我画图', '画图很好玩', '[按钮] 用户点击了：其他插件功能']:
            event = self.event([TextElement(text)])
            await self.plugin.handle_command(event)
            event.stop.assert_not_called()

    async def test_closed_commands_keep_ai_generation_but_disable_settings_wizard(self):
        self.plugin.plugin_cfg['command_features_enabled'] = False
        event = self.event()
        with patch.object(self.plugin, '_start_settings', AsyncMock()) as setup:
            self.assertIn('已关闭', await self.plugin.tool_user_settings(event))
            setup.assert_not_awaited()
        with patch.object(self.plugin, '_start_job', return_value={}) as start, \
             patch.object(self.plugin, '_wait_for_job', AsyncMock(return_value='generated')):
            self.assertEqual(await self.plugin.tool_generate_image(event, 'cat'), 'generated')
            start.assert_called_once()

    def test_closed_commands_remove_action_buttons_but_keep_source_links(self):
        self.plugin.plugin_cfg['command_features_enabled'] = False
        self.assertIsNone(self.plugin._buttons_payload({'urls': []}))
        payload = self.plugin._buttons_payload({'urls': ['https://test/image']})
        buttons = [b for row in payload['content']['rows'] for b in row['buttons']]
        self.assertEqual([b['action']['type'] for b in buttons], [0])

    @staticmethod
    def png(size=(160, 80), color='red'):
        buffer = BytesIO()
        PILImage.new('RGB', size, color).save(buffer, format='PNG')
        return buffer.getvalue()

    def event(self, chain=None, user='alice', sid='session'):
        return SimpleNamespace(sid=sid, session=SimpleNamespace(sid=sid),
            message=SimpleNamespace(chain=chain or [], sender=SimpleNamespace(user_id=user)),
            is_notice=False, stop=Mock())

    async def test_im_message_without_batch_sid_and_callback_routing(self):
        event=self.event([TextElement('hello'),ImageElement('https://test/source')])
        del event.sid
        await self.plugin.handle_command(event)
        self.assertIn(('session','alice'),ns['_RECENT_INPUTS'])
        event.message.chain=[TextElement('[按钮] 萧洋 点击了：/comfy again')]
        event.is_notice=True
        with patch.object(self.plugin,'_handle_again_command',AsyncMock()) as again:
            await self.plugin.handle_command(event)
        again.assert_awaited_once()
        event.stop.assert_called_once()

    async def test_tool_waits_for_direct_delivery_and_sends_once(self):
        job=ns['_new_job']('llm','session','cat',512,512,1)
        job.update(status='done',files=['one.png'],urls=['https://test/image'],deliver_directly=True)
        job['done'].set()
        entered,release=asyncio.Event(),asyncio.Event()
        async def send(*args):
            entered.set();await release.wait()
        self.ctx.send_message_chain.side_effect=send
        background=asyncio.create_task(self.plugin._deliver_job(job))
        await entered.wait()
        waiter=asyncio.create_task(self.plugin._wait_for_job(job))
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())
        release.set()
        _,result=await asyncio.gather(background,waiter)
        self.ctx.send_message_chain.assert_awaited_once()
        self.assertIn('不要重复发图',result)

    async def test_ai_delivery_is_files_even_when_markdown_and_buttons_enabled(self):
        job=ns['_new_job']('llm','session','cat',512,512,2)
        job.update(status='done',files=['one.png','two.png'],urls=['https://test/one','https://test/two'])
        await self.plugin._deliver_job(job)
        calls=self.ctx.send_message_chain.call_args_list
        self.assertEqual([c.args[1][0].image for c in calls],['one.png','two.png'])
        self.assertTrue(all(len(c.args[1])==1 and isinstance(c.args[1][0],ImageElement) for c in calls))
        self.assertTrue(job['delivered'])

    async def test_ai_file_retry_does_not_repeat_successful_files(self):
        job=ns['_new_job']('llm','session','cat',512,512,2)
        job.update(status='done',files=['one.png','two.png'])
        self.ctx.send_message_chain.side_effect=[None,RuntimeError('network unavailable'),None]
        await self.plugin._deliver_job(job)
        self.assertFalse(job['delivered'])
        self.assertIn('发送未完成',self.plugin._tool_result_for_done_job(job))
        await self.plugin._deliver_job(job)
        self.assertEqual([c.args[1][0].image for c in self.ctx.send_message_chain.call_args_list],['one.png','two.png','two.png'])
        self.assertTrue(job['delivered']);self.assertNotIn('delivery_error',job)

    async def test_ai_tool_always_requests_direct_file_delivery(self):
        self.plugin.plugin_cfg.update(buttons_enabled=False,send_mode='media')
        with patch.object(self.plugin,'_start_job',return_value={}) as start, \
             patch.object(self.plugin,'_wait_for_job',AsyncMock(return_value='ok')):
            await self.plugin.tool_generate_image(self.event(),'cat')
            self.assertIs(start.call_args.kwargs['deliver_directly'],True)
            await self.plugin.tool_img2img(self.event(),'edit',source_image='https://test/image')
            self.assertIs(start.call_args.args[6],True)

    async def test_ai_execution_skips_progress_and_bed_for_both_generation_modes(self):
        client=SimpleNamespace(resolve_unet=AsyncMock(return_value='model'),list_loras=AsyncMock(return_value=[]),
            submit=AsyncMock(return_value='prompt'),wait_result=AsyncMock(return_value=SimpleNamespace(ok=True,images=['image'])),
            download_image=AsyncMock(return_value=self.png()))
        for mode in ('txt2img','img2img'):
            job=ns['_new_job']('llm','session','cat',512,512,1)
            job.update(mode=mode,denoise=.5,deliver_directly=True)
            self.ctx.send_message_chain.reset_mock()
            with patch.dict(ns,ComfyClient=lambda *a,**kw:client), \
                 patch.object(self.plugin,'_prepare_img2img_job',AsyncMock(return_value=({},None))), \
                 patch.object(self.plugin,'_notify_generation',AsyncMock()) as notice, \
                 patch.object(self.plugin,'_upload_one',AsyncMock()) as upload:
                await self.plugin._execute_job(job)
                notice.assert_not_awaited();upload.assert_not_awaited()
            self.assertEqual(job['status'],'done')
            self.assertTrue(job['delivered'])
            self.assertIsInstance(self.ctx.send_message_chain.call_args.args[1][0],ImageElement)

    async def test_retry_keeps_mask_and_reference_images(self):
        job=ns['_new_job']('command','session','edit',512,384,2)
        job.update(status='done',files=['one.png'],mode='img2img',denoise=0.5,resize_mode='crop',
                   _source_image='https://test/source',_mask_image='https://test/mask',
                   _reference_images=['https://test/ref'])
        start=Mock(return_value={'id':'retry'})
        with patch.object(self.plugin,'_start_job',start):
            await self.plugin._handle_again_command(self.event(),'/comfy','/lora','/model','/upscale')
        self.assertEqual(start.call_args.kwargs['mask_image'],'https://test/mask')
        self.assertEqual(start.call_args.kwargs['reference_images'],['https://test/ref'])
        self.assertNotIn('seed',start.call_args.kwargs)

    async def test_upscale_reports_disabled_or_size_limit(self):
        job=ns['_new_job']('command','session','cat',1024,1024,1)
        job.update(status='done',files=['one.png'])
        with patch.object(self.plugin,'_start_job') as start:
            await self.plugin._handle_upscale_command(self.event(),'/upscale')
            self.assertIn('上限',self.ctx.send_message_chain.call_args.args[1][0].text)
            self.plugin.plugin_cfg['img2img_enabled']=False
            await self.plugin._handle_upscale_command(self.event(),'/upscale')
            self.assertIn('启用图生图',self.ctx.send_message_chain.call_args.args[1][0].text)
        start.assert_not_called()

    def test_buttons_use_callback_actions(self):
        payload=self.plugin._buttons_payload({'urls':['https://test/image']})
        buttons=payload['content']['rows'][0]['buttons']
        self.assertEqual([b['action']['type'] for b in buttons],[1,1])
        self.assertTrue(all(b['action']['permission']['type']==2 for b in buttons))

    async def settings_pick(self, event, action, value='0'):
        owner=self.plugin._settings_owner(event)
        key=settings_module.ProfileStore.profile_id(owner['adapter'],owner['user_id'])
        draft=self.plugin._profile_store.drafts[key]
        await self.plugin._settings_action(event,f"/__comfy_settings {draft['token']} {draft['revision']} {action} {value}")

    async def settings_to_lora(self, event):
        self.plugin.plugin_cfg['model_presets']=[{'name':'Anima','unet':'anima'}]
        await self.plugin._start_settings(event)
        await self.settings_pick(event,'pick','1')
        await self.settings_pick(event,'pick','0')
        with patch.object(self.plugin,'_make_settings_client',return_value=SimpleNamespace(list_loras=AsyncMock(return_value=[f'loras/style{i}.safetensors' for i in range(61)]))):
            await self.settings_pick(event,'pick','2')

    async def test_wizard_pagination_save_and_bare_prompt(self):
        event=self.event()
        with patch.object(self.plugin,'_send_settings_page',AsyncMock()) as page:
            await self.settings_to_lora(event)
            await self.settings_pick(event,'page','1')
            await self.settings_pick(event,'pick','16')
            await self.settings_pick(event,'pick','0.8')
            await self.settings_pick(event,'finish')
            await self.settings_pick(event,'save')
        self.assertGreater(page.await_count,6)
        for call in page.call_args_list:
            payload=call.args[2]
            self.assertLessEqual(len(payload['content']['rows']),5)
            self.assertTrue(all(len(row['buttons'])<=5 for row in payload['content']['rows']))
            for row in payload['content']['rows']:
                for button in row['buttons']:
                    self.assertEqual(button['action']['permission']['specify_user_ids'],['alice'])
                    self.assertLessEqual(len(button['action']['data']),100)
        profile=self.plugin._user_profile(event)
        self.assertEqual(profile['loras'][0]['name'],'loras/style16.safetensors')
        loaded=settings_module.ProfileStore(self.temp.name)
        self.assertEqual(loaded.profiles[profile['id']]['count'],2)
        event.message.chain=[TextElement('/comfy a cat')]
        with patch.object(self.plugin,'_start_job',Mock(return_value={'id':'one'})) as start:
            await self.plugin.handle_command(event)
        self.assertEqual(start.call_args.kwargs['prompt'],'a cat')
        self.assertEqual(start.call_args.kwargs['count'],2)
        self.assertEqual(start.call_args.kwargs['preset']['name'],'Anima')
        self.assertTrue(start.call_args.kwargs['replace_default_loras'])

    async def test_wizard_old_buttons_other_users_and_restart(self):
        event=self.event()
        with patch.object(self.plugin,'_send_settings_page',AsyncMock()):
            await self.plugin._start_settings(event)
            draft=next(iter(self.plugin._profile_store.drafts.values()))
            command=f"/__comfy_settings {draft['token']} 0 pick 0"
            with self.assertRaises(settings_module.SettingsError):
                await self.plugin._settings_action(self.event(user='bob'),command)
            await self.plugin._settings_action(event,command)
            with self.assertRaises(settings_module.SettingsError):
                await self.plugin._settings_action(event,command)
            await self.plugin._start_settings(event)
            with self.assertRaises(settings_module.SettingsError):
                await self.plugin._settings_action(event,command)

    async def test_setup_only_command_does_not_generate(self):
        event=self.event([TextElement('设置')])
        with patch.object(self.plugin,'_start_settings',AsyncMock()) as setup,patch.object(self.plugin,'_start_job') as start:
            await self.plugin.handle_command(event)
        setup.assert_awaited_once();start.assert_not_called();event.stop.assert_called_once()

    async def test_profiles_are_user_and_adapter_specific(self):
        alice=self.event();owner=self.plugin._settings_owner(alice)
        key=settings_module.ProfileStore.profile_id(owner['adapter'],owner['user_id'])
        values=dict(model=None,width=512,height=512,count=1,loras=[])
        await self.plugin._profile_store.save(key,{**owner,**values},0)
        self.assertIsNone(self.plugin._user_profile(self.event(user='bob')))
        same=self.event(sid='session:gm:other')
        self.assertIsNotNone(self.plugin._user_profile(same))
        different=self.event(sid='other:gm:group')
        self.assertIsNone(self.plugin._user_profile(different))

    async def test_admin_update_conflict_and_wizard_cannot_overwrite(self):
        event=self.event();owner=self.plugin._settings_owner(event)
        key=settings_module.ProfileStore.profile_id(owner['adapter'],owner['user_id'])
        saved=await self.plugin._profile_store.save(key,{**owner,'model':None,'width':512,'height':512,'count':1,'loras':[]},0)
        with patch.object(self.plugin,'_send_settings_page',AsyncMock()):
            await self.settings_to_lora(event)
            await self.settings_pick(event,'none')
            changed=await self.plugin._admin_save_profile(key,{**saved,'count':2})
            self.assertEqual(changed['revision'],2)
            with self.assertRaises(settings_module.SettingsError):await self.settings_pick(event,'save')
            with self.assertRaises(settings_module.SettingsError):await self.plugin._admin_save_profile(key,saved)
        self.assertEqual(self.plugin._user_profile(event)['model'],None)

    async def test_user_profile_validation_and_empty_lora_override(self):
        base=dict(model=None,width=512,height=512,count=1,loras=[])
        for change in [dict(count=True),dict(width=99999),dict(loras=[{'name':'a','strength_model':float('nan')}]),dict(loras=[{'name':'a'},{'name':' a '}])]:
            with self.assertRaises(settings_module.SettingsError):self.plugin._validate_profile({**base,**change})
        self.plugin.plugin_cfg['default_loras']='global:1'
        observed=[]
        async def execute(worker,job):observed.append(worker.plugin_cfg['default_loras'])
        with patch.object(Plugin,'_execute_job',execute):
            self.plugin._start_job('test','session','cat',512,512,1,False,tool_loras=[],replace_default_loras=True)
            await asyncio.gather(*self.plugin._tasks)
        self.assertEqual(observed,[''])
        self.assertEqual(self.plugin.plugin_cfg['default_loras'],'global:1')

    async def test_generation_notice_contains_actual_parameters(self):
        job=dict(session='session',preset={'name':'Anima'},width=832,height=1216,count=2,loras=['style@0.8'],mode='txt2img')
        await self.plugin._notify_generation(None,job)
        text=self.ctx.send_message_chain.call_args.args[1][0].text
        for content in ['Anima','832 × 1216','张数：2','style@0.8']:
            self.assertIn(content,text)

    async def test_batch_event_can_open_own_settings(self):
        batch=SimpleNamespace(sid='qq:gm:group',session=SimpleNamespace(adapter_name='qq',sid='qq:gm:group'),
                              messages=[SimpleNamespace(sender=SimpleNamespace(user_id='alice',nickname='Alice'))])
        with patch.object(self.plugin,'_send_settings_page',AsyncMock()) as page:
            result=await self.plugin.tool_user_settings(batch)
        self.assertIn('已发送',result);page.assert_awaited_once()

    async def test_admin_profile_api_updates_and_rejects_stale_revision(self):
        class Body:
            def __init__(self, value):self.raw=__import__('json').dumps(value).encode()
            async def stream(self):yield self.raw
        owner=self.plugin._settings_owner(self.event())
        key=settings_module.ProfileStore.profile_id(owner['adapter'],owner['user_id'])
        saved=await self.plugin._profile_store.save(key,{**owner,'model':None,'width':512,'height':512,'count':1,'loras':[]},0)
        result=await self.plugin.api_update_user_settings(key,Body({**saved,'count':3}))
        self.assertEqual(result['profile']['count'],3)
        self.assertEqual((await self.plugin.api_user_settings())['profiles'][0]['count'],3)
        with self.assertRaises(ns['HTTPException']) as conflict:
            await self.plugin.api_update_user_settings(key,Body(saved))
        self.assertEqual(conflict.exception.status_code,409)
        with self.assertRaises(ns['HTTPException']) as oversized:
            await self.plugin.api_update_user_settings(key,Body({'text':'x'*17000}))
        self.assertEqual(oversized.exception.status_code,413)

    def test_img2img_graph_and_batch(self):
        graph = self.plugin._build_graph('model', 'clip', 'stable_diffusion', 'vae', 'default',
            'cat', 'bad', 512, 512, 3, 0, 20, 4, 'euler', 'simple',
            input_image='kira_studio/input.png', denoise=0.45)
        self.assertEqual(graph['6']['class_type'], 'VAEEncode')
        self.assertEqual(graph['11']['inputs']['amount'], 3)
        self.assertEqual(graph['7']['inputs']['latent_image'], ['11', 0])
        self.assertEqual(graph['7']['inputs']['denoise'], 0.45)
        self.assertEqual(graph['7']['inputs']['seed'], 0)

    def test_mask_graph_preserves_full_output_batch(self):
        graph = self.plugin._build_graph('model', 'clip', 'stable_diffusion', 'vae', 'default',
            'cat', 'bad', 512, 512, 3, 0, 20, 4, 'euler', 'simple',
            input_image='input.png', mask_image='mask.png', denoise=0.6)
        self.assertEqual(graph['6']['class_type'], 'VAEEncodeForInpaint')
        self.assertEqual(graph['17']['inputs']['amount'], 3)
        self.assertEqual(graph['14']['inputs']['destination'], ['17', 0])
        self.assertEqual(graph['9']['inputs']['images'], ['14', 0])

    def test_qwen21_conditioning_and_lora(self):
        graph = self.plugin._build_graph('model', 'clip', 'qwen_image', 'vae', 'default',
            'change sign', 'bad', 512, 512, 1, 1, 20, 4, 'euler', 'simple',
            input_image='input.png', denoise=0.6, img2img_backend='qwen_image_2_1',
            reference_images=['ref.png'], loras=[dict(name='style',strength_model=1,strength_clip=1)])
        self.assertEqual(graph['4']['class_type'], 'TextEncodeQwenImage21')
        self.assertEqual(graph['4']['inputs']['images.image_2'], ['15', 0])
        self.assertEqual(graph['7']['inputs']['negative'], ['4', 1])
        self.assertNotEqual(graph['4']['inputs']['clip'], ['2', 0])

    def test_qwen_edit_plus_reference_wiring(self):
        graph = self.plugin._build_graph('model', 'clip', 'qwen_image', 'vae', 'default',
            'change', 'bad', 512, 512, 1, 1, 20, 4, 'euler', 'simple',
            input_image='input.png', denoise=0.6, img2img_backend='qwen_edit_plus',
            reference_images=['ref1.png','ref2.png'])
        self.assertEqual(graph['4']['class_type'], 'TextEncodeQwenImageEditPlus')
        self.assertEqual(graph['4']['inputs']['image3'], ['16',0])

    async def test_current_image_and_one_based_selection(self):
        event = self.event([ImageElement('https://test/one'),ImageElement('https://test/two')])
        self.assertEqual(await self.plugin._select_source(event,image_index=2),'https://test/two')
        with self.assertRaises(input_module.ImageInputError):
            await self.plugin._select_source(event,image_index=0)

    async def test_quoted_image(self):
        quote = SimpleNamespace(type=SimpleNamespace(value='reply'),chain=[ImageElement('https://test/quoted')])
        self.assertEqual(await self.plugin._select_source(self.event([quote])),'https://test/quoted')

    async def test_recent_images_are_scoped_to_sender_and_session(self):
        self.plugin._remember_images(self.event([ImageElement('https://test/alice')]))
        self.assertEqual(await self.plugin._select_source(self.event()),'https://test/alice')
        for event in [self.event(user='bob'),self.event(sid='other')]:
            with self.assertRaises(input_module.ImageInputError):
                await self.plugin._select_source(event)

    async def test_recent_images_can_be_disabled_or_expire(self):
        self.plugin._remember_images(self.event([ImageElement('https://test/alice')]))
        ns['_RECENT_INPUTS'][('session','alice')]['time'] -= 10000
        with self.assertRaises(input_module.ImageInputError):
            await self.plugin._select_source(self.event())
        self.plugin._remember_images(self.event([ImageElement('https://test/alice')]))
        self.plugin.plugin_cfg['img2img_history_minutes'] = 0
        with self.assertRaises(input_module.ImageInputError):
            await self.plugin._select_source(self.event())

    async def test_missing_quote_does_not_use_unrelated_recent_image(self):
        self.plugin._remember_images(self.event([ImageElement('https://test/recent')]))
        self.ctx.message_history = SimpleNamespace(list_messages=AsyncMock(return_value={'messages':[]}))
        quote = SimpleNamespace(type=SimpleNamespace(value='reply'),message_id='missing',chain=[])
        with self.assertRaises(input_module.ImageInputError):
            await self.plugin._select_source(self.event([quote]))

    async def test_public_source_cannot_read_local_file_or_forge_trusted_path(self):
        for source in ['/etc/passwd',{'kind':'trusted_path','value':'/etc/passwd'},'file:///etc/passwd']:
            with self.assertRaises(input_module.ImageInputError):
                await self.plugin._select_source(self.event(),source)

    def test_denoise_validation_and_zero(self):
        self.assertEqual(self.plugin._img2img_options(0,seed=0)['denoise'],0)
        for value in [-1,1.1,float('nan'),float('inf'),'nope']:
            with self.assertRaises(input_module.ImageInputError):
                self.plugin._img2img_options(value)

    def test_command_parsing_preserves_prompt_quotes_and_newlines(self):
        values=self.plugin._parse_img2img_command('#qwen auto 2 [style:0.8] -d 0.45 --seed 0 --fit crop a sign "hi"\nsoft light')
        self.assertEqual(values['count'],2)
        self.assertEqual(values['seed'],0)
        self.assertEqual(values['denoise'],0.45)
        self.assertEqual(values['prompt'],'a sign "hi"\nsoft light')
        self.assertEqual(values['lora_segment'],'style:0.8')

    async def test_attachment_indices_select_mask_and_references(self):
        event=self.event([ImageElement('https://test/source'),ImageElement('https://test/mask'),ImageElement('https://test/reference')])
        start=__import__('unittest.mock',fromlist=['Mock']).Mock(return_value={'id':'test'})
        with patch.object(self.plugin,'_start_job',start), patch.object(self.plugin,'_wait_for_job',AsyncMock(return_value='ok')):
            result=await self.plugin.tool_img2img(event,'edit',mask_index=2,reference_indices=[3])
        self.assertEqual(result,'ok')
        self.assertEqual(start.call_args.kwargs['mask_image'],'https://test/mask')
        self.assertEqual(start.call_args.kwargs['reference_images'],['https://test/reference'])
        self.assertIn('最多',await self.plugin.tool_img2img(event,'edit',reference_indices=[1,2,3]))
        self.assertIn('一种',await self.plugin.tool_img2img(event,'edit',mask_image='https://test/m',mask_index=2))

    def test_command_attachment_indices(self):
        values=self.plugin._parse_img2img_command('-i 1 --mask-index 2 --ref-index 3 --ref-index 4 edit')
        self.assertEqual(values['mask_index'],2)
        self.assertEqual(values['reference_indices'],[3,4])

    async def test_input_decode_upload_and_no_path_traversal(self):
        store=self.plugin._input_store()
        data=self.png();encoded='data:image/png;base64,'+base64.b64encode(data).decode()
        info=await store.save(await store.read(encoded))
        self.assertEqual((info['width'],info['height']),(160,80))
        self.assertTrue(store.path(info['id']).is_file())
        self.assertEqual(await store.read('input:'+info['id']),store.path(info['id']).read_bytes())
        with self.assertRaises(input_module.ImageInputError):store.path('../outside')
        with self.assertRaises(input_module.ImageInputError):store.decode(b'not an image')

    def test_input_pixel_and_byte_limits(self):
        store=input_module.ImageInputStore(Path(self.temp.name),max_bytes=1024,max_pixels=100)
        with self.assertRaises(input_module.ImageInputError):store.decode(self.png())
        with self.assertRaises(input_module.ImageInputError):store.decode(b'a'*1025)

    def test_resize_and_mask_alignment(self):
        store=self.plugin._input_store()
        source=self.png();mask=self.png(color='white')
        for mode in ['contain','crop','stretch']:
            image,prepared_mask=store.prepare(source,128,128,mode,mask)
            self.assertEqual(PILImage.open(BytesIO(image)).size,(128,128))
            self.assertEqual(PILImage.open(BytesIO(prepared_mask)).size,(128,128))
        padded,_=store.prepare(source,128,128,'contain')
        self.assertEqual(PILImage.open(BytesIO(padded)).getpixel((0,0)),(255,255,255))
        with self.assertRaises(input_module.ImageInputError):store.prepare(source,128,128,'contain',self.png((64,64)))

    def test_job_public_does_not_expose_input_url_credentials(self):
        public=ns['_job_public']({'id':'one','_source_image':'https://test/image?token=secret','done':object()})
        self.assertEqual(public,{'id':'one'})

    async def test_zero_denoise_outputs_original_batch_without_comfy_submission(self):
        raw = self.png()
        source = Path(self.temp.name) / 'source.png'; source.write_bytes(raw)
        job=ns['_new_job']('test','session','keep',0,0,2)
        job.update(mode='img2img',denoise=0,resize_mode='contain',_source_image={'kind':'trusted_path','value':str(source)})
        self.plugin.plugin_cfg['bed_enabled']=False
        client=SimpleNamespace(resolve_unet=AsyncMock(return_value='model'),list_loras=AsyncMock(return_value=[]),submit=AsyncMock())
        with patch.dict(ns,ComfyClient=lambda *a,**kw:client):
            await self.plugin._execute_job(job)
        self.assertEqual(job['status'],'done')
        self.assertEqual(len(job['files']),2)
        self.assertFalse(client.submit.called)
        self.assertEqual((job['width'],job['height']),(160,80))
        self.assertEqual(PILImage.open(job['files'][0]).getpixel((0,0)),(255,0,0))

    async def test_client_upload_uses_returned_subfolder(self):
        import httpx
        def handler(request):
            self.assertEqual(request.url.path,'/upload/image')
            self.assertIn(b'name="image"',request.content)
            return httpx.Response(200,json={'name':'source.png','subfolder':'kira_studio','type':'input'})
        client=client_module.ComfyClient('http://test')
        client._client=lambda:httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.assertEqual(await client.upload_input_image(self.png(),'source.png'),'kira_studio/source.png')


if __name__ == '__main__':
    unittest.main()
