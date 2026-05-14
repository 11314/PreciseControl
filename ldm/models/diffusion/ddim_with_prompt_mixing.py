"""SAMPLING ONLY."""

import os
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from cv2 import dilate
from einops import rearrange, repeat
from diffusers.models.attention_processor import AttnProcessor2_0
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

from ldm.modules.prompt_mixing.attention_based_segmentation2 import Segmentor
from ldm.modules.prompt_mixing.attention_utils import show_cross_attention, aggregate_attention, get_current_cross_attn
# from ldm.modules.prompt_mixing.prompt_to_prompt_controllers import DummyController, AttentionStore
from ldm.modules.prompt_mixing.attention_controller import AttentionControl, AttentionStore, AttentionControlEdit, AttentionReplace, PartEditCrossAttnProcessor, LocalBlend, DummyController




def generate_original_image(model, model_config, args, **kwargs):   # 这一步是生成原始图像。args是实验参数，这个kwargs-把所有额外的“关键字参数”打包成一个字典/新加一个mask_prompt参数
    controller = AttentionStore(args.low_resource)
    ddim_sampler = DDIMSamplerWrapper(model=model, controller=controller, model_config=model_config)
    image, x_t, orig_all_latents, _, x0 = ddim_sampler.sample(args, **kwargs)
    orig_mask = Segmentor(controller, kwargs["image_for_ddim"]['caption'], args.num_segments, args.background_segment_threshold,    # 生成背景mask
                          background_nouns=args.background_nouns).get_background_mask(kwargs["image_for_ddim"]["caption"][-1].split(" ").index("sks")+1)

    average_attention = controller.get_average_attention()
    return image, x_t, orig_all_latents, orig_mask, average_attention, x0


class DDIMSamplerWrapper(object):
    def __init__(self, model, schedule="linear", controller=None, prompt_mixing=None, model_config=None,**kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.controller = controller
        if self.controller is None:
            self.controller = DummyController()
        self.prompt_mixing = prompt_mixing
        self.model_config = model_config
        self.register_attention_control()

    def register_buffer(self, name, attr):  # 并确保 Tensor 在 GPU 上。
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)   # 生成用于采样的时间步
        alphas_cumprod = self.model.alphas_cumprod  # 获取alphas累乘
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device) # tensor转换

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # 计算扩散 q(x_t | x_{t-1}) 或者其他计算。
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim 采样参数
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))    # 计算原始采样
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               args,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               image_for_ddim=None,
               orig_image_for_ddim=None,
               use_prompt_mixing=False,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:    # 如果条件不为空
            if isinstance(conditioning, dict):  # 是否是字典
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]   # 获得batch-size
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # 采样
        C, H, W = shape
        size = (batch_size, C, H, W)    # 构建latents tensor尺寸
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        attr_for_mask = kwargs.pop("attr_for_mask", None)

        image, x0, all_latents, object_mask = self.ddim_sampling(args, conditioning, size,  # 调用ddim采样
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    image_for_ddim=image_for_ddim,
                                                    orig_image_for_ddim=orig_image_for_ddim,
                                                    use_prompt_mixing=use_prompt_mixing,
                                                    **kwargs)
        return image, x_T, all_latents, object_mask, x0

    @torch.no_grad()
    def ddim_sampling(self, args, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,image_for_ddim=None,orig_image_for_ddim=None, use_prompt_mixing=False,
                      post_background = False, orig_all_latents = None, orig_mask = None, mask_prompt=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None: # 是否是初始噪声
            # img = torch.randn(shape, device=device)
            img = torch.randn((4,shape[1],shape[2],shape[3]), device=device)    # 生成随机噪声
            img = img[image_for_ddim["sample_id"]].unsqueeze(0)# 选择sample，应该是表示图像
            if(orig_image_for_ddim is not None):
                img = torch.cat([img,img])
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps: # 如果指定采样步
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1   # 计算timesteps子集
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}    # 初始化中间变量
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)    # 定义timesteps顺序，倒序
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, position=0) # 创建进度条

        self.enbale_attn_controller_changes = True  # 使用attention_control
        object_mask = None
        self.diff_step = 0
        all_latents = []
        # cond_list_for_each_timestep = []

        uc = cond
        prev_cross_attn_iou = np.zeros((1,1,32,32)) # 初始换attention_iou

        if mask is not None: 
            import torch.nn.functional as F
            print("mask shape before:", mask.shape)
            mask = torch.as_tensor(mask, device=img.device).float()
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, size=(64, 64), mode="nearest")
            mask = mask.repeat(img.shape[0], 1, 1, 1)
            print("mask shape after:", mask.shape)

        for i, step in enumerate(iterator): # 进入扩散循环
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # print(f"[DEBUG] step {i}, mask is None? {mask is None}")  #已知，没有mask的传入
            if mask is not None:    # 如果存在mask
                print(f"[DEBUG] Entered mask branch at step {i}")
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?，前向过程
                print("img shape:", img.shape)
                print("img_orig shape:", img_orig.shape)
                img = img_orig * mask + (1. - mask) * img

            self.input_cross_index = 0  # 重置cross-attention index
            self.middle_cross_index = 0
            self.output_cross_index = 0
            # getting condition from mapper
            # TODO: Don't hardcode c and checkwhether passing img to unet is correct (removed hardcoding, passing im is wrong, img is latent img)
            
            if image_for_ddim is not None:  # 如果有图像输入，获取图像信息
                two_ids = image_for_ddim.get('two_ids', False)
                face_img = image_for_ddim['face_img']
                img_ori = image_for_ddim['image_ori']
                aligned_faces = image_for_ddim['aligned_faces']
                c = image_for_ddim['caption']
                if(use_prompt_mixing):
                    steps_for_prompt_mixing = image_for_ddim['steps_for_prompt_mixing']

            h_space = {'h_space_feat': None, 't':ts}    # 构建hidden—space

            other_cond = None
            if(use_prompt_mixing and i < steps_for_prompt_mixing):  # 如果在使用嵌入混合的时间区间中
                prompt_mixing_text = image_for_ddim['prompt_mixing_prompt']
                # print("prompt mixing")
                c = prompt_mixing_text
                cond = self.model.get_learned_conditioning(c, face_img=face_img, image_ori=img_ori,aligned_faces=aligned_faces,h_space=h_space) # 将条件进行编码

                
            else:
                # c = c*face_img.shape[0]
                cond = self.model.get_learned_conditioning(c, face_img=face_img, image_ori=img_ori,aligned_faces=aligned_faces,h_space=h_space)
                
            # getting other context
            if(orig_image_for_ddim is not None):    # 如果存在另外一组条件
                other_cond = self.model.get_learned_conditioning(c, face_img, image_ori=orig_image_for_ddim['image_ori'],aligned_faces=aligned_faces,
                                                                    h_space=h_space)
                other_cond = torch.cat([cond, other_cond])  # 拼接条件

            outs = self.p_sample_ddim(args, img, cond, ts, index=index, use_original_steps=ddim_use_original_steps, # 然后进入ddim step，跳转到p_sample_ddim函数
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning, other_cond=other_cond, orig_image_for_ddim=orig_image_for_ddim)
            

            img = outs[0]   # 更新latents
            img = self.controller.step_callback(img)    # attention controller 回调

            # 在每个时间步保存交叉注意力
            save_cross_attn = False
            if(two_ids and save_cross_attn):
                save_dir = "./cross_attn_at_each_timestep"
                import os
                from torchvision import transforms
                import matplotlib.pyplot as plt
                os.makedirs(save_dir, exist_ok=True)
                attn = get_current_cross_attn(self.controller, res=16, from_where=("output", "input"), prompts=c,
                                                    is_cross=True, select=len(c) - 1)
                token_attn = attn[:,:,3].cpu().numpy()
                curr_noun_map = token_attn.repeat(2, axis=0).repeat(2, axis=1)
                normalised_noun_map1 = (curr_noun_map - np.abs(curr_noun_map.min())) / curr_noun_map.max()
                plt.imshow(normalised_noun_map1)
                plt.savefig(f"{save_dir}/cross_attn1_{i}.png")

                token_attn = attn[:,:,7].cpu().numpy()
                curr_noun_map = token_attn.repeat(2, axis=0).repeat(2, axis=1)
                normalised_noun_map2 = (curr_noun_map - np.abs(curr_noun_map.min())) / curr_noun_map.max()
                plt.imshow(normalised_noun_map2)
                plt.savefig(f"{save_dir}/cross_attn2_{i}.png")

                cross_attn_iou = np.sum(normalised_noun_map1 * normalised_noun_map2) / np.sum(normalised_noun_map1 + normalised_noun_map2)
                cross_attn_iou = np.ones((1,1,32,32)) * cross_attn_iou
                grad_attn_iou = cross_attn_iou - prev_cross_attn_iou
                
                prev_cross_attn_iou = cross_attn_iou
                iou_alpha = 0.01


            pm_and_matching_args = {}
            # object_mask = None
            prompt = c
            
            if post_background and (self.diff_step == args.background_blend_timestep):  # 判断是否进行背景融合，默认不融合
                object_mask = Segmentor(self.controller,
                                        prompt,
                                        args.num_segments,
                                        args.background_segment_threshold,
                                        background_nouns=args.background_nouns)\
                    .get_background_mask(orig_image_for_ddim["caption"][-1].split(" ").index("sks")+1)
                self.enbale_attn_controller_changes = False
                pm_mask = object_mask.astype(np.bool8) + orig_mask.astype(np.bool8)
                pm_mask = torch.from_numpy(pm_mask).float().cuda()
                shape = (1, 1, pm_mask.shape[0], pm_mask.shape[1])
                pm_mask = torch.nn.Upsample(size=(64, 64), mode='nearest')(pm_mask.view(shape))
                # plt.imshow(pm_mask.cpu().numpy()[0, 0])
                # plt.savefig(f"./pm_masks/pm_mask_{i}.png")
                mask_eroded = dilate(pm_mask.cpu().numpy()[0, 0], np.ones((3, 3), np.uint8), iterations=1)
                pm_mask = torch.from_numpy(mask_eroded).float().cuda().view(1, 1, 64, 64)
                img = pm_mask * img + (1 - pm_mask) * orig_all_latents[self.diff_step]


            all_latents.append(img)
            self.diff_step += 1

            img, pred_x0 = outs

            
            del h_space, face_img, img_ori, aligned_faces, ts, c, cond, outs, pm_and_matching_args # 删除临时变量, pred_x0
            if(two_ids):
                del attn, token_attn, curr_noun_map, normalised_noun_map1, normalised_noun_map2, cross_attn_iou

        image = self.latent2image(all_latents[-1])  # 最终 latent → image

        return image, pred_x0, all_latents, object_mask

def create_controller(
    prompts: List[str],
    cross_attention_kwargs: Dict,
    num_inference_steps: int,
    tokenizer,
    device: torch.device,
    attn_res: Tuple[int, int],
    extra_kwargs: dict,
) -> AttentionControl:
    edit_type = cross_attention_kwargs.get("edit_type", "replace")  # 从字典里取编辑类型
    local_blend_words = cross_attention_kwargs.get("local_blend_words") # 是否局部编辑
    equalizer_words = cross_attention_kwargs.get("equalizer_words") # 
    equalizer_strengths = cross_attention_kwargs.get("equalizer_strengths")
    n_cross_replace = cross_attention_kwargs.get("n_cross_replace", 0.4)    # 注意力替换比例
    n_self_replace = cross_attention_kwargs.get("n_self_replace", 0.4)
    print("local_blend_words is ",local_blend_words)
    print("cross_attention_kwargs is ", cross_attention_kwargs)
    print ("Whatever use LB?")


    # 局部替换，使用的是这个分支
    if edit_type == "replace" and local_blend_words is not None:
        print("yes")
        lb = LocalBlend(
            prompts,
            local_blend_words,
            tokenizer=tokenizer,
            device=device,
            attn_res=attn_res,
        )
        return AttentionReplace(
            prompts,
            num_inference_steps,
            n_cross_replace,
            n_self_replace,
            lb,
            tokenizer=tokenizer,
            device=device,
            attn_res=attn_res,
            extra_kwargs=extra_kwargs,
        )
    
    @torch.no_grad()
    # 在 UNet 中注册并配置注意力（attention）控制器，允许 PartEdit 对扩散模型的 cross-attention 层进行控制。
    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        self.attn_names = {}  # Name => Idx
        for name in self.unet.attn_processors:  # 这里开始循环遍历 UNet 中的所有 attention 层。
            (None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim)    # 跳过指定的层
            if name.startswith("mid_block"):    # 判断当前层是否属于 UNet 的中间部分（mid_block）。 
                self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):  # 判断当前层是否属于 UNet 的上采样部分（up_blocks）。
                block_id = int(name[len("up_blocks.")]) # 如果是，那就提取块的编号，
                list(reversed(self.unet.config.block_out_channels))[block_id]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):    # 同理
                block_id = int(name[len("down_blocks.")])
                self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            attn_procs[name] = PartEditCrossAttnProcessor(controller=controller, place_in_unet=place_in_unet)   # 创建 PartEditCrossAttnProcessor，下面有具体定义。
            # print(f'{cross_att_count}=>{name}')
            cross_att_count += 1    # 每添加一个 cross-attention 层，cross_att_count 就加 1，统计需要控制的层数。

        self.unet.set_attn_processor(attn_procs)    # 将 attn_procs 传给 self.unet。
        controller.num_att_layers = cross_att_count # 更新 controller 对象中的 num_att_layers，标记有多少个 attention 层需要控制。

    def unregister_attention_control(self):
        # if pytorch >= 2.0
        self.unet.set_attn_processor(AttnProcessor2_0())    # 将 UNet 的 attention 处理器恢复为标准的 UNet cross-attention 处理器。
        if hasattr(self, "controller") and self.controller is not None: # 检查 self 是否具有 controller 属性，且 controller 不为 None。
            if hasattr(self.controller, "last_otsu"):   # 如果 controller 具有 last_otsu 属性，就将 最后一个 OTSU 阈值保存到 self.last_otsu_value 中。
                self.last_otsu_value = self.controller.last_otsu[-1]
            del self.controller # 删除 controller 对象，释放内存。
            # self.controller.allow_edit_control = False
    
    @torch.no_grad()
    def latent2image(self, latents):
        x_samples_ddim = self.model.decode_first_stage(latents)
        image = (x_samples_ddim / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def p_sample_ddim(self, args, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, other_cond=None, orig_image_for_ddim=None):
        b, *_, device = *x.shape, x.device  # 获取 batch 和设备

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:    # 如果不使用CFG
            self.uncond_pred = True
            c = (c, None)
            e_t = self.model.apply_model(x, t, c)   # UNet预测噪声,Unet farward
        else:   # 如果使用CFG
            n = 2 if orig_image_for_ddim is None else 4 # 判断输入数量
            self.uncond_pred = False    # 标记非unconditional
            x_in = torch.cat([x] * 2)   # 复制latents
            t_in = torch.cat([t] * n)
            if(n==4):
                c_in = torch.cat([unconditional_conditioning, unconditional_conditioning, c, c])    # 拼接条件
            else:
                c_in = torch.cat([unconditional_conditioning, c])

            c_in = (c_in, other_cond)   
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2) # UNet预测噪声  
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)    # CFG公式

        if score_corrector is not None: # 如果使用 score correction 方法。
            assert self.model.parameterization == "eps" # 检查参数化
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)    # 修改score

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas  # 选择alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev   # 上一步 alpha
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # 选择与当前考虑的时间步长相对应的参数
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # 当前对x_0的预测
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # 指向x_t的方向
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
    
    @torch.no_grad()
    def init_latent(self, latent, batch_size):
        if latent is None:
            latent = torch.randn(
                (1, self.model.in_channels, self.height // 8, self.width // 8),
                generator=self.generator, device=self.device
            )
        latents = latent.expand(batch_size,  self.model.in_channels, self.height // 8, self.width // 8).to(self.device)
        return latent, latents


    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec
