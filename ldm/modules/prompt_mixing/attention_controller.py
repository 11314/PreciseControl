
import abc
import numpy as np
import einops
import torch
from enum import Enum
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import make_grid
from safetensors.torch import load_file
from torchvision.transforms import ToPILImage, ToTensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)


logger = logging.get_logger(__name__)

# 它定义了 PartEdit 中“如何把注意力图（attention map）变成可用掩码”的策略集合。,阈值策略
class Binarization(Enum):
    """Controls the binarization of attn maps
    in case of use_otsu lower_binarize and upper_binarizer are multilpiers of otsu threshold

    args:
        strategy: str: name of the strategy
        enabled: bool: if binarization is enabled
        lower_binarize: float: lower threshold for binarization
        upper_binarize: float: upper threshold for binarization
        use_otsu: bool: if otsu is used for binarization
    """

    P2P = "p2p", False, 0.5, 0.5, False  # Baseline
    PROVIDED_MASK = "mask", True, 0.5, 0.5, False
    BINARY_0_5 = "binary_0.5", True, 0.5, 0.5, False
    BINARY_OTSU = "binary_otsu", True, 1.0, 1.0, True
    PARTEDIT = "partedit", True, 0.5, 1.5, True
    DISABLED = "disabled", False, 0.5, 0.5, False

    # 让每一个枚举值，不只是一个字符串，而是一个“带属性的对象”
    def __new__(
        cls,
        strategy: str,
        enabled: bool,
        lower_binarize: float,
        upper_binarize: float,
        use_otsu: bool,
    ) -> "Binarization":
        obj = object.__new__(cls)
        obj._value_ = strategy
        obj.enabled = enabled   # 这里是给创建的对象绑定额外属性
        obj.lower_binarize = lower_binarize
        obj.upper_binarize = upper_binarize
        obj.use_otsu = use_otsu
        assert isinstance(obj.enabled, bool), "enabled should be of type bool"  # 类型检查，防止策略定义错误
        assert isinstance(obj.lower_binarize, float), "lower_binarize should be of type float"
        assert isinstance(obj.upper_binarize, float), "upper_binarize should be of type float"
        assert isinstance(obj.use_otsu, bool), "use_otsu should be of type bool"
        return obj

    # 重写等号
    def __eq__(self, other: Optional[Union["Binarization", str]] = None) -> bool:
        if not other:
            return False
        if isinstance(other, Binarization):
            return self.value.lower() == other.value.lower()
        if isinstance(other, str):
            return self.value.lower() == other.lower()

    @staticmethod
    def available_strategies() -> List[str]:    # 返回所有可用策略名称
        return [strategy.name for strategy in Binarization]

    def __str__(self) -> str:   # 打印使用
        return f"Binarization: {self.name} (Enabled: {self.enabled} Lower: {self.lower_binarize} Upper: {self.upper_binarize} Otsu: {self.use_otsu})"

    @staticmethod   # 把用户输入的字符串参数，转换成真正的 Binarization 策略对象
    def from_string(
        strategy: str,
        enabled: Optional[bool] = None,
        lower_binarize: Optional[bool] = None,
        upper_binarize: Optional[float] = None,
        use_otsu: Optional[bool] = None,
    ) -> "Binarization":
        strategy = strategy.strip().lower() # 修改输入
        for _strategy in Binarization:  # 遍历所有策略
            if _strategy.name.lower() == strategy:  # 匹配
                if enabled is not None: # 允许修改策略参数
                    _strategy.enabled = enabled
                if lower_binarize is not None:
                    _strategy.lower_binarize = lower_binarize
                if upper_binarize is not None:
                    _strategy.upper_binarize = upper_binarize
                if use_otsu is not None:
                    _strategy.use_otsu = use_otsu
                return _strategy
        raise ValueError(f"binarization_strategy={strategy} not recognized")
    
# 当 PartEdit 往 prompt embeddings 中插入额外 token embedding 后，其余 token 位应该用什么方式填充，以及是否对填充结果做归一化 / 缩放。
class PaddingStrategy(Enum):
    # 默认
    BG = "BG", False, False
    # 其他添加只是为了实验
    context = "context", False, False
    EOS = "EoS", False, False
    ZERO = "zero", False, False
    SOT_E = "SoT_E", False, False

    def __new__(cls, strategy: str, norm: bool, scale: bool) -> "PaddingStrategy":
        obj = object.__new__(cls)
        obj._value_ = strategy
        obj.norm = norm
        obj.scale = scale
        return obj

    # 按value比较
    def __eq__(self, other: Optional[Union["PaddingStrategy", str]] = None) -> bool:
        if not other:
            return False
        if isinstance(other, PaddingStrategy):
            return self.value.lower() == other.value.lower()
        if isinstance(other, str):
            return self.value.lower() == other.lower()

    @staticmethod
    def available_strategies() -> List[str]:
        return [strategy.name for strategy in PaddingStrategy]

    def __str__(self) -> str:
        return f"PaddStrategy: {self.name} Norm: {self.norm} Scale: {self.scale}"

    @staticmethod
    def from_string(strategy_str, norm: Optional[bool] = False, scale: Optional[bool] = False) -> "PaddingStrategy":
        for strategy in PaddingStrategy:
            if strategy.name.lower() == strategy_str.lower():
                if norm is not None:
                    strategy.norm = norm
                if scale is not None:
                    strategy.scale = scale
                return strategy
        raise ValueError(f"padd_strategy={strategy} not recognized")
    
# 取决于训练时使用的层
LAYERS_TO_USE = [
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    0,
    1,
    2,
    3,
]  # noqa: E501

# 是一个“带默认值 + 自动解析 + 自动规范化”的配置容器，用来承载并规范 PartEdit 所有 extra_kwargs 参数。
class DotDictExtra(dict):
    """
    dot.notation access to dictionary attributes
    Holds default values for the extra_kwargs
    """
    # 点访问
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    _layers_to_use = LAYERS_TO_USE  # 训练参数，不直接暴露给用户，类级“隐藏配置参数”
    _enable_non_agg_storing = False  # 是否保存未聚合 attention，但非常占显存！~35GB无卸载14GB带卸载
    _cpu_offload = False  # 是否把 attention store 放到 CPU，降低VRAM，但大幅减速，隐藏
    _default = {    # 默认参数表
        "th_strategy": Binarization.PARTEDIT,
        "pad_strategy": PaddingStrategy.BG,
        "omega": 1.5,  # 值应该在0.25到2.0之间  
        "use_agg_store": False,
        "edit_mask": None,
        "edit_steps": 50, # 在这个时间步结束
        "start_editing_at": 0,  # 推荐，但是会在想要改变的时候暴露
        "use_layer_subset_idx": None,  # 以防我们想要使用特定的层, NOTE: 顺序不与Unet层对齐
        "add_extra_step": False,
        "batch_indx": -1,  # 最后一个
        "blend_layers": None,
        "force_cross_attn": False,  # 强迫交叉注意力到图
        # 优化部分
        "VRAM_low": True,  # 默认情况下保持开启状态，除非会导致错误
        "grounding": None,
    }
    _default_explanations = {   # 参数解释说明表（用于文档）
        "th_strategy": "Binarization strategy for attention maps",
        "pad_strategy": "Padding strategy for the added tokens",
        "omega": "Omega value for the PartEdit",
        "use_agg_store": "If the attention maps should be aggregated",
        "add_extra_step": "If extra 0 step should be added to the diffusion process",
        "edit_mask": "Mask for the edit when using ProvidedMask strategy",
        "edit_steps": "Number of edit steps",
        "start_editing_at": "Step at which the edit should start",
        "use_layer_subset_idx": "Sublayers to use, recommended 0-8 if really needed to use some",
        "VRAM_low": "Recommended to not change",
        "force_cross_attn": "Force cross attention to use OPT token maps",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   # 调用 dict 初始化
        for key, value in self._default.items():    # 注入默认参数
            if key not in self:
                self[key] = value

        print(">>> init DotDictExtra")
        # 对二值化，填充策略进行了额外的更改
        if isinstance(self["th_strategy"], str):    # 把字符串策略转成枚举
            self["th_strategy"] = Binarization.from_string(self["th_strategy"])
        if isinstance(self["pad_strategy"], str):
            self["pad_strategy"] = PaddingStrategy.from_string(self["pad_strategy"])
        self["edit_steps"] = self["edit_steps"] + self["add_extra_step"]    # 时间步对齐

        if self.edit_mask is not None :
            if isinstance(self.edit_mask, str):
                # 从PIL or torch/safetensors中加载  
                if self.edit_mask.endswith(".safetensors"):
                    self.edit_mask = load_file(self.edit_mask)["edit_mask"]
                elif self.edit_mask.endswith(".pt"):
                    self.edit_mask = torch.load(self.edit_mask)["edit_mask"]
                else:
                    self.edit_mask = Image.open(self.edit_mask)
            if isinstance(self.edit_mask, Image.Image):
                self.edit_mask = ToTensor()(self.edit_mask.convert("L"))
            elif isinstance(self.edit_mask, np.ndarray):
                self.edit_mask = torch.from_numpy(self.edit_mask).unsqueeze(0)
            if self.edit_mask.ndim == 2:
                self.edit_mask = self.edit_mask[None, None, ...]
            elif self.edit_mask.ndim == 3:
                self.edit_mask = self.edit_mask[None, ...]
            
            if self.edit_mask.max() > 1.0:# 归一化
                self.edit_mask = self.edit_mask / self.edit_mask.max()
        if self.grounding is not None: # same as above, but slightly different function
            if isinstance(self.grounding, Image.Image):
                self.grounding = ToTensor()(self.grounding.convert("L"))
            elif isinstance(self.grounding, np.ndarray):
                self.grounding = torch.from_numpy(self.grounding).unsqueeze(0)
            if self.grounding.ndim == 2:
                self.grounding = self.grounding[None, None, ...]
            elif self.grounding.ndim == 3:
                self.grounding = self.grounding[None, ...]
            if self.grounding.max() > 1.0:  
                self.grounding = self.grounding / self.grounding.max()

        assert isinstance(self.th_strategy, Binarization), "th_strategy should be of type Binarization"
        assert isinstance(self.pad_strategy, PaddingStrategy), "pad_strategy should be of type PaddingStrategy"

    def th_from_str(self, strategy: str):
        return Binarization.from_string(strategy)

    @staticmethod   # 返回参数说明字符串
    def explain() -> str:
        """Returns a string with all the explanations of the parameters"""
        return "\n".join(
            [
                f"{key}: {DotDictExtra._default_explanations[key]}"
                for key in DotDictExtra._default
                if DotDictExtra._default_explanations.get(key, "Recommended to not change") != "Recommended to not change"
            ]
        )

# otsu阈值，这里定义otsu阈值
# 根据图像（或 attention map）的直方图，自适应计算一个能最大化前景/背景类间方差的全局阈值。
@torch.no_grad()
def threshold_otsu(image: torch.Tensor = None, nbins=256, hist=None):
    """Return threshold value based on Otsu's method using PyTorch.
    This is a reimplementation from scikit-image
    https://github.com/scikit-image/scikit-image/blob/b76ff13478a5123e4d8b422586aaa54c791f2604/skimage/filters/thresholding.py#L336

    Args:
    image: torch.Tensor
        Grayscale input image.
    nbins: int
        Number of bins used to calculate histogram.
    hist: torch.Tensor or tuple
        Histogram of the input image. If None, it will be calculated using the input image.
    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    """
    if image is not None and image.dim() > 2 and image.shape[-1] in (3, 4): # 检查是否是RGB类型的图片
        raise ValueError(f"threshold_otsu is expected to work correctly only for " f"grayscale images; image shape {image.shape} looks like " f"that of an RGB image.")
    # 在设备上将bbin转换为张量，将 nbins 放到同一设备
    nbins = torch.tensor(nbins, device=image.device)

    # 检查图像是否常量图像；如果不是，则返回该值
    if image is not None:
        first_pixel = image.view(-1)[0]
        if torch.all(image == first_pixel):
            return first_pixel.item()
    # counts：每个 bin 中的像素数量。bin_centers：每个 bin 的中心值
    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    # 所有可能阈值的类概率
    weight1 = torch.cumsum(counts, dim=0)
    weight2 = torch.cumsum(counts.flip(dims=[0]), dim=0).flip(dims=[0])
    # 所有可能阈值的类均值
    mean1 = torch.cumsum(counts * bin_centers, dim=0) / weight1
    mean2 = (torch.cumsum((counts * bin_centers).flip(dims=[0]), dim=0).flip(dims=[0])) / weight2

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2 # 计算类间方差

    idx = torch.argmax(variance12)  # 选择最大类间方差的阈值
    threshold = bin_centers[idx]    # 将对应的灰度值就是

    return threshold.item()

# 校验并构造灰度直方图表示，确保 Otsu 阈值计算阶段始终拿到合法、规范的 counts 和 bin_centers。
def _validate_image_histogram(image: torch.Tensor, hist, nbins):
    """Helper function to validate and compute histogram if necessary."""
    if hist is not None:    # 判断是否传入了hist
        if isinstance(hist, tuple) and len(hist) == 2:  # 如果hist 是 (counts, bin_centers) 形式
            counts, bin_centers = hist
            if not (isinstance(counts, torch.Tensor) and isinstance(bin_centers, torch.Tensor)):
                counts = torch.tensor(counts)
                bin_centers = torch.tensor(bin_centers)
        else:
            counts = torch.tensor(hist)
            bin_centers = torch.linspace(0, 1, len(counts))
    else:# 如果没有传入hist，从image计算
        if image is None:
            raise ValueError("Either image or hist must be provided.")
        image = image.to(torch.float32)
        counts, bin_edges = histogram(image, nbins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return counts, bin_centers

# 用于在GPU上统计张量的直方图，返回每个bin的计数和bin边界，作为 Otsu 阈值等算法的输入。
def histogram(xs: torch.Tensor, bins):
    # Like torch.histogram, but works with cuda
    # https://github.com/pytorch/pytorch/issues/69519#issuecomment-1183866843
    min, max = xs.min(), xs.max()
    counts = torch.histc(xs, bins, min=min, max=max).to(xs.device)  # 使用 torch.histc 统计直方图
    boundaries = torch.linspace(min, max, bins + 1, device=xs.device)
    return counts, boundaries


# 把 attention 从“token/flatten 形式”变成“空间特征图”，做插值，再还原回原来的 attention 形状。
def pack_interpolate_unpack(att, size, interpolation_mode, unwrap_last_dim=True, rewrap=False):
    has_last_dim = att.shape[-1] in [77, 1] # 77是CLIPtoken数，判断最后一维是不是“token 维”
    _last_dim = att.shape[-1]   # 记录 token 维大小
    if unwrap_last_dim: # 是否展开为二维空间，
        if has_last_dim:    # 有token维
            sq = int(att.shape[-2] ** 0.5)  # attention 是 flatten 的 HW。反推空间尺寸
            att = att.reshape(att.shape[0], sq, sq, -1).permute(0, 3, 1, 2)  # B x HW x D => B x D x H x W
        else:   # 没有token维，也同样反推。
            sq = int(att.shape[-1] ** 0.5)
            att = att.reshape(*att.shape[:-1], sq, sq)  # B x H x W
    att = att.unsqueeze(-3)  # 添加通道尺寸
    if att.shape[-2:] != size:  # 判断是否需要resize
        att, ps = einops.pack(att, "* c h w")   # 打包成任意 batch 维
        att = F.interpolate(    # 对 所有 token / channel 的 attention map，同时resize到目标空间。
            att,
            size=size,
            mode=interpolation_mode,
        )
        att = torch.stack(einops.unpack(att, ps, "* c h w"))    # 恢复原batch维
    if rewrap:  # 是否重新 wrap 回 token 形式
        if has_last_dim:
            att = att.reshape(att.shape[0], -1, att.shape[-1] * att.shape[-1], _last_dim)
        else:
            att = att.reshape(att.shape[0], -1, att.shape[-1] * att.shape[-1])




def min_max_norm(a, _min=None, _max=None, eps=1e-6):
    _max = a.max() if _max is None else _max
    _min = a.min() if _min is None else _min
    return (a - _min) / (_max - _min + eps)

# 在 latent 空间里，用 cross-attention 生成的空间 mask，只在“与指定词语相关的区域”应用编辑，其余区域保持原样。
class LocalBlend:
    def __call__(self, x_t, attention_store):   # 在 每个 diffusion step 中，用 attention 生成 mask 并应用到 latent
        # print(">>> In LocalBlend")
        # 请注意，此代码在潜在层上工作！
        k = 1
        maps = [m for m in attention_store["down_cross"] + attention_store["mid_cross"] + attention_store["up_cross"] if m.shape[1] == self.attn_res[0] * self.attn_res[1]]
        maps = [    # 重塑
            item.reshape(
                self.alpha_layers.shape[0],
                -1,
                1,
                self.attn_res[0],
                self.attn_res[1],
                self.max_num_words,
            )
            for item in maps
        ]
        maps = torch.cat(maps, dim=1)   # 拼接不同层的 attention
        maps = (maps * self.alpha_layers).sum(-1).mean(1)   # 只保留“被编辑词语”的 attention
        # 因为alpha_layers除了我们编辑的部分外都是0，所以product将除我们修改的部分外的所有内容归零。然后，将原始值和我们编辑的值相加。我们取dim=1的平均值，这是层数.
        mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))   # 局部膨胀（平滑 mask）
        mask = F.interpolate(mask, size=(x_t.shape[2:]))    # 插值到 latent 分辨率
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]    # 归一化
        mask = mask.gt(self.threshold)  # 二值化

        mask = mask[:1] + mask[1:]  # source + target mask 合并
        mask = mask.to(torch.float16)
        if mask.shape[0] < x_t.shape[0]:  # PartEdit 的 batch 对齐补丁
            # 再次连接最后一个掩码
            mask = torch.cat([mask, mask[-1:]], dim=0)

        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        # 代码对原始图像和每个生成的图像之间的图像差异应用掩码，有效地只保留所需的区域
        return x_t

    # NOTE(Alex): 复制到LocalBlend
    def __init__(
        self,
        prompts: List[str],
        words: List[List[str]],
        tokenizer,
        device,
        threshold=0.3,
        attn_res=None,
    ):
        print(">>> init LocalBlend")
        self.max_num_words = 77 # token固定长度=77
        self.attn_res = attn_res

        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, self.max_num_words)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):  # 标记需要编辑的 token
            if isinstance(words_, str):
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)  # 一个单热向量，其中1是我们修改的单词（源和目标）
        self.threshold = threshold


# 定义了一个注意力控制抽象基类，对 attention map 进行干预
class AttentionControl(abc.ABC):
    def step_callback(self, x_t):   # 扩散后的回调
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):    # 前多少层 attention 是 无条件分支
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str, store: bool = True):    # 控制注意力的地方
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, store: bool = True):
        # print(">>> In AttentionControl")
        if self.cur_att_layer >= self.num_uncond_att_layers:    # 是否当前层 >= uncond层数
            h = attn.shape[0]   # 获取batch size
            attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet, store)   # 只对 conditional 部分做编辑
        self.cur_att_layer += 1 # 当前层数+1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:  # 是否当前 step 的所有 attention 层已经走完
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):    # 初始化
        self.cur_step = 0
        self.cur_att_layer = 0
        self.allow_edit_control = True

    def __init__(self, attn_res=None, extra_kwargs: DotDictExtra = None):   # 初始化参数
        # PartEdit
        print(">>> init AttentionControl")
        self.extra_kwargs = extra_kwargs
        self.index_inside_batch = extra_kwargs.get("index_inside_batch", 1) # 默认值是我们之前设置的1!
        if not isinstance(self.index_inside_batch, list):
            self.index_inside_batch = [self.index_inside_batch]
        self.layers_to_use = extra_kwargs.get("_layers_to_use", LAYERS_TO_USE)  # Training parameter, not exposed directly
        # Params
        self.th_strategy: Binarization = extra_kwargs.get("th_strategy", Binarization.P2P)
        self.pad_strategy: PaddingStrategy = extra_kwargs.get("pad_strategy", PaddingStrategy.BG)
        self.omega: float = extra_kwargs.get("omega", 1.0)
        self.use_agg_store: bool = extra_kwargs.get("use_agg_store", False)
        self.edit_mask: Optional[torch.Tensor] = extra_kwargs.get("edit_mask", None)  # edit_mask_t
        self.edit_steps: int = extra_kwargs.get("edit_steps", 50) # NOTE(Alex): This is the end step, IMPORTANT
        self.blend_layers: Optional[List] = None
        self.start_editing_at: int = extra_kwargs.get("start_editing_at", 0)
        self.use_layer_subset_idx: Optional[list[int]] = extra_kwargs.get("use_layer_subset_idx", None)
        self.batch_indx: int = extra_kwargs.get("batch_indx", 0)
        self.VRAM_low: bool = extra_kwargs.get("VRAM_low", False)
        self.allow_edit_control = True
        # Old
        self.cur_step: int = 0
        self.num_att_layers: int = -1
        self.cur_att_layer: int = 0
        self.attn_res: int = attn_res

    def get_maps_agg(self, resized_res, device):    # 用于返回聚合 attention map，默认未实现
        return None

    def _editing_allowed(self): # 当前时间步是否允许编辑
        return self.allow_edit_control  # TODO(Alex): Maybe make this only param, instead of unregister attn control?

# 保存Attention，生成空间mask
class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():  # 初始化容器，存储上中下采样过程中的自注意力和交叉注意力，优化的注意力，和背景注意力
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
            "opt_cross": [],
            "opt_bg_cross": [],
        }

    def maybe_offload(self, attn_device, attn_dtype):   # 显存优化
        if self.extra_kwargs.get("_cpu_offload", False):    # 如果开启，把 attention 从 GPU → CPU，并转成 float32（稳定）
            attn_device, attn_dtype = torch.device("cpu"), torch.float32
        return attn_device, attn_dtype

    def forward(self, attn, is_cross: bool, place_in_unet: str, store: bool = True):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"  # 创建key
        _device, _dtype = self.maybe_offload(attn.device, attn.dtype)   # 存储设备
        if store and self.batch_indx is not None and is_cross:  # 提取并存储attention
            # We always store for our method
            _dim = attn.shape[0] // self.num_prompt # 计算每个prompt对应的chunk，↓选取第batch_indx个样本，取所有空间位置，选指定token，聚合，转设备
            _val = attn[_dim * self.batch_indx : _dim * (self.batch_indx + 1), ..., self.index_inside_batch].sum(0, keepdim=True).to(_device, _dtype)
            if _val.shape[-1] != 1: # 归一化+token聚合
                # min_max each -1 seperately
                _max = _val.max()
                for i in range(_val.shape[-1]): # 对每个token单独归一化
                    _val[..., i] = min_max_norm(_val[..., i], _max=_max)
                _val = _val.sum(-1, keepdim=True)   # 跨token求和
            self.step_store["opt_cross"].append(_val)   # 存入opt_cross
        if self.extra_kwargs.get("_enable_non_agg_storing", False) and store:   # 可选存原始attention
            _attn = attn.clone().detach().to(_device, _dtype, non_blocking=True)
            if attn.shape[1] <= 32**2:  # avoid memory overhead
                self.step_store[key].append(_attn)
        return attn

    def offload_stores(self, device):   # 显存释放
        """Created for low VRAM usage, where we want to do this before Decoder"""
        for key in self.step_store:
            self.step_store[key] = [a.to(device) for a in self.step_store[key]]
        for key in self.attention_store:
            self.attention_store[key] = [a.to(device) for a in self.attention_store[key]]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def calculate_mask_t_res(self, use_step_store: bool = False):   # 将注意力转换成mask
        mask_t_res = aggregate_attention(   # 输入opt_cross，多层注意力，输出[H,W]
            self,
            res=1024,
            from_where=["opt"],
            batch_size=1,
            is_cross=True,
            upsample_everything=False,
            return_all_layers=False, # Removed sum in this function
            use_same_layers_as_train=True,
            train_layers=self.layers_to_use,
            use_step_store=use_step_store,
            use_layer_subset_idx=self.use_layer_subset_idx,
        )[..., 0]   # 取单通道

        strategy: Binarization = self.th_strategy   # 获取阈值

        mask_t_res = min_max_norm(mask_t_res)   # 归一化

        upper_threshold = strategy.upper_binarize
        lower_threshold = strategy.lower_binarize
        use_otsu = strategy.use_otsu
        tt = threshold_otsu(mask_t_res)  # Otsu阈值
        if not hasattr(self, "last_otsu") or self.last_otsu == []:  # 判断是否保存历史阈值
            self.last_otsu = [tt]
        else:
            self.last_otsu.append(tt)
        if use_otsu:    # 根据策略调整阈值
            upper_threshold, lower_threshold = (
                tt * upper_threshold,
                tt * lower_threshold,
            )

        if strategy == Binarization.PARTEDIT:   # 特殊策略
            upper_threshold = self.omega * tt  # Assuming we are not chaning upper in PartEdit

        if strategy in [Binarization.P2P, Binarization.PROVIDED_MASK]:  # 不二值化情况
            return mask_t_res

        mask_t_res[mask_t_res < lower_threshold] = 0    # 掩码二值化
        mask_t_res[mask_t_res >= upper_threshold] = 1.0

        return mask_t_res

    def has_maps(self) -> bool:
        return len(self.mask_storage_step) > 0 or len(self.mask_storage_agg) > 0

    def _store_agg_map(self) -> None:   # 存mask
        if self.use_agg_store:
            self.mask_storage_agg[self.cur_step] = self.calculate_mask_t_res().cpu()    # 跨步累计
        else:
            self.mask_storage_step[self.cur_step] = self.calculate_mask_t_res(True).cpu()   # 每步单独

    def between_steps(self):
        no_items = len(self.attention_store) == 0   # 是否是第一次运行
        if no_items:
            self.attention_store = self.step_store  # 初始化
        else:
            for key in self.attention_store:# 跨step累加
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]

        self._store_agg_map()   # 存mask
        if not no_items:
            # only in this case, otherwise we are just assigning it
            for key in self.step_store:
                # Clear the list while maintaining the dictionary structure
                del self.step_store[key][:] # 清空step_store

        self.step_store = self.get_empty_store()    # 重建空store

    def get_maps_agg(self, res, device, use_agg_store: bool = None, keepshape: bool = False):   # 获取mask
        if use_agg_store is None:   # 确定使用的哪种store
            use_agg_store = self.use_agg_store
        _store = self.mask_storage_agg if use_agg_store else self.mask_storage_step # 确定mask存储来源
        last_idx = sorted(_store.keys())[-1]    # 取最后一个step
        mask_t_res = _store[last_idx].to(device)  # Should be 1 1 H W   # 取出对应的mask
        mask_t_res = F.interpolate(mask_t_res, (res, res), mode="bilinear") # 上采样到分辨率
        if not keepshape:   # 是否reshape
            mask_t_res = mask_t_res.reshape(1, -1, 1)
        return mask_t_res

    def visualize_maps_agg(self, use_agg_store: bool, make_grid_kwargs: dict = None):   # 三个都是将mask转成图片
        _store = self.mask_storage_agg if use_agg_store else self.mask_storage_step
        if make_grid_kwargs is None:
            make_grid_kwargs = {"nrow": 10}
        return ToPILImage()(make_grid(torch.cat(list(_store.values())), **make_grid_kwargs))

    def visualize_one_map(self, use_agg_store: bool, idx: int):
        _store = self.mask_storage_agg if use_agg_store else self.mask_storage_step
        return ToPILImage()(_store[idx])

    def visualize_final_map(self, use_agg_store: bool):
        """This method returns the agg non-binarized attn map of the whole process

        Args:
            use_agg_store (bool): If True, it will return the agg store, otherwise the step store

        Returns:
            [PIL.Image]: The non-binarized attention map
        """
        _store = self.mask_storage_agg if use_agg_store else self.mask_storage_step
        return ToPILImage()(torch.cat(list(_store.values())).mean(0))

    # 对存储的attention按step做平均
    def get_average_attention(self, step: bool = False):
        _store = self.attention_store if not step else self.step_store
        average_attention = {key: [item / self.cur_step for item in _store[key]] for key in _store}
        return average_attention

    def reset(self):    # 清空运行状态
        super(AttentionStore, self).reset()
        for key in self.step_store:
            del self.step_store[key][:]
        for key in self.attention_store:
            del self.attention_store[key][:]
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.last_otsu = []

    def __init__(   # 初始化整个数据结构
        self,
        num_prompt: int,
        attn_res=None,
        extra_kwargs: DotDictExtra = None,
    ):
        super(AttentionStore, self).__init__(attn_res, extra_kwargs)

        print(">>> init AttentionStore")
        self.num_prompt = num_prompt
        self.mask_storage_step = {}
        self.mask_storage_agg = {}
        if self.batch_indx is not None:
            assert num_prompt > 0, "num_prompt must be greater than 0 if batch_indx is not None"
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.last_otsu = []


# 把 UNet 各层、各位置、各 step 的 attention map 汇总成统一空间分辨率的 attention 张量
def aggregate_attention(
    attention_store:AttentionStore,
    res: int,
    batch_size: int,
    from_where: List[str],
    is_cross: bool,
    upsample_everything: int = None,
    return_all_layers: bool = False,
    use_same_layers_as_train: bool = False,
    train_layers: Optional[List[int]] = None,
    use_layer_subset_idx: List[int] = None,
    use_step_store: bool = False,
    ):
    out = []    # 初始化输出列表
    attention_maps = attention_store.get_average_attention(use_step_store)  # 从 AttentionStore 中取出 attention
    num_pixels = res**2 # 计算目标像素数
    for location in from_where: # 遍历Unet的位置（up,middle,down)
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:    # 遍历该位置下的所有 attention 层

            if upsample_everything or (use_same_layers_as_train and is_cross):  # 是否需要插值（空间对齐）
                item = pack_interpolate_unpack(item, (res, res), "bilinear", rewrap=True)
            if item.shape[-2] == num_pixels:    # 只保留空间大小正确的 attention
                cross_maps = item.reshape(batch_size, -1, res, res, item.shape[-1])[None]   # 重排Attention维度
                out.append(cross_maps)
    _dim = 0    # 设定聚合维度
    if is_cross and use_same_layers_as_train and train_layers is not None:  # 训练层对齐（高级用法）
        out = [out[i] for i in train_layers]
        if use_layer_subset_idx is not None:  # 再次筛选 layer，只用特定深度的 attention
            out = [out[i] for i in use_layer_subset_idx]

    out = torch.cat(out, dim=_dim)  # 拼接所有 attention
    if return_all_layers:   # 是否返回所有层
        return out
    else:
        out = out.sum(_dim) / out.shape[_dim]   # 聚合所有层
    return out

class AttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t):   # 回调使用，这里的x_t是传入的latents
        if self.local_blend is not None:    # 是否进行局部融合
            # x_t = self.local_blend(x_t, self.attention_store) # TODO: Check if there is more memory efficient way
            x_t = self.local_blend(x_t, self)   # 查看，这两个参数
        return x_t

    def replace_self_attention(self, attn_base, att_replace):   # 自注意力替换策略
        if att_replace.shape[2] <= self.attn_res[0] ** 2:   # 判断是否是低分辨率层
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod # 抽象函数是什么？
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str, store: bool = True):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet, store) # 先执行父类
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):  # 判断是否需要编辑
            h = attn.shape[0] // (self.batch_size)  # 计算每个prompt的块大小
            try:
                attn = attn.reshape(self.batch_size, h, *attn.shape[1:])    # 重塑attn的维度
            except RuntimeError as e:   # 如果重置失败就打印
                logger.error(f"Batch size: {self.batch_size}, h: {h}, attn.shape: {attn.shape}")
                raise e

            attn_base, attn_replace = attn[0], attn[1:] # 分离base和replace
            if is_cross:    # 交叉注意力分支
                alpha_words = self.cross_replace_alpha[self.cur_step].to(attn_base.device)  # 获取alpha
                attn_replace_new = self.replace_cross_attention(attn_base, attn_replace) * alpha_words + (1 - alpha_words) * attn_replace   # 替换
                

                attn[1:] = attn_replace_new
                if self.has_maps() and self.extra_kwargs.get("force_cross_attn", False):  # 强制mask控制
                    mask_t_res = self.get_maps_agg( # 获取mask
                        res=int(attn_base.shape[1] ** 0.5),
                        device=attn_base.device,
                        use_agg_store=self.use_agg_store,  # Agg is across time, Step is last step without time agg
                        keepshape=False,
                    ).repeat(h, 1, 1)   # 扩展到head数
                    zero_index = torch.argmax(torch.eq(self.cross_replace_alpha[0], 0).to(mask_t_res.dtype)).item() # 找到被替换的token
                    # zero_index = torch.eq(self.cross_replace_alpha[0].flatten(), 0)
                    mean_curr = attn[1:2, ..., zero_index].mean()   # 当前attention均值
                    ratio_to_mean = mean_curr / mask_t_res[..., 0].mean()   # mask均值
                    # print(f'{ratio_to_mean=}')
                    extra_mask = torch.where(mask_t_res[..., 0] > self.last_otsu[-1], ratio_to_mean * 2, 0.5)   # 构建额外mask

                    attn[1:2, ..., zero_index : zero_index + 1] += mask_t_res[None] * extra_mask[None, ..., None]  # 加到attention上
                    # attn[1:2, ..., zero_index] = (mask_t_res[..., 0][None] > self.last_otsu[-1] * 1.5).to(mask_t_res.dtype) * mean_curr
            else:   # 自注意分支
                attn[1:] = self.replace_self_attention(attn_base, attn_replace)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])   # reshape到原始形状
        return attn

    def __init__(
        self,
        prompts: List[str],
        num_steps: int,
        cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
        self_replace_steps: Union[float, Tuple[float, float]],
        local_blend: Optional[LocalBlend],
        tokenizer,
        device: torch.device,
        attn_res=None,
        extra_kwargs: DotDictExtra = None,
    ):
        super(AttentionControlEdit, self).__init__(
            attn_res=attn_res,
            num_prompt=len(prompts),
            extra_kwargs=extra_kwargs,
        )
        # 在这里添加分词器和设备
        print(">>> init AttentionControlEdit")
        self.tokenizer = tokenizer
        self.device = device

        self.batch_size = len(prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, self.tokenizer).to(self.device)
        if isinstance(self_replace_steps, float):
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

# 具体实现的编辑控制器，用“token映射（mapper）”来替换 cross-attention
class AttentionReplace(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):  # 对替换具体实现
        return torch.einsum("hpw,bwn->bhpn", attn_base, self.mapper.to(attn_base.device))

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,   # localblend可以是Localblend实例，也可以是None
        tokenizer=None,
        device=None,
        attn_res=None, 
        extra_kwargs: DotDictExtra = None,
    ):
        super(AttentionReplace, self).__init__(
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
            tokenizer,
            device,
            attn_res,
            extra_kwargs,
        )
        print(">>> init AttentionReplace")
        self.mapper = get_replacement_mapper(prompts, self.tokenizer).to(self.device)

# 在“时间维度（扩散步）+ 单词维度”上，控制某些词的 attention 何时生效
def update_alpha_time_word(
    alpha,
    bounds: Union[float, Tuple[float, float]],
    prompt_ind: int,
    word_inds: Optional[torch.Tensor] = None,
):
    if isinstance(bounds, float):   # 判断 bounds 是否是 float
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])   # 计算时间步范围
    if word_inds is None:   # 如果没有指定词，那就使用全部词
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1 # 前部分关闭，中间打开，后部分关闭
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha

# 为所有 prompt + 所有词，构建一个完整的“时间-词 attention 控制表（alpha）
def get_time_words_attention_alpha(
    prompts,
    num_steps,
    cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
    tokenizer,
    max_num_words=77,
):
    if not isinstance(cross_replace_steps, dict):   # 如果不是 dict → 转成 dict
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:   # 如果没有default，补一个
        cross_replace_steps["default_"] = (0.0, 1.0)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)  # 初始化alpha_tensor
    for i in range(len(prompts) - 1):   # 第一轮：给所有词应用 default 规则
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"], i)
    for key, item in cross_replace_steps.items():   # 第二轮：对特定词进行覆盖（override）
        if key != "default_":
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)   # reshape
    return alpha_time_words

### util函数用于LocalBlend和replacentedit
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")    # 按空格切分文本
    if isinstance(word_place, str): # 如果输入的是“词字符串”
        word_place = [i for i, word in enumerate(split_text) if word_place == word] # 找到这个词在句子中的位置
    elif isinstance(word_place, int):   # 否则，统一成 list 格式
        word_place = [word_place]
    out = []    # 初始化输出
    if len(word_place) > 0: # 如果确实找到了词
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]   # 将一个词拆分成多个 token
        cur_len, ptr = 0, 0 # 初始化两个指针

        for i in range(len(words_encode)):  # 遍历token
            cur_len += len(words_encode[i]) # 累加token长度
            if ptr in word_place:   # 如果当前词是目标词
                out.append(i + 1)   # 把 token index 加入结果
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)

    ### util函数用于replacementit，构造一个 token-level 映射矩阵（mapper），用来把「原 prompt 的 cross-attention」映射到「新 prompt」
def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77):
    words_x = x.split(" ")  # 按空格切词
    words_y = y.split(" ")
    print(f"words_x is {words_x},length is {len(words_x)}")
    print(f"words_y is {words_y},length is {len(words_y)}")
    if len(words_x) != len(words_y):    # 检查长度是否一样
        raise ValueError(
            f"attention replacement edit can only be applied on prompts with the same length" f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words."
        )
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]] # 找到哪些位置的词变了
    inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]    # 找到它在 token 级别的位置
    inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]    # 同样
    mapper = np.zeros((max_len, max_len))   # 初始化 mapper 矩阵
    i = j = 0   # 初始化指针
    cur_inds = 0
    while i < max_len and j < max_len:  # 遍历整个 token 空间
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:   # 判断当前是否遇到替换词
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]   # 取出对应的token
            if len(inds_source_) == len(inds_target_):  # token相同
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1   # 移动指针
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):   # 如果是没到替换词，那就一一对齐
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1

    # return torch.from_numpy(mapper).float()
    return torch.from_numpy(mapper).to(torch.float16)

# 是对上面函数的批量封装
def get_replacement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):    # 遍历所有prompt
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
    return torch.stack(mappers)