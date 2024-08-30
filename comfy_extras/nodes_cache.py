from comfy.model_management import get_torch_device, load_model_gpu, intermediate_device
from comfy.model_patcher import ModelPatcher

default_device = get_torch_device()
default_offload = intermediate_device()
global_cache = {}


def load_model_cached(cache_key, load_model, device=default_device, offload=default_offload):
    if cache_key in global_cache:
        wrapped_model = global_cache[cache_key]
    else:
        model = load_model()
        wrapped_model = ModelPatcher(fix_device_attr(model), load_device=device, offload_device=offload)
        global_cache[cache_key] = wrapped_model
    load_model_gpu(wrapped_model)
    return wrapped_model.model


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")


class ReadOnlyDeviceWrapper:
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def __setattr__(self, name, value):
        if name == '_obj':
            super().__setattr__(name, value)
        elif name == 'device':
            pass
        else:
            setattr(self._obj, name, value)

    def __delattr__(self, name):
        delattr(self._obj, name)

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)


class DeviceWrapper:
    def __init__(self, obj):
        self._obj = obj
        self._device = None  # 初始化 device 属性

    def __getattr__(self, name):
        if name == "device":
            return self._device
        return getattr(self._obj, name)

    def __setattr__(self, name, value):
        if name == "_obj" or name == "_device":
            super().__setattr__(name, value)
        elif name == "device":
            self._device = value
        else:
            setattr(self._obj, name, value)

    def __delattr__(self, name):
        if name == "device":
            del self._device
        else:
            delattr(self._obj, name)

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)


def fix_device_attr(obj):
    try:
        device = getattr(obj, 'device')
    except AttributeError:
        return DeviceWrapper(obj)

    try:
        setattr(obj, 'device', device)
    except AttributeError:
        return ReadOnlyDeviceWrapper(obj)

    return obj


class CacheModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": (any_typ, {"lazy": True}),
                             "device": (["auto", "cpu", "gpu"], {"default": "auto"}),
                             "offload": (["auto", "cpu", "gpu"], {"default": "auto"}),
                             "cache_key": ("STRING", {"default": ""})}}

    RETURN_TYPES = (any_typ,)
    FUNCTION = "cache_model"
    CATEGORY = "cache"

    def cache_model(self, model, device, offload, cache_key):
        if device == "cpu":
            device = "cpu"
        elif device == "gpu":
            device = get_torch_device()
        else:
            device = None

        if offload == "cpu":
            offload = "cpu"
        elif offload == "gpu":
            offload = get_torch_device()
        else:
            offload = None

        model = load_model_cached(cache_key, lambda: model, device, offload)
        return model,

    def check_lazy_status(self, model, device, offload, cache_key):
        if cache_key in global_cache:
            return []
        return ["model"]


NODE_CLASS_MAPPINGS = {
    "CacheModel": CacheModel,
}
