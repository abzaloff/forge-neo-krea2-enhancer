import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

import torch


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "krea2_enhancer.py"


def load_extension():
    scripts = types.ModuleType("modules.scripts")
    scripts.ScriptBuiltinUI = type("ScriptBuiltinUI", (), {})
    scripts.AlwaysVisible = object()
    modules = types.ModuleType("modules")
    modules.scripts = scripts
    ui_components = types.ModuleType("modules.ui_components")
    ui_components.InputAccordion = object
    with patch.dict(sys.modules, {"modules": modules, "modules.scripts": scripts,
                                  "modules.ui_components": ui_components}):
        spec = importlib.util.spec_from_file_location("krea2_enhancer_under_test", SCRIPT)
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
    return extension


class TextFusion:
    def __init__(self):
        self.layerwise_blocks = []
        self.refiner_blocks = []
        self.projector = lambda x: x.mean(dim=-1, keepdim=True)

    def forward(self, x, mask=None, transformer_options=None):
        return x.mean(dim=2)


class TextMLP:
    def forward(self, x):
        return x + 1


class DiffusionModel:
    txtlayers = 12
    txtdim = 2560

    def __init__(self):
        self.txtfusion = TextFusion()
        self.txtmlp = TextMLP()
        self.blocks = [object()]


class Patcher:
    def __init__(self, dm):
        self.dm = dm
        self.model_options = {}

    def clone(self):
        cloned = Patcher(self.dm)
        cloned.model_options = self.model_options.copy()
        return cloned

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.dm

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


class EnhancerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extension = load_extension()

    def test_current_forge_layout_is_accepted_without_unpack_context(self):
        self.assertTrue(self.extension._is_krea2_dm(DiffusionModel()))

    def test_text_scale_only_and_previous_wrapper(self):
        dm = DiffusionModel()
        original_fusion = dm.txtfusion.forward
        original_mlp = dm.txtmlp.forward
        calls = []

        def previous(model_function, kwargs):
            calls.append("previous")
            return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])

        patcher = Patcher(dm)
        patcher.model_options["model_function_wrapper"] = previous
        p = types.SimpleNamespace(
            sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=patcher)),
            extra_generation_params={},
        )
        self.extension.Krea2EnhancerForForge().process_before_every_sampling(p, True, 0, 2)
        wrapper = p.sd_model.forge_objects.unet.model_options["model_function_wrapper"]

        def model_function(x, timestep, **c):
            return dm.txtmlp.forward(dm.txtfusion.forward(x))

        x = torch.ones(1, 1, 12, 2560)
        result = wrapper(model_function, {"input": x, "timestep": 1, "c": {}})
        self.assertTrue(torch.all(result == 4))
        self.assertEqual(calls, ["previous"])
        self.assertIs(dm.txtfusion.forward.__func__, original_fusion.__func__)
        self.assertIs(dm.txtmlp.forward.__func__, original_mlp.__func__)

    def test_enhancement_changes_output_and_restores_after_error(self):
        dm = DiffusionModel()
        original_fusion = dm.txtfusion.forward
        original_mlp = dm.txtmlp.forward
        wrapper = self.extension._make_unet_wrapper(dm, 1.0, 1.5)
        x = torch.ones(1, 1, 12, 2560)

        def model_function(x, timestep, **c):
            return dm.txtmlp.forward(dm.txtfusion.forward(x))

        result = wrapper(model_function, {"input": x, "timestep": 1, "c": {}})
        self.assertGreater(float(result.mean()), 3.0)

        def broken_model(x, timestep, **c):
            dm.txtfusion.forward(x)
            raise RuntimeError("sampling failed")

        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            wrapper(broken_model, {"input": x, "timestep": 1, "c": {}})
        self.assertIs(dm.txtfusion.forward.__func__, original_fusion.__func__)
        self.assertIs(dm.txtmlp.forward.__func__, original_mlp.__func__)


if __name__ == "__main__":
    unittest.main()
