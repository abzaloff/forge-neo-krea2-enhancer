# Forge Neo Krea2 Enhancer

Krea2 Enhancer is a Forge Neo extension for Krea 2 models. It improves prompt adherence by applying a controlled adjustment inside the Krea 2 text-fusion path during sampling.

The implementation is based on the same generation-path intervention used by the ComfyUI Krea2T Enhancer custom node, adapted for Forge Neo's UNet wrapper pipeline.

## Features

- Adds a `Krea2 Enhancer` accordion to the Forge Neo generation UI.
- Can be enabled or disabled per generation.
- Provides a single `Strength` control.
- Uses `0.5` as the default strength.
- Automatically skips non-Krea2 models.
- Does not modify prompts, conditioning text, checkpoints, or saved model files.

## How It Works

Krea 2 uses a text-fusion stage that combines the multi-layer text conditioning stack before it is passed into the diffusion blocks. This extension temporarily wraps that text-fusion forward pass during sampling.

When enabled, the extension:

1. Detects the expected Krea 2 text-conditioning layout.
2. Runs the normal text-fusion path as a reference.
3. Applies the enhancer profile to selected internal text-conditioning chunks.
4. Runs the adjusted text-fusion path.
5. Blends the delta back with a per-token relative cap to keep the effect bounded.
6. Restores the original model forward path immediately after the UNet call.

The patch is runtime-only. It is applied through Forge Neo's `model_function_wrapper` and is removed after each wrapped diffusion-model call.

## Controls

| Control | Default | Range | Description |
| --- | ---: | ---: | --- |
| Enabled | Off | On/Off | Turns Krea2 Enhancer on for the current generation. |
| Strength | `0.5` | `0.0` to `2.0` | Controls how strongly the internal text-fusion adjustment is applied. `0.0` is neutral. |

## Installation

### Install from URL

In Forge Neo, open the `Extensions` tab, go to `Install from URL`, and use:

```text
https://github.com/abzaloff/forge-neo-krea2-enhancer
```

Click `Install`, then restart Forge Neo.

### Manual Git Clone

Clone this repository into your Forge Neo `extensions` directory:

```bash
cd /path/to/sd-webui-forge-neo/extensions
git clone https://github.com/abzaloff/forge-neo-krea2-enhancer.git
```

Restart Forge Neo after installation.

## Compatibility

This extension is intended for Forge Neo builds that include Krea 2 model support.

It checks for the Krea 2 `12 x 2560` text-conditioning layout before applying the patch. If the loaded diffusion model does not match that layout, the extension does nothing.

## Notes

- The debug option from the ComfyUI node is intentionally not exposed in the Forge Neo UI.
- The extension may increase sampling cost because it evaluates the text-fusion path twice while enabled.
- Effects depend on the prompt, sampler settings, and selected Krea 2 checkpoint.

## Credits

This Forge Neo adaptation follows the behavior of the ComfyUI Krea2T Enhancer custom node.
