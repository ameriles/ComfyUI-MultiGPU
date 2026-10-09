# Load LoRA Model Only (MultiGPU)

Loads a model-only LoRA for a model distributed by a DisTorch2 loader.

## Inputs

- `model`: Model returned by a DisTorch2 model loader.
- `lora_name`: LoRA file from `ComfyUI/models/loras`.
- `strength_model`: LoRA strength applied to the diffusion model.
- `patch_mode`:
  - `standard_safe` uses ComfyUI's normal patching behavior and preserves weight backups so patches can be changed or removed normally.
  - `target_device_low_ram` materializes each patched weight directly on the device assigned to its layer. It avoids full base-weight backups and the extra transfer from the compute device to the final DisTorch2 device.

## Low-RAM mode limitation

`target_device_low_ram` is intentionally one-way for the lifetime of the loaded transformer. Reload the base transformer before changing the LoRA, changing its strength, or forcing weights to be patched again. The node raises an error instead of attempting an unsafe unpatch without backups.

Use `standard_safe` when a workflow needs to switch LoRAs dynamically. Use `target_device_low_ram` when the model is large, system RAM is constrained, or model layers are spread across multiple GPUs and the LoRA will remain fixed for the session.
