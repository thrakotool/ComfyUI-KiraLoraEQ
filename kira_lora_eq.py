# kira_lora_eq.py
#
# Memory-only 20-band equalizer for a single LoRA file.
# Supports Z-Image/Flux (standard comfy.lora) and Qwen (via nunchaku compose_loras_v2).
#
# Band Mapping:
#   - Layers are grouped and mapped to bands 1-15
#   - Bands 16-20 reserved for future use
#
# Global Controls:
#   - Prompt Match:         cross-attention layers (text conditioning)
#   - Shape & Composition:  self-attention layers (image structure)
#   - Texture & Color:      feed-forward / MLP layers
#
# Effective strength per band: Master Gain * band_value * type_gain

from typing import Dict, Tuple, Any, Optional, Set, List
import re

import folder_paths

try:
    import torch
    from safetensors.torch import load_file
except Exception as e:
    torch = None
    load_file = None
    print("[KiraLora_EQ] WARNING: torch/safetensors import failed:", e)

HAS_COMFY = True
try:
    import comfy
    import comfy.lora
    import comfy.lora_convert
    import comfy.sd
    import comfy.model_management
except Exception as e:
    comfy = None
    HAS_COMFY = False
    print("[KiraLora_EQ] WARNING: comfy internals import failed:", e)

DEBUG_QWEN = False

def _dprint(*args, **kwargs):
    if DEBUG_QWEN:
        print(*args, **kwargs)

BAND_COUNT = 20
BAND_NAMES = [f"band_{i:02d}" for i in range(1, BAND_COUNT + 1)]

_LORA_CACHE: Dict[str, Tuple[Dict, set, set, set, int]] = {}

_LAYER_PATTERNS = [
    re.compile(r"diffusion_model\.transformer_blocks\.(\d+)\."),
    re.compile(r"transformer_blocks\.(\d+)\."),
    re.compile(r"diffusion_model\.layers\.(\d+)\."),
    re.compile(r"transformer\.layers\.(\d+)\."),
    re.compile(r"model\.layers\.(\d+)\."),
    re.compile(r"blocks\.(\d+)\."),
]

_QWEN_WRAPPER_NAMES = (
    "ComfyQwenImageWrapper",
    "NunchakuQwenImageTransformer2DModel",
    "QwenImageTransformer",
    "QwenImage",
)


def _is_qwen_model(model) -> bool:
    """Detect if the model uses Qwen/Nunchaku wrapper."""
    if model is None:
        return False
    try:
        diffusion_model = getattr(model.model, "diffusion_model", None)
        if diffusion_model is None:
            return False
        class_name = type(diffusion_model).__name__
        if any(qwen_name in class_name for qwen_name in _QWEN_WRAPPER_NAMES):
            return True
        if hasattr(diffusion_model, "loras") and isinstance(diffusion_model.loras, list):
            return True
    except Exception:
        pass
    return False


def _classify_key(name: str) -> str:
    """Classify LoRA key by layer type: 'cross', 'self', 'mlp', or 'other'."""
    if ".lora_" not in name:
        return "other"

    if ".attn.add_" in name:
        return "cross"
    if ".attn2." in name:
        return "cross"
    if ".context_attn." in name or ".xattn." in name:
        return "cross"
    if ".txt_attn." in name:
        return "cross"

    if ".attn.to_" in name:
        return "self"
    if ".attn1." in name:
        return "self"
    if ".self_attn." in name:
        return "self"
    if ".img_attn." in name:
        return "self"
    if ".adaLN_modulation." in name:
        return "self"

    if ".attention." in name:
        return "cross"

    if ".img_mlp." in name:
        return "mlp"
    if ".txt_mlp." in name:
        return "mlp"
    if ".feed_forward." in name or ".ff." in name or ".mlp." in name:
        return "mlp"

    return "other"


def _extract_layer_index(key: str) -> int:
    """Extract layer index from key using multiple patterns."""
    for pattern in _LAYER_PATTERNS:
        m = pattern.search(key)
        if m:
            return int(m.group(1))
    return -1


def _detect_max_layer(keys: set) -> int:
    """Detect the maximum layer index from a set of keys."""
    max_layer = -1
    for k in keys:
        idx = _extract_layer_index(k)
        if idx > max_layer:
            max_layer = idx
    return max_layer


def _load_and_classify_lora(lora_name: str) -> Tuple[Dict, set, set, set, int]:
    """Load LoRA and classify keys by type."""
    if lora_name in _LORA_CACHE:
        return _LORA_CACHE[lora_name]

    full_path = folder_paths.get_full_path("loras", lora_name)
    if full_path is None:
        print(f"[KiraLora_EQ] Cannot find LoRA: {lora_name}")
        return ({}, set(), set(), set(), -1)

    try:
        state = load_file(full_path)
    except Exception as e:
        print(f"[KiraLora_EQ] Failed to load {lora_name}: {e}")
        return ({}, set(), set(), set(), -1)

    if not state:
        print(f"[KiraLora_EQ] Empty state_dict for {lora_name}")
        return ({}, set(), set(), set(), -1)

    cross_keys = set()
    self_keys = set()
    mlp_keys = set()

    for k in state.keys():
        kind = _classify_key(k)
        if kind == "cross":
            cross_keys.add(k)
        elif kind == "self":
            self_keys.add(k)
        elif kind == "mlp":
            mlp_keys.add(k)
        else:
            mlp_keys.add(k)

    all_keys = cross_keys | self_keys | mlp_keys
    max_layer = _detect_max_layer(all_keys)

    result = (state, cross_keys, self_keys, mlp_keys, max_layer)
    _LORA_CACHE[lora_name] = result

    print(
        f"[KiraLora_EQ] Loaded {lora_name}: "
        f"cross={len(cross_keys)}, self={len(self_keys)}, mlp={len(mlp_keys)}, "
        f"layers=0..{max_layer}"
    )

    return result


def _layer_to_band_index(layer_idx: int, max_layer: int) -> int:
    """Map layer index to band index (0..14 for bands 1-15)."""
    if layer_idx < 0 or max_layer < 0:
        return -1
    if max_layer == 0:
        return 0
    total_layers = max_layer + 1
    band = int((layer_idx / total_layers) * 15)
    return min(band, 14)


def _split_keys_into_bands_by_layer(keys: set, max_layer: int):
    """Split keys into BAND_COUNT groups by layer index."""
    bands = [set() for _ in range(BAND_COUNT)]
    if not keys:
        return bands

    for k in keys:
        layer_idx = _extract_layer_index(k)
        if layer_idx == -1:
            bands[7].add(k)
            continue
        b = _layer_to_band_index(layer_idx, max_layer)
        if b == -1:
            continue
        bands[b].add(k)

    return bands


def _build_paired_filtered_state(state_dict: Dict[str, Any], keys_to_use: set) -> Dict[str, Any]:
    """Build a filtered state dict containing only valid LoRA pairs."""
    if not keys_to_use:
        return {}

    filtered_state: Dict[str, Any] = {}

    for k in keys_to_use:
        if k.endswith(".lora_A.weight"):
            partner = k.replace(".lora_A.weight", ".lora_B.weight")
            if partner in state_dict and partner in keys_to_use:
                filtered_state[k] = state_dict[k]
                filtered_state[partner] = state_dict[partner]
        elif k.endswith(".lora_B.weight"):
            partner = k.replace(".lora_B.weight", ".lora_A.weight")
            if partner in state_dict and partner in keys_to_use:
                filtered_state[k] = state_dict[k]
                filtered_state[partner] = state_dict[partner]
        else:
            if k in state_dict:
                filtered_state[k] = state_dict[k]

    return filtered_state


def _apply_lora_subset_standard(
    model,
    clip,
    state_dict: Dict[str, Any],
    keys_to_use: set,
    strength_model: float,
    strength_clip: float = 0.0,
):
    """Apply LoRA subset using standard comfy.lora path (Z-Image/Flux)."""
    if not keys_to_use or (strength_model == 0.0 and strength_clip == 0.0):
        return model, clip

    filtered_state = _build_paired_filtered_state(state_dict, keys_to_use)
    if not filtered_state:
        return model, clip

    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    converted = comfy.lora_convert.convert_lora(filtered_state)
    loaded = comfy.lora.load_lora(converted, key_map)

    if not loaded:
        return model, clip

    if model is not None and strength_model != 0.0:
        model.add_patches(loaded, strength_model)

    if clip is not None and strength_clip != 0.0:
        clip.add_patches(loaded, strength_clip)

    return model, clip


def _strip_diffusion_model_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove 'diffusion_model.' prefix from keys if present.
    compose_loras_v2 may expect keys without this prefix.
    """
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("diffusion_model."):
            new_key = k[len("diffusion_model."):]
        else:
            new_key = k
        new_state[new_key] = v
    return new_state


def _apply_qwen_loras_via_wrapper(model, lora_list: List[Tuple[Dict, float]]):
    """
    Apply LoRAs to Qwen model via the wrapper's loras mechanism.
    Tries multiple approaches in order of preference.
    """
    if not lora_list:
        return False
    
    try:
        wrapper = model.model.diffusion_model
        
        # Debug: print wrapper info
        wrapper_type = type(wrapper).__name__
        wrapper_attrs = [a for a in dir(wrapper) if not a.startswith('_')]
        _dprint(f"[KiraLora_EQ] DEBUG wrapper type: {wrapper_type}")
        _dprint(f"[KiraLora_EQ] DEBUG wrapper attrs: {wrapper_attrs[:30]}...")
        
        # Check for nunchaku in sys.modules
        import sys
        nunchaku_modules = [m for m in sys.modules.keys() if 'nunchaku' in m.lower()]
        _dprint(f"[KiraLora_EQ] DEBUG nunchaku modules loaded: {nunchaku_modules}")
        
        # Find the inner transformer model
        inner = None
        for attr in ["model", "transformer", "dit", "inner_model"]:
            if hasattr(wrapper, attr):
                candidate = getattr(wrapper, attr)
                if candidate is not None:
                    inner = candidate
                    inner_type = type(inner).__name__
                    _dprint(f"[KiraLora_EQ] DEBUG found inner model via '{attr}': {inner_type}")
                    break
        
        if inner is None:
            inner = wrapper
            _dprint(f"[KiraLora_EQ] DEBUG using wrapper as inner model")
        
        # Method 1: Try compose_loras_v2 from nunchaku
        compose_func = None
        reset_func = None
        
        try:
            from nunchaku.lora import compose_loras_v2, reset_lora_v2
            compose_func = compose_loras_v2
            reset_func = reset_lora_v2
            _dprint("[KiraLora_EQ] DEBUG imported from nunchaku.lora")
        except ImportError as e1:
            _dprint(f"[KiraLora_EQ] DEBUG nunchaku.lora import failed: {e1}")
            try:
                from nunchaku.lora.compose import compose_loras_v2, reset_lora_v2
                compose_func = compose_loras_v2
                reset_func = reset_lora_v2
                _dprint("[KiraLora_EQ] DEBUG imported from nunchaku.lora.compose")
            except ImportError as e2:
                _dprint(f"[KiraLora_EQ] DEBUG nunchaku.lora.compose import failed: {e2}")
        
        # Check if inner model has lora-related methods
        inner_attrs = [a for a in dir(inner) if 'lora' in a.lower()]
        _dprint(f"[KiraLora_EQ] DEBUG inner model lora-related attrs: {inner_attrs}")
        
        # Check for any 'load', 'apply', 'patch' methods
        inner_methods = [a for a in dir(inner) if any(x in a.lower() for x in ['load', 'apply', 'patch', 'merge'])]
        _dprint(f"[KiraLora_EQ] DEBUG inner model load/apply/patch methods: {inner_methods[:20]}")
        
        if compose_func is not None:
            # Reset existing LoRAs
            if reset_func:
                try:
                    reset_func(inner)
                except Exception as e:
                    print(f"[KiraLora_EQ] reset_lora_v2 failed (non-fatal): {e}")
            
            # Prepare loras for compose_loras_v2
            # Try with stripped prefixes first
            prepared_loras = []
            for state_dict, strength in lora_list:
                stripped = _strip_diffusion_model_prefix(state_dict)
                prepared_loras.append((stripped, strength))
            
            try:
                compose_func(inner, prepared_loras)
                print(f"[KiraLora_EQ] Applied {len(prepared_loras)} Qwen LoRA subsets via compose_loras_v2 (stripped keys)")
                return True
            except Exception as e1:
                print(f"[KiraLora_EQ] compose_loras_v2 with stripped keys failed: {e1}")
                
                # Try with original keys
                try:
                    if reset_func:
                        reset_func(inner)
                    compose_func(inner, lora_list)
                    print(f"[KiraLora_EQ] Applied {len(lora_list)} Qwen LoRA subsets via compose_loras_v2 (original keys)")
                    return True
                except Exception as e2:
                    print(f"[KiraLora_EQ] compose_loras_v2 with original keys failed: {e2}")
        
        # Method 2: Set wrapper.loras directly
        wrapper_lora_attrs = [a for a in dir(wrapper) if 'lora' in a.lower()]
        _dprint(f"[KiraLora_EQ] DEBUG wrapper lora-related attrs: {wrapper_lora_attrs}")
        
        if hasattr(wrapper, "loras"):
            # Try stripped keys
            prepared_loras = []
            for state_dict, strength in lora_list:
                stripped = _strip_diffusion_model_prefix(state_dict)
                prepared_loras.append((stripped, strength))
            
            wrapper.loras = prepared_loras
            print(f"[KiraLora_EQ] Set {len(prepared_loras)} LoRAs on wrapper.loras (stripped keys)")
            return True
        
        print("[KiraLora_EQ] No suitable Qwen LoRA application method found")
        
        # Method 3:
        _dprint("[KiraLora_EQ] DEBUG Attempting comfy standard path as fallback...")
        try:
            for state_dict, strength in lora_list:
                converted = comfy.lora_convert.convert_lora(state_dict)
                # Build key_map from model
                key_map = comfy.lora.model_lora_keys_unet(model.model, {})
                _dprint(f"[KiraLora_EQ] DEBUG key_map has {len(key_map)} entries")
                if key_map:
                    sample_keys = list(key_map.keys())[:3]
                    _dprint(f"[KiraLora_EQ] DEBUG sample key_map keys: {sample_keys}")
                loaded = comfy.lora.load_lora(converted, key_map)
                if loaded:
                    model.add_patches(loaded, strength)
                    _dprint(f"[KiraLora_EQ] DEBUG comfy fallback: loaded {len(loaded)} patches with strength {strength}")
                else:
                    _dprint(f"[KiraLora_EQ] DEBUG comfy fallback: load_lora returned empty")
            return True
        except Exception as e:
            _dprint(f"[KiraLora_EQ] DEBUG comfy fallback failed: {e}")
        
        return False
        
    except Exception as e:
        print(f"[KiraLora_EQ] Error applying Qwen LoRAs: {e}")
        import traceback
        traceback.print_exc()
        return False


class KiraLora_EQ:
    """20-band equalizer loader for a single LoRA. Memory-only, no files created on disk."""

    @classmethod
    def INPUT_TYPES(cls):
        lora_files = ["None"] + folder_paths.get_filename_list("loras")

        required = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "lora_name": (
                lora_files,
                {
                    "default": "None",
                    "label": "LoRA",
                    "tooltip": "Select LoRA file to apply with EQ controls.",
                },
            ),
            "gain": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "label": "Master Gain",
                },
            ),
        }

        for name in BAND_NAMES:
            required[name] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.5,
                    "step": 0.05,
                },
            )

        required["Prompt Match"] = (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
        )
        required["Shape & Composition"] = (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
        )
        required["Texture & Color"] = (
            "FLOAT",
            {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
        )

        return {"required": required}

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    FUNCTION = "apply"
    CATEGORY = "Kira/LoRA EQ"

    def apply(self, model, clip, lora_name, gain, **kwargs):
        if torch is None or not HAS_COMFY:
            print("[KiraLora_EQ] Missing dependencies, returning unchanged.")
            return (model, clip)

        if lora_name == "None":
            return (model, clip)

        is_qwen = _is_qwen_model(model)
        model_type = "Qwen" if is_qwen else "Standard"
        print(f"[KiraLora_EQ] Loading LoRA: {lora_name} (model type: {model_type})")

        g = float(gain)
        pm = float(kwargs.get("Prompt Match", 1.0))
        sc = float(kwargs.get("Shape & Composition", 1.0))
        tc = float(kwargs.get("Texture & Color", 1.0))

        band_vals = []
        for name in BAND_NAMES:
            try:
                v = float(kwargs.get(name, 1.0))
            except Exception:
                v = 1.0
            band_vals.append(max(0.0, v))

        if g == 0.0 or all(v == 0.0 for v in band_vals):
            print("[KiraLora_EQ] All gains are zero, skipping apply.")
            return (model, clip)

        state_dict, cross_keys, self_keys, mlp_keys, max_layer = _load_and_classify_lora(lora_name)

        if not state_dict:
            print(f"[KiraLora_EQ] Failed to load {lora_name}")
            return (model, clip)

        current_model = model.clone()
        current_clip = clip

        # For Qwen, accumulate all LoRA subsets as (state_dict, strength) tuples
        qwen_lora_list: List[Tuple[Dict, float]] = []

        # Fast path: all bands and type gains at default
        if (
            all(abs(v - 1.0) < 1e-6 for v in band_vals)
            and abs(pm - 1.0) < 1e-6
            and abs(sc - 1.0) < 1e-6
            and abs(tc - 1.0) < 1e-6
        ):
            all_keys = cross_keys | self_keys | mlp_keys
            if is_qwen:
                filtered = _build_paired_filtered_state(state_dict, all_keys)
                if filtered:
                    qwen_lora_list.append((filtered, g))
                _apply_qwen_loras_via_wrapper(current_model, qwen_lora_list)
            else:
                current_model, current_clip = _apply_lora_subset_standard(
                    current_model, current_clip, state_dict, all_keys, g, 0.0
                )
            print(f"[KiraLora_EQ] Applied full LoRA with strength={g:.3f}")
            return (current_model, current_clip)

        # Standard path: apply per-band
        cross_bands = _split_keys_into_bands_by_layer(cross_keys, max_layer)
        self_bands = _split_keys_into_bands_by_layer(self_keys, max_layer)
        mlp_bands = _split_keys_into_bands_by_layer(mlp_keys, max_layer)

        applied_count = 0

        for idx, band_val in enumerate(band_vals):
            if band_val == 0.0:
                continue

            base_strength = g * band_val
            if base_strength == 0.0:
                continue

            if pm != 0.0 and cross_bands[idx]:
                strength = base_strength * pm
                if is_qwen:
                    filtered = _build_paired_filtered_state(state_dict, cross_bands[idx])
                    if filtered:
                        qwen_lora_list.append((filtered, strength))
                else:
                    current_model, current_clip = _apply_lora_subset_standard(
                        current_model, current_clip, state_dict, cross_bands[idx], strength, 0.0
                    )
                applied_count += 1

            if sc != 0.0 and self_bands[idx]:
                strength = base_strength * sc
                if is_qwen:
                    filtered = _build_paired_filtered_state(state_dict, self_bands[idx])
                    if filtered:
                        qwen_lora_list.append((filtered, strength))
                else:
                    current_model, current_clip = _apply_lora_subset_standard(
                        current_model, current_clip, state_dict, self_bands[idx], strength, 0.0
                    )
                applied_count += 1

            if tc != 0.0 and mlp_bands[idx]:
                strength = base_strength * tc
                if is_qwen:
                    filtered = _build_paired_filtered_state(state_dict, mlp_bands[idx])
                    if filtered:
                        qwen_lora_list.append((filtered, strength))
                else:
                    current_model, current_clip = _apply_lora_subset_standard(
                        current_model, current_clip, state_dict, mlp_bands[idx], strength, 0.0
                    )
                applied_count += 1

        # Apply all accumulated Qwen LoRAs
        if is_qwen and qwen_lora_list:
            _apply_qwen_loras_via_wrapper(current_model, qwen_lora_list)

        print(f"[KiraLora_EQ] Applied {applied_count} LoRA subsets.")
        return (current_model, current_clip)


NODE_CLASS_MAPPINGS = {
    "KiraLora_EQ": KiraLora_EQ,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KiraLora_EQ": "Kira LoRA EQ",
}
