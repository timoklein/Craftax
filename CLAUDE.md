# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Craftax is a JAX-native reinforcement learning environment inspired by Crafter and NetHack. The entire environment (world gen, game logic, rendering) is written in pure JAX, making it fully JIT-compilable and vectorizable. It conforms to the **gymnax** environment interface. Fork of `MichaelTMatthews/Craftax`.

## Commands

```bash
# Install (editable)
pip install -e ".[dev]"

# Format
black .

# Interactive play (expect ~30s initial JIT compilation)
play_craftax
play_craftax_classic
```

There is no test suite. To verify changes work, run a quick smoke test:

```python
import jax
from craftax.craftax_env import make_craftax_env_from_name

rng = jax.random.PRNGKey(0)
env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
env_params = env.default_params
obs, state = env.reset(rng, env_params)
for i in range(100):
    rng, rng_step, rng_act = jax.random.split(rng, 3)
    action = env.action_space(env_params).sample(rng_act)
    obs, state, reward, done, info = env.step(rng_step, state, action, env_params)
print(f"obs shape: {obs.shape}, reward: {reward}, done: {done}")
```

## Architecture

Two parallel game variants share the same structure:

- **`craftax/craftax/`** — Extended version (9-level dungeon, spells, enchantments, potions, boss)
- **`craftax/craftax_classic/`** — Original Crafter-equivalent (simpler mechanics)

Each variant contains:

| File | Purpose |
|------|---------|
| `craftax_state.py` | Flax dataclasses: `EnvState`, `EnvParams`, `StaticEnvParams`, `Inventory`, `Mobs` |
| `constants.py` | Enums (`BlockType`, `Action`, `ItemType`, `MobType`), observation dimensions, game parameters |
| `game_logic.py` | Core step function (`craftax_step`) — combat, crafting, movement, intrinsics |
| `renderer.py` | Observation rendering (symbolic one-hot and pixel modes) |
| `world_gen/` | Procedural world generation using Perlin noise |
| `envs/` | Gymnax-compatible env classes (auto-reset and no-auto-reset variants) |
| `play_craftax.py` | Pygame interactive play entry point |

Shared base:
- **`craftax/environment_base/environment_bases.py`** — `EnvironmentNoAutoReset` base class (JIT-decorated `step`/`reset`)
- **`craftax/craftax_env.py`** — Factory functions: `make_craftax_env_from_name()`, `make_craftax_env_from_params()`

## Environment Variants

Available via `make_craftax_env_from_name(name, auto_reset)`:

| Name | Obs type |
|------|----------|
| `Craftax-Symbolic-v1` | Flat symbolic (8268,) |
| `Craftax-Pixels-v1` | Pixel rendering |
| `Craftax-Classic-Symbolic-v1` | Classic flat symbolic |
| `Craftax-Classic-Pixels-v1` | Classic pixel rendering |

All have auto-reset (gymnax `Environment`) and no-auto-reset variants. NoAutoReset envs must use `OptimisticResetVecEnvWrapper` or `AutoResetEnvWrapper`.

## Key Constants (extended variant)

- Map: 48x48, 9 dungeon levels
- Observation window: 9x11 tiles, 83 channels (37 block + 5 item + 40 mob + 1 light)
- 43 discrete actions, 37 block types, 22 achievements
- `EnvParams.max_timesteps = 100000`, `day_length = 300`

## Float64 / x64 Compatibility

**Both variants** (`craftax/craftax/` and `craftax/craftax_classic/`) are fully patched to produce zero user-visible float64/int64 values under `jax.config.update("jax_enable_x64", True)`. The environments never create float64 values — all computation stays in float32/int32 even when float64 is globally enabled.

### Design principle

Every operation that could silently promote to float64 under x64 has been pinned to float32/int32. The only float64 remaining in the JIT-compiled HLO (~110 ops) is **irreducible JAX internals** — `jax.random.uniform` uses float64 constants for its min/max bounds even with `dtype=jnp.float32`. These produce no user-visible float64 values.

### Changes made

**Selection operators:**
- All `jax.lax.select` → `jnp.where` in `game_logic.py`, `game_logic_utils.py`, `renderer.py`, and `world_gen.py`
- Auto-reset env classes override gymnax's `step` and `discount` methods to use `jnp.where` with `jnp.float32` literals
- `environment_bases.py` `discount()` uses `jnp.where` with `jnp.float32(0.0/1.0)`

**Array constructors:**
- Explicit `dtype=jnp.float32` on all float array constructors, `dtype=jnp.int32` on integer ones
- All `jnp.arange()` calls have explicit `dtype=jnp.int32` (prevents int64 intermediates under x64)
- Import-time constants (`ACHIEVEMENT_REWARD_MAP`, `TORCH_LIGHT_MAP`) have explicit dtypes
- `get_distance_map` returns float32 (casts before `jnp.sqrt`)

**Random number generation:**
- All `jax.random.uniform` calls have `dtype=jnp.float32` (both scalar and array-producing)
- All `jax.random.randint` calls have `dtype=jnp.int32`
- All `jax.random.choice` calls with probability arrays ensure `p` is float32 — bool masks are `.astype(jnp.float32)` before passing as `p` to prevent `bool/int → float64` promotion in the internal cumsum

**Literal wrapping:**
- Python float literals wrapped with `jnp.float32(...)` in arithmetic and `jnp.clip` bounds
- `jnp.pi` wrapped with `jnp.float32(jnp.pi)` in `calculate_light_level` and noise generation
- `dynamic_slice`/`dynamic_update_slice` index tuples use `jnp.int32(0)` instead of bare `0`

**Noise pipeline (`noise.py`):**
- `jnp.mgrid` output cast to float32
- `2π` precomputed as `jnp.float32(2.0 * jnp.pi)`
- `jnp.sqrt(2)` → `jnp.sqrt(jnp.float32(2.0))`

**Boundary enforcement:**
- `enforce_state_dtypes()` at end of `craftax_step` casts all scalar state fields to canonical dtypes (float32/int32), catching any `bool * Python_int` promotion that slips through
- `reward` explicitly cast to `jnp.float32` before return

### Verification

```bash
uv run python test_x64.py
```

The test enables x64, runs 500 steps for all four variants (extended + classic, symbolic + pixels), and asserts:
- `obs.dtype == float32`, `reward.dtype == float32`, `discount.dtype == float32`
- Every leaf in the state pytree is float32, int32, or bool (no float64/int64)
- Zero `FutureWarning` from JAX scatter (treated as errors)

## Gotchas

- **Texture cache**: Pixel renderer caches compiled textures as `texture_cache.pbz2`. If you edit rendering assets/code, set `CRAFTAX_RELOAD_TEXTURES=true` to force regeneration. Also regenerate after toggling x64 since texture dtypes differ.
- **Pre-commit**: Black formatter runs on commit via pre-commit hooks.
- **All game state is immutable JAX arrays** — modifications use Flax's `.replace()` on dataclasses, not mutation.
- **Adding new game logic**: Any new `jax.random.*` call must include explicit `dtype=`. Any bool array passed as `p=` to `jax.random.choice` must be `.astype(jnp.float32)` first. Any Python float literal in JAX arithmetic must be wrapped with `jnp.float32(...)`. All `jnp.arange()` calls must include `dtype=jnp.int32`. The `enforce_state_dtypes` boundary will catch scalar state field issues, but array fields and intermediate computations must be correct at the source.
- **JAX weak-type safety**: Python int/float literals are "weak types" in JAX — they adapt to the strong-typed JAX array in the expression (`float32 * 0.5` stays float32). But `jnp.arange()`, `jnp.array([ints])`, and `int32_array.sum()` produce strong-typed int64 under x64. Always use explicit `dtype=` on array constructors and be cautious with `.sum()` results feeding into integer arithmetic.
