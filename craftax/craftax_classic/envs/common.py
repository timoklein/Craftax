from craftax.craftax_classic.constants import *
from craftax.craftax_classic.envs.craftax_state import EnvState


def compute_score(state: EnvState, done: bool):
    achievements = state.achievements * done * jnp.float32(100.0)
    info = {}
    for achievement in Achievement:
        name = f"Achievements/{achievement.name.lower()}"
        info[name] = achievements[achievement.value]
    # Geometric mean with an offset of 1%
    info["score"] = jnp.exp(jnp.mean(jnp.log(jnp.float32(1) + achievements))) - jnp.float32(1.0)
    return info
