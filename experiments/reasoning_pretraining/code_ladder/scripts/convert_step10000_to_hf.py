# Convert step-10000 Levanter checkpoints to HF format for the 3 1.4B runs,
# so we can run gsm8k_cot looping test on each.

import os
import sys
from pathlib import Path

# Inject .secrets to env
SECRETS = Path("/fsx/users/dongweij/marin/.secrets")
if SECRETS.exists():
    for line in SECRETS.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, "/fsx/users/dongweij/marin")

import equinox as eqx
import haliax as hax
import jax

from experiments.reasoning_pretraining.code_ladder.models.models import model_dict
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import load_tokenizer

CONFIGS = [
    (
        "baseline_step10000",
        "/fsx/users/dongweij/marin/checkpoints/1_4b_wd1_6_x16_nocrossblock/6xx0hu3l/step-10000",
        "/fsx/users/dongweij/marin/checkpoints/baseline_step10000_hf",
    ),
    (
        "v1_step10000",
        "/fsx/users/dongweij/marin/checkpoints/1_4b_25code_alg/p2n84bo3/step-10000",
        "/fsx/users/dongweij/marin/checkpoints/v1_step10000_hf",
    ),
    (
        "v2_step10000",
        "/fsx/users/dongweij/marin/checkpoints/1_4b_25code_alg_v2/joqfahkl/step-10000",
        "/fsx/users/dongweij/marin/checkpoints/v2_step10000_hf",
    ),
]


def main() -> None:
    mc = model_dict["1_4b4k"]
    tok = load_tokenizer("meta-llama/Meta-Llama-3.1-8B")
    Vocab = hax.Axis("vocab", len(tok))
    mesh = jax.sharding.Mesh(jax.devices("cpu")[:1], ("data",))
    for label, src, dst in CONFIGS:
        print(f"=== {label}: {src} -> {dst} ===", flush=True)
        with hax.partitioning.set_mesh(mesh):
            model = eqx.filter_eval_shape(mc.build, Vocab, key=jax.random.PRNGKey(0))
            model = load_checkpoint(model, src, subpath="model")
            cv = mc.hf_checkpoint_converter().replaced(tokenizer=tok)
            cv.save_pretrained(model, dst)
            tok.save_pretrained(dst)
        print(f"   ✓ saved to {dst}", flush=True)


if __name__ == "__main__":
    main()
