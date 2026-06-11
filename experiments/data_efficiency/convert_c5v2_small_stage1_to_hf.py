"""Convert C5-v2 small stage-1 (step-6400) to HF format."""

import os, sys
from pathlib import Path

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

from experiments.data_efficiency.models import model_dict
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import load_tokenizer

SRC = "/fsx/users/dongweij/marin/checkpoints/1_4b_c5v2_small/5hb7vl3u/step-6400"
DST = "/fsx/users/dongweij/marin/checkpoints/c5v2_small_stage1_step6400_hf"


def main():
    mc = model_dict["1_4b4k"]
    tok = load_tokenizer("meta-llama/Meta-Llama-3.1-8B")
    Vocab = hax.Axis("vocab", len(tok))
    mesh = jax.sharding.Mesh(jax.devices("cpu")[:1], ("data",))
    print(f"=== c5v2_small_stage1: {SRC} -> {DST} ===", flush=True)
    with hax.partitioning.set_mesh(mesh):
        model = eqx.filter_eval_shape(mc.build, Vocab, key=jax.random.PRNGKey(0))
        model = load_checkpoint(model, SRC, subpath="model")
        cv = mc.hf_checkpoint_converter().replaced(tokenizer=tok)
        cv.save_pretrained(model, DST)
        tok.save_pretrained(DST)
    print(f"   ✓ saved to {DST}", flush=True)


if __name__ == "__main__":
    main()
