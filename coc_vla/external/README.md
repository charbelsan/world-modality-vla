# External repos (for inspection)

This folder is used to **clone and inspect** relevant open repositories (JEPA, GR00T, Cosmos) while developing the CoC VLA line.

We do **not** vendor these repos into this project. Instead, we provide a helper script to clone them into `coc_vla/external/repos/` (which is gitignored).

## Fetch

```bash
bash coc_vla/external/fetch_external_repos.sh
```

## What gets cloned

- JEPA / V-JEPA2:
  - `facebookresearch/jepa`
  - `facebookresearch/vjepa2`
- GR00T:
  - `NVIDIA/Isaac-GR00T`
- Cosmos:
  - `nvidia-cosmos/cosmos-reason1`
  - `nvidia-cosmos/cosmos-predict2.5`

After cloning, read the top-level READMEs and configs to extract:

- encoder choices,
- training recipes,
- inference utilities,
- licensing constraints for weights.

