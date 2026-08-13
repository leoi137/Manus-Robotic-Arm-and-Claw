# A guided tour of the ACT model our Stage-1 run actually uses

*Written 2026-08-12, against the code that is installed and running, not against
the paper.*

Every file path below is inside the venv on the rented box:

```
/root/.venv-lerobot/lib/python3.12/site-packages/lerobot/policies/act/
    modeling_act.py       ← the network
    configuration_act.py  ← the hyperparameters
    processor_act.py      ← normalization / batching pipeline
```

Line numbers are from **lerobot 0.6.1**. If you upgrade, re-check them.

## The shapes, once, for our data

These are the numbers to hold in your head for the whole tour. They come from
`datasets/lerobot/grasp_*_v2/meta/info.json` and our launch flags.

| symbol | value | where it comes from |
|---|---|---|
| `B` (batch) | **64** | measured by the VRAM probe in `scripts/train_act.py` |
| cameras | **1** (`observation.images.wrist`) | the only image feature in the datasets |
| image | **3 x 240 x 320** | recorded at 320x240, downscaled from 640x480 |
| state dim | **6** | the SO-101's six joints |
| action dim | **6** | same six joints, as *commanded* positions |
| `chunk_size` | **50** | 1.67 s at 30 Hz (see `train_act.py`'s docstring) |
| `dim_model` `D` | **512** | `configuration_act.py:103` |
| `latent_dim` | **32** | `configuration_act.py:114` |

No resizing or cropping happens anywhere: ACT has no `resize`/`crop` config and
our frames are already 240x320. The images arrive as float in [0, 1].

---

## Stage 0 — the entry point and the loss

**`ACTPolicy.forward`** — `modeling_act.py:137-165`.

This is the function `scripts/train_act.py` calls every step. It does three
things:

1. **`:139-141`** gathers the image features into a list under one key:
   `batch[OBS_IMAGES] = [batch["observation.images.wrist"]]`. One camera, so a
   one-element list. (This is the seam where a second, third-person camera would
   drop in for free — see the note at the end.)
2. **`:143`** runs the network: `actions_hat, (mu, log_sigma_x2) = self.model(batch)`.
3. **`:145-164`** computes the loss.

The loss is worth reading closely, because it is the number you will be staring
at for the next twelve hours:

```python
abs_err   = F.l1_loss(batch[ACTION], actions_hat, reduction="none")  # (B, 50, 6)
valid_mask = ~batch["action_is_pad"].unsqueeze(-1)                   # (B, 50, 1)
l1_loss   = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)    # scalar
...
loss = l1_loss + mean_kld * self.config.kl_weight   # kl_weight = 10.0
```

* **L1, not MSE** (`:145`). L2 punishes outliers quadratically, and in
  demonstration data the outliers are usually a *different valid strategy*, not
  an error. L1 regresses toward the median motion instead of the mean, which for
  multi-modal demos is the one that actually reaches the object.
* **`action_is_pad` masking** (`:146-148`). Near the end of an episode there are
  fewer than 50 real future actions. LeRobot pads them and flags them; those
  positions are excluded from both the numerator and the denominator. Without
  this, the policy would be trained to predict padding.
* **`kl_weight = 10.0`** (`:161`). The KL term is *ten times* weighted, which is
  why the total loss is dominated by KL early in the run and why `l1_loss` is
  the column you should actually plot. `train_act.py` logs both to
  `train_curve.csv` for exactly this reason.

---

## Stage 1 — the CVAE encoder (train time only)

**`ACT.forward`** — `modeling_act.py:407-451`. Modules built at `:299-322`.

Guarded by `if self.config.use_vae and ACTION in batch and self.training`
(`:407`) — so this entire stage **does not exist at inference**, and does not
exist during our validation pass either. That is why val loss is pure L1.

It is a small BERT-style transformer encoder over the *answer*:

| step | line | tensor | shape |
|---|---|---|---|
| CLS token, repeated over batch | `:409` | `cls_embed` | `(64, 1, 512)` |
| state → D | `:413` | `robot_state_embed` | `(64, 1, 512)` |
| the ground-truth chunk → D | `:415` | `action_embed` | `(64, 50, 512)` |
| concatenated | `:421` | `vae_encoder_input` | `(64, 52, 512)` |
| fixed sinusoidal pos. enc. | `:425` | `pos_embed` | `(1, 52, 512)` |
| encoder, take CLS output only | `:440-444` | `cls_token_out` | `(64, 512)` |
| → latent distribution params | `:445` | `latent_pdf_params` | `(64, 64)` |
| split | `:446-448` | `mu`, `log_sigma_x2` | `(64, 32)` each |
| reparameterization trick | `:451` | `latent_sample` | `(64, 32)` |

Note the transformer here runs **sequence-first**: `.permute(1, 0, 2)` at `:441`
turns `(B, S, D)` into `(S, B, D)`. All of lerobot's ACT transformer code is
`(S, B, D)`; only the outside world is batch-first.

**The thing to understand:** this encoder gets to *cheat*. It sees the answer.
Its job is to compress "which of the many valid ways of doing this was actually
demonstrated here" into 32 numbers. The decoder then gets those 32 numbers for
free, so it is not punished for failing to predict the unpredictable. The KL
term stops it from cheating *too* much — if `z` were unconstrained the model
would just encode the whole answer into it and learn nothing from the image.

At inference (`:452-458`) `latent_sample` is **exactly zeros** — the mode of the
prior — which yields the single most typical motion for the observation.

---

## Stage 2 — the ResNet backbone → image tokens

**Built at `modeling_act.py:325-334`, run at `:474-486`.**

```python
backbone_model = getattr(torchvision.models, config.vision_backbone)(   # resnet18
    replace_stride_with_dilation=[False, False, False],
    weights="ResNet18_Weights.IMAGENET1K_V1",
    norm_layer=FrozenBatchNorm2d,
)
self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
```

* `IntermediateLayerGetter` (`:334`) **throws away the classifier head** — we
  want the spatial feature map from `layer4`, not 1000 ImageNet logits. There is
  no global pooling: pooling would destroy *where* the cube is, which is the
  only thing we need.
* `FrozenBatchNorm2d` (`:329`) freezes the ImageNet BatchNorm running statistics.
  At batch 64 split across one camera the per-batch statistics are noisy, and
  DETR-lineage code has always frozen them.

The per-camera loop, with our shapes:

| step | line | tensor | shape |
|---|---|---|---|
| input frame | — | `img` | `(64, 3, 240, 320)` |
| ResNet-18 `layer4` (total stride 32) | `:475` | `cam_features` | `(64, 512, 8, 10)` |
| 2-D sinusoidal position embedding | `:476` | `cam_pos_embed` | `(64, 512, 8, 10)` |
| 1x1 conv, 512 → `dim_model` | `:477` | `cam_features` | `(64, 512, 8, 10)` |
| flatten the grid into a sequence | `:480` | `(h w) b c` | `(80, 64, 512)` |

**240/32 = 7.5 → 8 and 320/32 = 10, so the image becomes 80 tokens.** Each token
is one cell of an 8x10 grid over the wrist image, roughly a 30x32-pixel patch.
The 1x1 conv at `:477` is a no-op in shape for ResNet-18 (512 → 512) but is
there because ResNet-50 would arrive with 2048 channels.

The 2-D sinusoidal embedding (`ACTSinusoidalPositionEmbedding2d`,
`modeling_act.py:686-739`) is what tells attention that token 0 is top-left and
token 79 is bottom-right. Without it the encoder would see an unordered bag of
80 patches.

---

## Stage 3 — the transformer encoder

**Assembled at `modeling_act.py:461-493`. Classes: `ACTEncoder` (`:515-533`),
`ACTEncoderLayer` (`:534-572`).**

The token sequence is built by concatenation (`:461-490`):

| # tokens | what | line |
|---|---|---|
| 1 | latent `z`, projected 32 → 512 | `:461` |
| 1 | robot state, projected 6 → 512 | `:465` |
| 80 | image patches | `:485` |
| **82** | **total** | `:489` |

so `encoder_in_tokens` is **`(82, 64, 512)`**. The first two get *learned*
positional embeddings (`encoder_1d_feature_pos_embed`, `:361`); the 80 image
tokens get the sinusoidal 2-D ones. Then:

```python
encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)  # (82, 64, 512)
```

Four standard post-norm encoder layers (`n_encoder_layers = 4`,
`configuration_act.py:107`), 8 heads, FFN width 3200, dropout 0.1. Each layer is
self-attention → FFN with residuals (`ACTEncoderLayer.forward`, `:552-571`).
Note `maybe_add_pos_embed`: the positional embedding is re-added to the query and
key at **every layer**, DETR-style, rather than once at the input.

**What this stage is for:** it fuses the three sources. After four layers, the
state token knows what the image sees and the image tokens know where the arm
is. `(82, 64, 512)` in, `(82, 64, 512)` out — this is a pure "understand the
scene" stage; nothing about actions has happened yet.

---

## Stage 4 — the transformer decoder (the DETR trick)

**`modeling_act.py:495-508`. Classes: `ACTDecoder` (`:573-595`),
`ACTDecoderLayer` (`:596-667`).**

```python
decoder_in = torch.zeros((50, 64, 512))            # :495-499  literally zeros
decoder_out = self.decoder(
    decoder_in, encoder_out,
    encoder_pos_embed=encoder_in_pos_embed,
    decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),   # (50, 1, 512)
)                                                   # (50, 64, 512)
decoder_out = decoder_out.transpose(0, 1)           # :508  → (64, 50, 512)
```

This is the part that surprises people coming from language models. **There is
no autoregression and no causal mask.** The decoder input is all zeros; all of
the information about "which timestep am I?" lives in
`decoder_pos_embed` (`nn.Embedding(chunk_size, dim_model)`, `:367`) — 50 learned
query vectors, exactly DETR's object queries, except ours are *timestep*
queries. Query *i* is trained to mean "the action 1/30 s x *i* into the future".

Each of the 50 queries then does, per layer (`ACTDecoderLayer.forward`,
`:620-666`):
1. self-attention across the 50 queries — so the chunk is temporally coherent
   rather than 50 independent guesses;
2. cross-attention into the 82 encoder tokens — this is where a query actually
   *looks at the cube*;
3. FFN.

All 50 timesteps are produced **in one parallel pass**. `n_decoder_layers = 1`
(`configuration_act.py:111`) — the config comments that the original ACT's 7
layers were partly a bug, and 1 is what lerobot ships.

---

## Stage 5 — the action head

**`modeling_act.py:370` (built), `:510` (run).**

```python
self.action_head = nn.Linear(config.dim_model, 6)   # 512 → 6
actions = self.action_head(decoder_out)             # (64, 50, 512) → (64, 50, 6)
```

One linear layer. That is the whole head. The output is 50 future joint-position
commands per batch element, in **normalized** units — the postprocessor
(`processor_act.py`, built by `make_pre_post_processors`) multiplies the dataset
statistics back in to get real joint angles.

---

## The whole forward pass in one column

```
observation.images.wrist  (64, 3, 240, 320)
observation.state         (64, 6)
action (train only)       (64, 50, 6)
        │
        ├── VAE encoder ── (64, 52, 512) ── CLS ── (64, 512) ── μ,logσ² (64, 32) ── z (64, 32)
        │                                                            [z = 0 at inference]
        ├── ResNet-18 layer4 ── (64, 512, 8, 10) ── flatten ── (80, 64, 512)
        │
        └── tokens = [z(1)] + [state(1)] + [image(80)] = (82, 64, 512)
                 │
            encoder x4                        (82, 64, 512)
                 │
            decoder x1  ← 50 learned timestep queries
                 │                            (50, 64, 512)
            action_head (Linear 512→6)
                 │
            actions                           (64, 50, 6)
                 │
            L1 vs ground truth (padding-masked) + 10.0 * KL
```

Roughly **51.6M parameters**: ~11M of ResNet-18 and ~40M of transformer.

---

## Two things to notice for our project

**1. One camera is the whole observation.** `config.image_features` has exactly
one entry, so the loop at `:474` runs once and the encoder gets 82 tokens. If a
third-person camera is ever added to the datasets, *nothing in the model needs
to change*: the loop runs twice, the sequence becomes 162 tokens, and encoder
memory grows roughly linearly. That is the single highest-leverage change
available if Stage 1 shows the policy cannot tell where the object is — see the
open question in `docs/TRAINING_PLAN.md`.

**2. `n_action_steps` vs `chunk_size`.** We set both to 50, meaning the policy
predicts 50 and (in `ACTPolicy.select_action`, `:101-125`) would execute all 50
open-loop from a queue. Our `scripts/eval_policy.py` deliberately does **not**
use that path — it requests a chunk every tick and blends the overlaps itself
(temporal ensembling, `ACTTemporalEnsembler` at `:167-256` is the reference
implementation it mirrors). `config.temporal_ensemble_coeff` stays `None`
because lerobot's built-in ensembler requires `n_action_steps == 1`, and the
ensembling lives on the eval client's side of the socket instead.
