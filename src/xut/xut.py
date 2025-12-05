from typing import Optional, Sequence, Tuple, List

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import RMSNorm

# Compatibility shim for older Flax versions without nnx.List
if not hasattr(nnx, 'List'):
    # Use tuple as fallback (pytree compatible)
    nnx.List = lambda x: tuple(x)

from .modules.transformer import TransformerBlock
from .modules.patch import PatchEmbed, UnPatch
from .modules.axial_rope import make_axial_pos
from .modules.time_emb import TimestepEmbedding

def _isiterable(x) -> bool:
    try:
        iter(x)
    except TypeError:
        return False
    return True

class _SharedAdaLNHead(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.rnglib.Rngs):
        self.dim = dim
        self.norm = nnx.LayerNorm(num_features=dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, dim * 4, rngs=rngs)
        self.fc2 = nnx.Linear(
            dim * 4,
            dim * 3,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self.norm(x)
        y = self.fc1(y)
        y = jax.nn.mish(y)
        y = self.fc2(y)
        return y  # (B, dim*3)

def _chunk3(v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    d3 = v.shape[-1]
    assert d3 % 3 == 0
    d = d3 // 3
    return v[..., :d], v[..., d:2*d], v[..., 2*d:]

class TBackBone(nnx.Module):
    def __init__(
        self,
        dim: int = 1024,
        ctx_dim: Optional[int] = 1024,
        heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 3072,
        pos_dim: int = 2,
        depth: int = 8,
        use_adaln: bool = False,
        use_shared_adaln: bool = False,
        use_dyt: bool = False,
        grad_ckpt: bool = False,  # remat (not used)
        *,
        rngs: nnx.rnglib.Rngs,
    ):
        norm_layer = RMSNorm
        self.use_adaln = use_adaln
        self.use_shared_adaln = use_shared_adaln
        self.blocks = tuple([
            TransformerBlock(
                dim=dim,
                ctx_dim=ctx_dim,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                pos_dim=pos_dim,
                use_adaln=use_adaln,
                use_shared_adaln=use_shared_adaln,
                norm_layer=norm_layer,
                rngs=rngs,
            )
            for _ in range(depth)
        ])

    def __call__(
        self,
        x: jnp.ndarray,
        ctx: Optional[jnp.ndarray] = None,
        x_mask: Optional[jnp.ndarray] = None,
        ctx_mask: Optional[jnp.ndarray] = None,
        pos_map: Optional[jnp.ndarray] = None,
        y: Optional[jnp.ndarray] = None,
        shared_adaln: Optional[Sequence[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        for block in self.blocks:
            x = block(
                x,
                ctx=ctx,
                pos_map=pos_map,
                ctx_pos_map=None,
                y=y,
                x_mask=x_mask,
                ctx_mask=ctx_mask,
                shared_adaln=shared_adaln,
                deterministic=deterministic
            )
        return x

class XUTBackBone(nnx.Module):
    def __init__(
        self,
        dim: int = 1024,
        ctx_dim: Optional[int] = None,
        heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 3072,
        pos_dim: int = 2,
        depth: int = 8,
        enc_blocks: int | Sequence[int] = 1,
        dec_blocks: int | Sequence[int] = 2,
        dec_ctx: bool = False,
        use_adaln: bool = False,
        use_shared_adaln: bool = False,
        use_dyt: bool = False,
        grad_ckpt: bool = False,
        *,
        rngs: nnx.rnglib.Rngs,
    ):
        norm_layer = RMSNorm
        self.dim = dim
        self.ctx_dim = ctx_dim
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.pos_dim = pos_dim
        self.depth = depth
        self.dec_ctx = dec_ctx
        self.use_adaln = use_adaln
        self.use_shared_adaln = use_shared_adaln

        if _isiterable(enc_blocks):
            enc_list = list(enc_blocks)
            assert len(enc_list) == depth
        else:
            enc_list = [int(enc_blocks)] * depth

        if _isiterable(dec_blocks):
            dec_list = list(dec_blocks)
            assert len(dec_list) == depth
        else:
            dec_list = [int(dec_blocks)] * depth

        def mk_block(ctx_dim_inner, ctx_from_self: bool = False):
            return TransformerBlock(
                dim=dim,
                ctx_dim=ctx_dim_inner,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                pos_dim=pos_dim,
                use_adaln=use_adaln,
                use_shared_adaln=use_shared_adaln,
                ctx_from_self=ctx_from_self,
                norm_layer=norm_layer,
                rngs=rngs,
            )

        # Flax NNX 0.12.0+: nnx.List 사용 (pytree 호환성)
        self.enc_stacks = nnx.List([
            nnx.List([mk_block(ctx_dim) for _ in range(enc_list[i])])
            for i in range(depth)
        ])

        self.dec_stacks = nnx.List([
            nnx.List([
                mk_block(dim, ctx_from_self=True) if bid == 0
                else mk_block(ctx_dim if dec_ctx else None)
                for bid in range(dec_list[i])
            ])
            for i in range(depth)
        ])

    def __call__(
        self,
        x: jnp.ndarray,
        ctx: Optional[jnp.ndarray] = None,
        x_mask: Optional[jnp.ndarray] = None,
        ctx_mask: Optional[jnp.ndarray] = None,
        pos_map: Optional[jnp.ndarray] = None,
        y: Optional[jnp.ndarray] = None,
        shared_adaln: Optional[Sequence[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]] = None,
        return_enc_out: bool = False,
        deterministic: bool = True,
    ) -> jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]:
        self_ctx: List[jnp.ndarray] = []
        for blocks in self.enc_stacks:
            for block in blocks:
                x = block(x,
                        ctx=ctx,
                        pos_map=pos_map,
                        ctx_pos_map=None,
                        y=y,
                        x_mask=x_mask,
                        ctx_mask=ctx_mask,
                        shared_adaln=shared_adaln,
                        deterministic=deterministic)
            self_ctx.append(x)
        enc_out = x

        for blocks in self.dec_stacks:
            # first block cross attends to last encoder output
            first = blocks[0]
            x = first(x,
                      ctx=self_ctx[-1],
                      pos_map=pos_map,
                      ctx_pos_map=pos_map,
                      y=y,
                      x_mask=x_mask,
                      ctx_mask=ctx_mask,
                      shared_adaln=shared_adaln,
                      deterministic=deterministic)
            for block in blocks[1:]:
                x = block(x,
                        ctx=ctx if self.dec_ctx else None,
                        pos_map=pos_map,
                        ctx_pos_map=None,
                        y=y,
                        x_mask=x_mask,
                        ctx_mask=ctx_mask,
                        shared_adaln=shared_adaln,
                        deterministic=deterministic)

        if return_enc_out:
            return x, enc_out
        return x

class XUDiT(nnx.Module):
    def __init__(
        self,
        patch_size: int = 2,
        input_dim: int = 4,
        dim: int = 1024,
        ctx_dim: Optional[int] = 1024,
        ctx_size: int = 256,
        heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 3072,
        depth: int = 8,
        enc_blocks: int | Sequence[int] = 1,
        dec_blocks: int | Sequence[int] = 2,
        dec_ctx: bool = False,
        class_cond: int = 0,
        shared_adaln: bool = True,
        concat_ctx: bool = True,
        use_dyt: bool = False,
        double_t: bool = False,
        addon_info_embs_dim: Optional[int] = None,
        tread_config: Optional[dict] = None,
        grad_ckpt: bool = False,
        *,
        rngs: nnx.rnglib.Rngs,
    ):
        self.rngs = rngs
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.dim = dim
        self.ctx_dim = ctx_dim
        self.ctx_size = ctx_size
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.depth = depth
        self.enc_blocks = enc_blocks
        self.dec_blocks = dec_blocks
        self.dec_ctx = dec_ctx
        self.class_cond = class_cond
        self.shared_adaln = shared_adaln
        self.concat_ctx = concat_ctx
        self.use_dyt = use_dyt
        self.double_t = double_t
        self.addon_info_embs_dim = addon_info_embs_dim
        self.tread_config = tread_config
        self.grad_ckpt = grad_ckpt

        self.backbone = XUTBackBone(
            dim=self.dim,
            ctx_dim=(None if self.concat_ctx else self.ctx_dim),
            heads=self.heads,
            dim_head=self.dim_head,
            mlp_dim=self.mlp_dim,
            pos_dim=2,
            depth=self.depth,
            enc_blocks=self.enc_blocks,
            dec_blocks=self.dec_blocks,
            dec_ctx=self.dec_ctx,
            use_adaln=True,
            use_shared_adaln=self.shared_adaln,
            use_dyt=self.use_dyt,
            grad_ckpt=self.grad_ckpt,
            rngs=rngs,
        )

        self.use_tread = self.tread_config is not None
        if self.use_tread:
            dr = float(self.tread_config["dropout_ratio"])
            prev_d = int(self.tread_config["prev_trns_depth"])
            post_d = int(self.tread_config["post_trns_depth"])
            self.dropout_ratio = dr
            self.prev_tread_trns = TBackBone(
                dim=self.dim,
                ctx_dim=(None if self.concat_ctx else self.ctx_dim),
                heads=self.heads,
                dim_head=self.dim_head,
                mlp_dim=self.mlp_dim,
                pos_dim=2,
                depth=prev_d,
                use_adaln=True,
                use_shared_adaln=self.shared_adaln,
                use_dyt=self.use_dyt,
                grad_ckpt=self.grad_ckpt,
                rngs=rngs,
            )
            self.post_tread_trns = TBackBone(
                dim=self.dim,
                ctx_dim=(None if self.concat_ctx else self.ctx_dim),
                heads=self.heads,
                dim_head=self.dim_head,
                mlp_dim=self.mlp_dim,
                pos_dim=2,
                depth=post_d,
                use_adaln=True,
                use_shared_adaln=self.shared_adaln,
                use_dyt=self.use_dyt,
                grad_ckpt=self.grad_ckpt,
                rngs=rngs,
            )

        self.patch_size_ = self.patch_size
        self.in_patch = PatchEmbed(
            patch_size=self.patch_size,
            in_channels=self.input_dim,
            embed_dim=self.dim,
            rngs=rngs,
        )
        self.out_patch = UnPatch(
            patch_size=self.patch_size,
            input_dim=self.dim,
            out_channels=self.input_dim,
            rngs=rngs,
        )

        self.time_emb = TimestepEmbedding(self.dim, rngs=rngs)
        if self.double_t:
            self.r_emb = TimestepEmbedding(self.dim, rngs=rngs)

        if self.shared_adaln:
            self.shared_adaln_attn = _SharedAdaLNHead(self.dim, rngs=rngs)
            self.shared_adaln_xattn = _SharedAdaLNHead(self.dim, rngs=rngs)
            self.shared_adaln_ffw = _SharedAdaLNHead(self.dim, rngs=rngs)

        if self.class_cond > 0:
            self.class_token = nnx.Embed(num_embeddings=self.class_cond, features=self.dim, rngs=rngs)
        else:
            self.class_token = None

        if self.concat_ctx and (self.ctx_dim is not None):
            self.ctx_proj = nnx.Linear(self.ctx_dim, self.dim, rngs=rngs)
        else:
            self.ctx_proj = None

        if self.addon_info_embs_dim is not None:
            self.addon_info_embs_proj_1 = nnx.Linear(self.addon_info_embs_dim, self.dim, rngs=rngs)
            self.addon_info_embs_proj_2 = nnx.Linear(
                self.dim,
                self.dim,
                kernel_init=nnx.initializers.zeros_init(),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )

    def _make_shared_adaln_state(self, t_emb: jnp.ndarray):
        attn = self.shared_adaln_attn(t_emb)
        xattn = self.shared_adaln_xattn(t_emb)
        ffw = self.shared_adaln_ffw(t_emb)
        return [_chunk3(attn), _chunk3(xattn), _chunk3(ffw)]

    def __call__(self,
        x: jnp.ndarray,                           # NHWC (B, H, W, C)
        t: jnp.ndarray,                           # (B,) | (B, 1)
        ctx: Optional[jnp.ndarray] = None,        # (B, T_ctx, ctx_dim) | None | (B,) class ids if class_cond>0
        pos_map: Optional[jnp.ndarray] = None,    # (B, N, 2)
        r: Optional[jnp.ndarray] = None,
        addon_info: Optional[jnp.ndarray] = None, # (B, D_addon) or (B,)
        tread_rate: Optional[float] = None,
        return_enc_out: bool = False,
        ctx_mask: Optional[jnp.ndarray] = None,   # (B, T_ctx) optional attention mask for context
        deterministic: bool = True,
    ):
        """
        -> (B, C, H, W)
        """
        B, H, W, C = x.shape
        x_seq, pos_map_resized = self.in_patch(x, pos_map)  # x_seq: (B, N, D), pos_map_resized: (B, N, 2) or None
        N = x_seq.shape[1]
        if pos_map_resized is None:
            pos_map_resized = make_axial_pos(H // self.patch_size_,
                                             W // self.patch_size_)
            pos_map_resized = jnp.broadcast_to(pos_map_resized[None, ...],
                                               (B, pos_map_resized.shape[0], pos_map_resized.shape[1]))

        t_emb = self.time_emb(t)
        if r is not None:
            r = jnp.reshape(r, (B, -1))
            t_emb = t_emb + self.r_emb((t.reshape(B, -1) - r))

        if (self.class_token is not None) and (ctx is not None):
            if ctx.ndim == 1:
                cls_ids = ctx
            else:
                cls_ids = ctx.reshape((ctx.shape[0], -1))[:, 0]
            t_emb = t_emb + self.class_token(cls_ids)
            ctx = None

        if addon_info is not None:
            if addon_info.ndim == 1:
                addon_info = addon_info[:, None]
            addon_embs = jax.nn.mish(self.addon_info_embs_proj_1(addon_info))
            addon_embs = self.addon_info_embs_proj_2(addon_embs)
            addon_embs = addon_embs[:, None, :]  # (B,1,D)
            t_emb = t_emb + addon_embs

        need_ctx = (self.ctx_dim is not None)
        if (ctx is None) and need_ctx and (not self.concat_ctx):
            ctx = jnp.zeros((B, self.ctx_size, self.ctx_dim), dtype=x_seq.dtype)

        shared_adaln_state = None
        if self.shared_adaln:
            shared_adaln_state = self._make_shared_adaln_state(t_emb)

        length = x_seq.shape[1]

        if self.ctx_proj is not None and ctx is not None:
            ctx_proj = self.ctx_proj(ctx)
            x_seq = jnp.concatenate([x_seq, ctx_proj], axis=1)
            pad_pos = jnp.zeros((B, ctx_proj.shape[1], pos_map_resized.shape[-1]), dtype=pos_map_resized.dtype)
            pos_map_resized = jnp.concatenate([pos_map_resized, pad_pos], axis=1)
            ctx = None

        # TREAD (pre)
        if self.use_tread:
            x_seq = self.prev_tread_trns(
                x_seq, ctx=ctx, pos_map=pos_map_resized, y=t_emb, ctx_mask=ctx_mask,
                shared_adaln=shared_adaln_state, deterministic=deterministic
            )
            do_tread = (not deterministic) or (tread_rate is not None)
            if do_tread:
                rate = (tread_rate if tread_rate is not None else self.dropout_ratio)
                keep_len = length - int(length * rate)
                key = self.rngs.dropout()
                perm = jax.vmap(lambda k: jax.random.permutation(k, length))(jax.random.split(key, B))
                sel_mask = perm < keep_len
                if self.ctx_proj is not None:
                    ctx_len = x_seq.shape[1] - length
                    ctx_keep = jnp.ones((B, ctx_len), dtype=sel_mask.dtype)
                    sel_mask = jnp.concatenate([sel_mask, ctx_keep], axis=1)
                    keep_len = keep_len + ctx_len

                def _gather_sel(x_row, m_row):
                    idx = jnp.nonzero(m_row, size=m_row.shape[0], fill_value=0)[0]
                    idx = idx[:keep_len]
                    return x_row[idx]
                x_not_sel = jnp.where(sel_mask[..., None], jnp.zeros_like(x_seq), x_seq)
                x_sel = jax.vmap(_gather_sel)(x_seq, sel_mask)
                x_seq = x_sel
                raw_pos = pos_map_resized
                pos_map_resized = jax.vmap(_gather_sel)(pos_map_resized, sel_mask)

        # Backbone
        out = self.backbone(
            x_seq, ctx=ctx, pos_map=pos_map_resized, y=t_emb,
            shared_adaln=shared_adaln_state, return_enc_out=return_enc_out,
            deterministic=deterministic, ctx_mask=ctx_mask
        )
        if return_enc_out:
            out, enc_out = out

        # TREAD (post)
        if self.use_tread:
            if (not deterministic) or (tread_rate is not None):
                full_len = (raw_pos.shape[1])
                def _scatter_back(x_row, sel_mask_row):
                    idx = jnp.nonzero(sel_mask_row, size=sel_mask_row.shape[0], fill_value=0)[0]
                    idx = idx[:x_row.shape[0]]
                    base = jnp.zeros((full_len, x_row.shape[-1]), dtype=x_row.dtype)
                    return base.at[idx].set(x_row)
                out_full = jax.vmap(_scatter_back)(out, sel_mask)
                out = out_full
                pos_map_resized = raw_pos

            out = self.post_tread_trns(
                out, ctx=ctx, pos_map=pos_map_resized, y=t_emb, ctx_mask=ctx_mask,
                shared_adaln=shared_adaln_state, deterministic=deterministic
            )

        out = out[:, :length]
        img = self.out_patch(out, axis1=H, axis2=W)
        return img if not return_enc_out else (img, enc_out[:, :length])
