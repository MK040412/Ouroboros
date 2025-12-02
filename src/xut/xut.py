from typing import Optional, Sequence, Tuple, List
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

from .modules.norm import RMSNorm
from .modules.transformer import TransformerBlock
from .modules.patch import PatchEmbed, UnPatch
from .modules.axial_rope import make_axial_pos
from .modules.time_emb import TimestepEmbedding
from .modules.norm import RMSNorm #, DyT

def _isiterable(x) -> bool:
    try:
        iter(x)
    except TypeError:
        return False
    return True

class _SharedAdaLNHead(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.dim * 4)(y)
        y = jax.nn.mish(y)
        y = nn.Dense(self.dim * 3,
                     kernel_init=nn.initializers.zeros,
                     bias_init=nn.initializers.zeros)(y)
        return y  # (B, dim*3)

def _chunk3(v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    d3 = v.shape[-1]
    assert d3 % 3 == 0
    d = d3 // 3
    return v[..., :d], v[..., d:2*d], v[..., 2*d:]

class TBackBone(nn.Module):
    dim: int = 1024
    ctx_dim: Optional[int] = 1024
    heads: int = 16
    dim_head: int = 64
    mlp_dim: int = 3072
    pos_dim: int = 2
    depth: int = 8
    use_adaln: bool = False
    use_shared_adaln: bool = False
    use_dyt: bool = False
    grad_ckpt: bool = False  # remat

    def setup(self):
        norm_layer = RMSNorm
        block = partial(
            TransformerBlock,
            dim=self.dim,
            ctx_dim=self.ctx_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            mlp_dim=self.mlp_dim,
            pos_dim=self.pos_dim,
            use_adaln=self.use_adaln,
            use_shared_adaln=self.use_shared_adaln,
            norm_layer=norm_layer,
        )
        if self.grad_ckpt:
            self.blocks = [nn.remat(block()) for _ in range(self.depth)]
        else:
            self.blocks = [block() for _ in range(self.depth)]

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

class XUTBackBone(nn.Module):
    dim: int = 1024
    ctx_dim: Optional[int] = None
    heads: int = 16
    dim_head: int = 64
    mlp_dim: int = 3072
    pos_dim: int = 2
    depth: int = 8
    enc_blocks: int | Sequence[int] = 1
    dec_blocks: int | Sequence[int] = 2
    dec_ctx: bool = False
    use_adaln: bool = False
    use_shared_adaln: bool = False
    use_dyt: bool = False
    grad_ckpt: bool = False

    def setup(self):
        norm_layer = RMSNorm

        if _isiterable(self.enc_blocks):
            enc_list = list(self.enc_blocks)
            assert len(enc_list) == self.depth
        else:
            enc_list = [int(self.enc_blocks)] * self.depth

        if _isiterable(self.dec_blocks):
            dec_list = list(self.dec_blocks)
            assert len(dec_list) == self.depth
        else:
            dec_list = [int(self.dec_blocks)] * self.depth

        mk_block = lambda ctx_dim, ctx_from_self=False: TransformerBlock(
            dim=self.dim,
            ctx_dim=ctx_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            mlp_dim=self.mlp_dim,
            pos_dim=self.pos_dim,
            use_adaln=self.use_adaln,
            use_shared_adaln=self.use_shared_adaln,
            ctx_from_self=ctx_from_self,
            norm_layer=norm_layer,
        )

        self.enc_stacks = tuple([
            tuple([
                mk_block(self.ctx_dim)
                for _ in range(enc_list[i])
            ])
            for i in range(self.depth)
        ])

        self.dec_stacks = tuple([
            tuple([
                mk_block(self.dim, ctx_from_self=True) if bid == 0
                else mk_block(self.ctx_dim if self.dec_ctx else None)
                for bid in range(dec_list[i])
            ])
            for i in range(self.depth)
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

class XUDiT(nn.Module):
    patch_size: int = 2
    input_dim: int = 4
    dim: int = 1024
    ctx_dim: Optional[int] = 1024
    ctx_size: int = 256
    heads: int = 16
    dim_head: int = 64
    mlp_dim: int = 3072
    depth: int = 8
    enc_blocks: int | Sequence[int] = 1
    dec_blocks: int | Sequence[int] = 2
    dec_ctx: bool = False
    class_cond: int = 0
    shared_adaln: bool = True
    concat_ctx: bool = True
    use_dyt: bool = False
    double_t: bool = False
    addon_info_embs_dim: Optional[int] = None
    tread_config: Optional[dict] = None   # {"dropout_ratio": float, "prev_trns_depth": int, "post_trns_depth": int}
    grad_ckpt: bool = False

    def setup(self):
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
        )

        # TREAD
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
            )

        self.patch_size_ = self.patch_size
        self.in_patch = PatchEmbed(patch_size=self.patch_size,
                                   in_channels=self.input_dim,
                                   embed_dim=self.dim)
        self.out_patch = UnPatch(patch_size=self.patch_size,
                                 input_dim=self.dim,
                                 out_channels=self.input_dim)

        self.time_emb = TimestepEmbedding(self.dim)
        if self.double_t:
            self.r_emb = TimestepEmbedding(self.dim)

        if self.shared_adaln:
            self.shared_adaln_attn = _SharedAdaLNHead(self.dim)
            self.shared_adaln_xattn = _SharedAdaLNHead(self.dim)
            self.shared_adaln_ffw = _SharedAdaLNHead(self.dim)

        if self.class_cond > 0:
            self.class_token = nn.Embed(num_embeddings=self.class_cond, features=self.dim)
        else:
            self.class_token = None

        if self.concat_ctx and (self.ctx_dim is not None):
            self.ctx_proj = nn.Dense(self.dim)
        else:
            self.ctx_proj = None

        if self.addon_info_embs_dim is not None:
            self.addon_info_embs_proj_1 = nn.Dense(self.dim)
            self.addon_info_embs_proj_2 = nn.Dense(self.dim,
                                                   kernel_init=nn.initializers.zeros,
                                                   bias_init=nn.initializers.zeros)

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
                x_seq, ctx=ctx, pos_map=pos_map_resized, y=t_emb,
                shared_adaln=shared_adaln_state, deterministic=deterministic
            )
            do_tread = (not deterministic) or (tread_rate is not None)
            if do_tread:
                rate = (tread_rate if tread_rate is not None else self.dropout_ratio)
                keep_len = length - int(length * rate)
                key = self.make_rng("dropout")
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
            deterministic=deterministic
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
                out, ctx=ctx, pos_map=pos_map_resized, y=t_emb,
                shared_adaln=shared_adaln_state, deterministic=deterministic
            )

        out = out[:, :length]
        img = self.out_patch(out, axis1=H, axis2=W)
        return img if not return_enc_out else (img, enc_out[:, :length])
