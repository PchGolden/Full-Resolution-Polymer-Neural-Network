"""FRPN models for MD records with an explicit polymer graph."""
import torch
import torch.nn as nn
from .features import (
    AtomFeaturePlus,
    EdgeFeaturePlus,
    SE3InvariantKernel,
    MovementPredictionHead,
    MaskLMHead,
    ChainTokenFeaturePlus,
    ChainEdgeFeaturePlus
)
from frpn.common.encoder import EncoderBlock
from frpn.common.modeling import MonomerEncodingMixin


class MultiMolModel(MonomerEncodingMixin, nn.Module):
    """Encode monomers and the supplied polymer graph, with branch ablations."""

    def __init__(self, args):
        super().__init__()
        self.args = args  # Shared configuration, also serialized in checkpoints.
        self._fp_table_built = False

        # --------------------------------------------------------------
        # Stage-1: Embeddings & basic feature extractors
        # --------------------------------------------------------------
        d_model = getattr(args, "encoder_embed_dim", 768)
        self.embed_tokens = nn.Embedding(
            128, d_model, getattr(args, "padding_idx", 0)
        )

        self.atom_feature = AtomFeaturePlus(
            num_atom=getattr(args, "num_atom", 512),
            num_degree=getattr(args, "num_degree", 128),
            hidden_dim=d_model,
            wo_node=getattr(args, "wo_node", False),
            wo_atom_feat=getattr(args, "wo_atom_feat", None)

        )

        # Keep pair representation width consistent across all stage-1 modules.
        pair_embed_dim = getattr(args, "pair_embed_dim", None)
        if pair_embed_dim is None:
            pair_embed_dim = getattr(args, "pair_dim", 512)

        self.edge_feature = EdgeFeaturePlus(
            pair_dim=int(pair_embed_dim),
            num_edge=getattr(args, "num_edge", 64),
            num_spatial=getattr(args, "num_spatial", 512),
            wo_spd=getattr(args, "wo_spd", False),
            wo_edge=getattr(args, "wo_edge", False),
        )

        # --------------------------------------------------------------
        # Stage-1: Encoder (token-pair joint reasoning)
        # --------------------------------------------------------------
        self.encoder = EncoderBlock(
            num_encoder_layers=getattr(args, "encoder_layers", 12),
            embedding_dim=d_model,
            pair_dim=int(pair_embed_dim),
            pair_hidden_dim=getattr(args, "pair_hidden_dim", 64),
            ffn_embedding_dim=getattr(args, "encoder_ffn_embed_dim", 3072),
            num_attention_heads=getattr(args, "encoder_attention_heads", 48),
            dropout=getattr(args, "dropout", 0.1),
            attention_dropout=getattr(args, "attention_dropout", 0.1),
            activation_dropout=getattr(args, "activation_dropout", 0.1),
            activation_fn=getattr(args, "activation_fn", "gelu"),
            droppath_prob=getattr(args, "droppath_prob", 0.0),
            pair_dropout=getattr(args, "pair_dropout", 0.25),
            wo_triopm=getattr(args, "wo_triopm", False),
            wo_pair=getattr(args, "wo_pair", False),
        )

        # 3-D geometric bias (shared across layers)
        self.se3_invariant_kernel = SE3InvariantKernel(
            pair_dim=int(pair_embed_dim),
            num_pair=128*128,
            num_kernel=getattr(args, "num_kernel", 128),
            std_width=getattr(args, "gaussian_std_width", 1.0),
            start=getattr(args, "gaussian_mean_start", 0.0),
            stop=getattr(args, "gaussian_mean_stop", 9.0),
        )


        # --------------------------------------------------------------
        # Stage-2: chain-level modules
        # --------------------------------------------------------------

        self.chain_token_feature = ChainTokenFeaturePlus(
            embed_dim=d_model,
            num_glob_feat=args.num_chain_glob_feat,    # e.g. temperature, coexistence
            num_block_feat=args.num_chain_block_feat,  # e.g. volume fraction
            enable_topology_symmetry_break=not getattr(args, "disable_chain_symmetry_break", False),
        )

        # Stage-2-only path: initialize segment tokens from a SMILES-id dictionary embedding.
        # This keeps topology-level modeling while avoiding atom-level chemistry encoding.
        self.stage2_only_smiles_embed = nn.Embedding(
            getattr(args, "num_seg_smiles_types", 20000), d_model
        )

        fp_nbits = int(getattr(args, "fp_nbits", 2048))
        self.fp_proj = nn.Sequential(
            nn.Linear(fp_nbits, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.register_buffer("fp_table", torch.empty(0), persistent=True)

        # ---- chain-level edge / pair bias ----
        self.chain_edge_feature = ChainEdgeFeaturePlus(
            pair_dim=args.chain_pair_dim,
            max_chain_dist=args.max_chain_dist,
        )


        self.chain_encoder = EncoderBlock(
            num_encoder_layers=getattr(args, "chain_encoder_layers", 6),
            embedding_dim=d_model,
            pair_dim=args.chain_pair_dim,
            pair_hidden_dim=getattr(args, "chain_pair_hidden_dim", 64),
            ffn_embedding_dim=getattr(args, "chain_ffn_embed_dim", 4*d_model),
            num_attention_heads=args.chain_attention_heads,
            dropout=getattr(args, "dropout", 0.1),
            attention_dropout=getattr(args, "attention_dropout", 0.1),
            wo_triopm=True,
            wo_pair=False,
        )

        # --------------------------------------------------------------
        # Pretrain Heads
        # --------------------------------------------------------------
        self.lm_head = MaskLMHead(
            embed_dim=getattr(args, "encoder_embed_dim", 768),
            output_dim=128,
            weight=self.embed_tokens.weight,
        )

        self.movement_pred_head = MovementPredictionHead(
            getattr(args, "encoder_embed_dim", 768),
            int(pair_embed_dim),
            getattr(args, "encoder_attention_heads", 48),
        )

        # --------------------------------------------------------------
        # Downstream NN (dimension based on args)
        # --------------------------------------------------------------
        stage1_capacity_hidden = int(getattr(args, "stage1_capacity_mlp_hidden", 0) or 0)
        if stage1_capacity_hidden > 0:
            self.stage1_capacity_mlp = nn.Sequential(
                nn.Linear(getattr(args, "encoder_embed_dim", 768), stage1_capacity_hidden),
                nn.GELU(),
                nn.Linear(stage1_capacity_hidden, getattr(args, "encoder_embed_dim", 768)),
            )
        else:
            self.stage1_capacity_mlp = None

        self.reg_head = nn.Sequential(
        nn.Linear(getattr(args, "encoder_embed_dim", 768), 128),
        nn.GELU(),
        nn.Linear(128, 32),
        nn.GELU(),
        nn.Linear(32, getattr(args, "num_tasks", 1))
        )

    def _ensure_fp_table(self):
        if self._fp_table_built:
            return
        if getattr(self.args, "chain_only_repr", "smiles_embed") != "fp_morgan2048":
            return
        # If fp_table was restored from a checkpoint, reuse it without rebuilding.
        # This allows inference in environments without RDKit.
        if isinstance(getattr(self, "fp_table", None), torch.Tensor) and self.fp_table.numel() > 0:
            self._fp_table_built = True
            return

        import pickle
        import numpy as np

        pkl_path = getattr(self.args, "pkl_path", None)
        if not pkl_path:
            raise ValueError("fp_morgan2048 requires args.pkl_path to read seg_smiles_vocab")

        header = pickle.load(open(pkl_path, "rb"))
        seg_smiles_vocab = header.get("seg_smiles_vocab", None)
        seg_smiles_vocab_size = header.get("seg_smiles_vocab_size", None)
        if not isinstance(seg_smiles_vocab, dict) or not seg_smiles_vocab:
            raise ValueError("fp_morgan2048 requires pkl header['seg_smiles_vocab'] (string->id mapping)")

        if seg_smiles_vocab_size is None:
            seg_smiles_vocab_size = int(max(int(v) for v in seg_smiles_vocab.values()) + 1)
        seg_smiles_vocab_size = int(seg_smiles_vocab_size)

        # Build id->smiles table aligned with seg_smiles_id indices.
        id_to_smiles = [None] * seg_smiles_vocab_size
        for smi, idx in seg_smiles_vocab.items():
            try:
                iid = int(idx)
            except Exception:
                continue
            if 0 <= iid < seg_smiles_vocab_size:
                id_to_smiles[iid] = str(smi)

        fp_radius = int(getattr(self.args, "fp_radius", 2))
        fp_nbits = int(getattr(self.args, "fp_nbits", 2048))
        replace_star = bool(getattr(self.args, "fp_replace_star_fallback", True))

        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.*")
        RDLogger.DisableLog("rdWarning.*")
        RDLogger.DisableLog("rdError.*")

        fp_table = np.zeros((seg_smiles_vocab_size, fp_nbits), dtype=np.float32)
        failed = []

        for iid in range(1, seg_smiles_vocab_size):
            smi = id_to_smiles[iid]
            if not smi:
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None and replace_star:
                mol = Chem.MolFromSmiles(smi.replace("*", "C"))

            if mol is None:
                failed.append(smi)
                continue

            try:
                bv = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
                arr = np.zeros((fp_nbits,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(bv, arr)
                fp_table[iid] = arr.astype(np.float32, copy=False)
            except Exception:
                failed.append(smi)

        if failed:
            print(f"[fp_morgan2048] RDKit failed for {len(failed)} SMILES; set fp row to zeros.")
            uniq = sorted(set(failed))
            print(f"[fp_morgan2048] failed_smiles (unique={len(uniq)}): {uniq[:50]}")

        fp_tensor = torch.from_numpy(fp_table)
        self.fp_table = fp_tensor.to(device=self.embed_tokens.weight.device)
        self._fp_table_built = True

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def _build_chain_only_seg_repr(self, batch):
        """Initialize graph nodes from monomer identifiers or fixed fingerprints."""
        seg_valid_mask = batch["seg_valid_mask"]
        seg_smiles_id = batch.get("seg_smiles_id", None)
        if seg_smiles_id is None:
            # Backward-compatible fallback for old preprocessed files.
            seg_smiles_id = batch["seg_block_id"]

        seg_smiles_id = seg_smiles_id.long()
        if getattr(self.args, "chain_only_repr", "smiles_embed") == "fp_morgan2048":
            self._ensure_fp_table()
            fp_size = int(self.fp_table.size(0))
            if fp_size <= 0:
                raise RuntimeError("fp_table is empty; failed to build fingerprints")
            seg_smiles_id = torch.remainder(seg_smiles_id, fp_size)
            seg_fp = self.fp_table[seg_smiles_id]
            seg_repr = self.fp_proj(seg_fp.to(dtype=self.embed_tokens.weight.dtype))
        else:
            vocab_size = int(self.stage2_only_smiles_embed.num_embeddings)
            seg_smiles_id = torch.remainder(seg_smiles_id, vocab_size)
            seg_repr = self.stage2_only_smiles_embed(seg_smiles_id)
        seg_repr = seg_repr * seg_valid_mask.unsqueeze(-1).float()
        seg_repr = seg_repr.to(dtype=self.embed_tokens.weight.dtype)

        return seg_repr

    def _predict_chain(self, batch, segment_repr):
        """Apply the supplied polymer graph to learned or identifier segment features."""
        tokens, attention_mask, metadata = self.chain_token_feature(
            seg_repr=segment_repr,
            seg_dop=batch["seg_dop"],
            seg_block_id=batch["seg_block_id"],
            glob_feat=batch["chain_glob_feat"],
            glob_feat_mask=batch["chain_glob_mask"],
            block_feat=batch["block_feat"],
            block_feat_mask=batch["block_feat_mask"],
            chain_node_seg_id=batch.get("chain_node_seg_id", None),
            chain_topo_dist=batch.get("chain_topo_dist", None),
            num_heads=self.args.chain_attention_heads,
        )
        batch_size, token_count, _ = tokens.shape
        pair_bias = tokens.new_zeros(batch_size, token_count, token_count, self.args.chain_pair_dim)
        pair_bias = self.chain_edge_feature(
            batch, pair_bias, repeat_start=int(metadata["repeat_start"]),
            chain_len=metadata["chain_len"], topo_dist=metadata["topo_dist"],
        )
        representation, _ = self.chain_encoder(
            tokens, pair_bias, atom_mask=None, pair_mask=None, attn_mask=attention_mask,
        )
        return self.reg_head(representation[:, 0, :])

    def forward(self, batch):
        """Return property logits/values, or atom/coordinate/distance pretraining outputs."""
        task = self.args.main_task
        if task == "chain_only":
            return self._predict_chain(batch, self._build_chain_only_seg_repr(batch))

        pair_dim = getattr(self.args, "pair_embed_dim", None)
        if pair_dim is None:
            pair_dim = getattr(self.args, "pair_dim", 512)
        encoded = self._encode_monomers(
            batch, pair_dim=int(pair_dim),
            wo_pair=getattr(self.args, "wo_pair", False),
            wo_geom_3d=getattr(self.args, "wo_geom_3d", False),
        )
        if task == "finetune":
            pooled = encoded.nodes[:, 0, :]
            if self.stage1_capacity_mlp is not None:
                pooled = pooled + self.stage1_capacity_mlp(pooled)
            return self.reg_head(pooled)
        if task == "chain":
            return self._predict_chain(batch, encoded.nodes[:, 2:encoded.atom_start, :])
        if task == "pretrain":
            return self._reconstruct_atoms(batch, encoded)
