"""Dataset utilities for PULSAR.

Provides a PyTorch Dataset that groups cell-level embeddings by donor from
AnnData objects, enabling donor-level training and inference with the PULSAR
model.
"""

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import random
import h5py


class DonorDatasetAnndata(Dataset):
    """PyTorch Dataset that organizes single-cell data by donor from an AnnData object.

    Each item represents a single donor and contains all (or a sampled subset of)
    their cell embeddings. This allows the PULSAR transformer encoder to aggregate
    cell-level representations into a donor-level representation.

    The dataset supports:
        - Extracting pre-computed cell embeddings (ie. UCE) from ``adata.obsm`` or
          raw expression from ``adata.X``.
        - Optional highly variable gene (HVG) filtering when using raw expression.
        - Donor-level labels from ``adata.obs`` columns or ``adata.uns`` mappings.
        - Pre-sampling and collate-time sampling strategies to handle variable
          cell counts per donor.
        - Evaluation expansion (repeating donors) for multi-pass inference.
    """

    def __init__(
        self,
        adata,
        label_name=None,
        donor_id_key="donor_id",
        embedding_key="X_uce",
        eval_expansion_factor=None,
        use_hvg=False,
        collate_sampling=True,
        use_expr=False,
        device="cuda",
        return_cell_expr=False,
        dataset_presampling=False,
        max_length=1024,
    ):
        """Initialize the donor dataset from an AnnData object.

        Args:
            adata: An AnnData object containing single-cell data with cell
                embeddings and donor metadata.
            label_name: Column name in ``adata.obs`` or key in ``adata.uns``
                that holds donor-level labels (e.g. age, disease status).
                If ``None``, no labels are returned.
            donor_id_key: Column name in ``adata.obs`` that identifies which
                donor each cell belongs to.
            embedding_key: Key in ``adata.obsm`` for pre-computed cell
                embeddings (e.g. ``"X_uce"``). Use ``"X"`` to read directly
                from ``adata.X``.
            eval_expansion_factor: If set, repeats each donor this many times
                in the dataset to enable multi-pass stochastic inference.
            use_hvg: If ``True`` and ``embedding_key="X"``, subset to highly
                variable genes (requires ``adata.var["highly_variable"]``).
            collate_sampling: If ``True`` (default), returns all cell
                embeddings per donor for later sampling in the collate
                function. If ``False``, returns the mean embedding instead.
            use_expr: If ``True`` and ``embedding_key="X"``, cast expression
                data to float16.
            device: Device to place embedding tensors on (``"cuda"`` or
                ``"cpu"``).
            return_cell_expr: If ``True``, also returns raw gene expression
                from ``adata.X`` alongside embeddings.
            dataset_presampling: If ``True``, randomly sample ``max_length``
                cells per donor at ``__getitem__`` time instead of deferring
                to the collate function.
            max_length: Maximum number of cells to sample per donor when
                ``dataset_presampling=True``.
        """
        self.adata = adata
        self.label_name = label_name
        self.donor_id_key = donor_id_key
        self.embedding_key = embedding_key
        self.meta_data = self.adata.obs.copy()
        self.max_length = max_length
        self.use_hvg = use_hvg
        try:
            self.meta_data.reset_index(inplace=True)
        except:
            print("meta_data already has index")
        self.donor_ids = self.meta_data[self.donor_id_key].unique().tolist()
        self.eval_expansion_factor = eval_expansion_factor
        self.return_cell_expr = return_cell_expr
        if embedding_key == "X":
            self.embedding = self.adata.X
            # if embedding not dense, convert to dense
            try:
                self.embedding = self.embedding.toarray()
                print("Converted to dense")
            except:
                pass
            if use_hvg:
                # calculate highly variable genes
                self.embedding = self.embedding[:, self.adata.var["highly_variable"]]
                # verify nan
                if np.isnan(self.embedding).any():
                    print("Nan in embedding")
                    # show nan
                    print(np.argwhere(np.isnan(self.embedding)))
                print("Using highly variable genes")
                print(type(self.embedding))
                self.embedding = torch.as_tensor(self.embedding, device="cuda")
            if use_expr:
                self.embedding = torch.as_tensor(
                    self.embedding,
                    dtype=torch.float16,
                )
        else:
            self.embedding = self.adata.obsm[self.embedding_key]
            if device == "cuda":
                self.embedding = torch.as_tensor(
                    self.embedding, dtype=torch.float16, device="cuda"
                )
            else:
                self.embedding = torch.as_tensor(self.embedding)
        self.donor_id_to_indices = {}
        self.donor_id_to_indices = self.meta_data.groupby(self.donor_id_key, observed=False).indices
        self.embedding_dim = self.embedding.shape[1]
        self.dataset_presampling = dataset_presampling

        self.collate_sampling = collate_sampling

        if eval_expansion_factor is not None:
            self.donor_ids = np.repeat(self.donor_ids, eval_expansion_factor)

        self.donor_id_to_label = {}
        if label_name is not None:
            if label_name in self.meta_data.columns:
                self.label_name = label_name
                for donor_id in self.donor_ids:
                    idx = self.donor_id_to_indices[donor_id]
                    label = self.meta_data.loc[idx][self.label_name].values[0]
                    self.donor_id_to_label[donor_id] = label
            elif label_name in adata.uns.keys():
                self.label_name = label_name
                for donor_id in self.donor_ids:
                    label = self.adata.uns[label_name].loc[donor_id]
                    self.donor_id_to_label[donor_id] = label

    def __len__(self):
        """Return the number of donor entries (includes repeats from ``eval_expansion_factor``)."""
        return len(self.donor_ids)

    def get_group(self):
        """Return the list of donor IDs (may contain duplicates if expanded)."""
        return self.donor_ids

    def get_group_num(self):
        """Return the number of unique donors in the dataset."""
        group_set = set(self.donor_ids)
        return len(group_set)

    def get_gruop_idx_by_idx(self, idx_list):
        """Return unique donor IDs corresponding to the given dataset indices.

        Args:
            idx_list: List of integer indices into the dataset.

        Returns:
            List of unique donor IDs for the specified indices.
        """
        group_set = set([self.donor_ids[idx] for idx in idx_list])
        return list(group_set)

    def get_idx_by_group_idx(self, group_id_list):
        """Return all dataset indices that belong to the given donor IDs.

        Args:
            group_id_list: List of donor IDs to look up.

        Returns:
            List of integer indices into the dataset for the specified donors.
        """
        res = []
        for group_id in group_id_list:
            res += [
                idx
                for idx, donor_id in enumerate(self.donor_ids)
                if donor_id == group_id
            ]
        return res

    def __getitem__(self, idx):
        """Retrieve all data for a single donor.

        Args:
            idx: Integer index into the dataset.

        Returns:
            A dictionary with the following keys:

            - ``"donor_id"``: The donor identifier string.
            - ``"cell_embedding"``: Tensor or array of shape
              ``(n_cells, embedding_dim)`` containing cell embeddings for this
              donor. If ``dataset_presampling`` is enabled, ``n_cells`` is
              capped at ``max_length``. If ``collate_sampling`` is disabled,
              shape is ``(1, embedding_dim)`` (mean-pooled).
            - ``"cell_type_idx"``: List of ``-1`` values (placeholder for
              compatibility with the collate function).
            - ``"labels"`` *(optional)*: Donor-level label, present when
              ``label_name`` was specified.
            - ``"cell_expr"`` *(optional)*: Raw gene expression tensor of
              shape ``(n_cells, n_genes)``, present when
              ``return_cell_expr=True``.
        """
        donor_id = self.donor_ids[idx]
        indices = self.donor_id_to_indices[donor_id]
        cell_embedding = self.embedding[indices]

        if self.dataset_presampling:
            # sample the dataset
            replace = len(cell_embedding) < self.max_length
            sample_idx = np.random.choice(
                len(cell_embedding), size=self.max_length, replace=replace
            )
            cell_embedding = cell_embedding[sample_idx]
            indices = indices[sample_idx]
            # sort the indices
            indices = np.sort(indices)

        if not self.collate_sampling:
            # if tensor
            if isinstance(cell_embedding, torch.Tensor):
                cell_embedding = torch.mean(cell_embedding, dim=0).unsqueeze(0)
            else:
                cell_embedding = np.mean(cell_embedding, axis=0).reshape(1, -1)
                
        if self.use_hvg:

            # normalize the embedding
            if isinstance(cell_embedding, torch.Tensor):
                cell_embedding = cell_embedding / torch.norm(
                    cell_embedding, dim=1, keepdim=True
                )
            else:
                cell_embedding = cell_embedding / np.linalg.norm(cell_embedding, axis=1, keepdims=True)
        
        return_dict = {
            "donor_id": donor_id,
            "cell_embedding": cell_embedding,
            "cell_type_idx": [-1]
            * len(indices),  # for compatibility with the collate function
        }
        if self.label_name is not None:
            label = self.donor_id_to_label[donor_id]
            return_dict["labels"] = label
        if self.return_cell_expr:
            cell_expression = self.adata.X[indices]
            if isinstance(cell_expression, np.ndarray):
                cell_expression = torch.tensor(cell_expression, dtype=torch.float32)
            else:
                cell_expression = torch.tensor(
                    cell_expression.toarray(), dtype=torch.float32
                )
            return_dict["cell_expr"] = cell_expression


        return return_dict