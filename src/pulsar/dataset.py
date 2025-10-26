from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import random
import h5py


class DonorDatasetAnndata(Dataset):
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
        return len(self.donor_ids)

    def get_group(self):
        return self.donor_ids

    def get_group_num(self):
        group_set = set(self.donor_ids)
        return len(group_set)

    def get_gruop_idx_by_idx(self, idx_list):
        group_set = set([self.donor_ids[idx] for idx in idx_list])
        return list(group_set)

    def get_idx_by_group_idx(self, group_id_list):
        # get group_id's indices in donor_ids
        res = []
        for group_id in group_id_list:
            res += [
                idx
                for idx, donor_id in enumerate(self.donor_ids)
                if donor_id == group_id
            ]
        return res

    def __getitem__(self, idx):
        donor_id = self.donor_ids[idx]
        indices = self.donor_id_to_indices[donor_id]
        donor_embedding = self.embedding[indices]

        if self.dataset_presampling:
            # sample the dataset
            replace = len(donor_embedding) < self.max_length
            sample_idx = np.random.choice(
                len(donor_embedding), size=self.max_length, replace=replace
            )
            donor_embedding = donor_embedding[sample_idx]
            indices = indices[sample_idx]
            # sort the indices
            indices = np.sort(indices)

        if not self.collate_sampling:
            # if tensor
            if isinstance(donor_embedding, torch.Tensor):
                donor_embedding = torch.mean(donor_embedding, dim=0).unsqueeze(0)
            else:
                donor_embedding = np.mean(donor_embedding, axis=0).reshape(1, -1)
                
        if self.use_hvg:

            # normalize the embedding
            if isinstance(donor_embedding, torch.Tensor):
                donor_embedding = donor_embedding / torch.norm(
                    donor_embedding, dim=1, keepdim=True
                )
                # donor_embedding = (donor_embedding - donor_embedding.mean(axis=0)) / (donor_embedding.std(axis=0)+ 1e-8)
            else:
                donor_embedding = donor_embedding / np.linalg.norm(donor_embedding, axis=1, keepdims=True)
        
        return_dict = {
            "donor_id": donor_id,
            "cell_embedding": donor_embedding,
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