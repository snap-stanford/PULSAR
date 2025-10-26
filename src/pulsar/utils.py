from typing import Dict, List, Optional, Any

import numpy as np
import torch
from tqdm import tqdm

from pulsar.dataset import DonorDatasetAnndata


def collate_fn(
    batch: List[Dict[str, Any]],
    max_length: int,
    resample: bool = True,
    replace: bool = True,
    return_cell_expr: bool = False,
) -> Dict[str, Any]:
    """
    Collate function for batching donor data.
    
    Args:
        batch: List of donor data dictionaries
        max_length: Maximum number of cells to sample per donor
        resample: Whether to resample cells to max_length
        replace: Whether to sample with replacement when num_cells >= max_length
        return_cell_expr: Whether to return cell expression data
        
    Returns:
        Dictionary containing batched tensors for cell embeddings, cell type indices,
        labels, and donor IDs
    """
    cell_embeddings_list = []
    cell_type_idx_list = []
    labels = []
    donor_ids = []
    return_cell_expr_list = []
    
    for item in batch:
        donor_embedding = item["cell_embedding"]
        cell_type_data = item["cell_type_idx"]
        num_cells = len(cell_type_data)
        
        # Sample max_length cells from each donor
        if resample:
            if num_cells < max_length:
                # Sample with replacement if fewer cells than max_length
                sample_idx = np.random.choice(num_cells, max_length, replace=True)
            else:
                sample_idx = np.random.choice(num_cells, max_length, replace=replace)
            sampled_embeddings = donor_embedding[sample_idx]
            
            if return_cell_expr:
                cell_expr = item["cell_expr"][sample_idx]
        else:
            sampled_embeddings = donor_embedding
            if return_cell_expr:
                cell_expr = item["cell_expr"]
        
        # -1 for compatibility with the model
        cell_type_indices = [-1]
        
        cell_embeddings_list.append(sampled_embeddings)
        cell_type_idx_list.append(cell_type_indices)
        labels.append(item.get("labels", 0.0))
        donor_ids.append(item["donor_id"])
        
        if return_cell_expr:
            return_cell_expr_list.append(cell_expr)
    
    # Convert to tensors
    if isinstance(cell_embeddings_list[0], np.ndarray):
        cell_embeddings_list = [torch.tensor(x) for x in cell_embeddings_list]
    
    cell_embeddings_tensor = torch.stack(cell_embeddings_list)
    cell_type_idx_tensor = torch.tensor(cell_type_idx_list, dtype=torch.long)
    
    labels_array = np.array(labels)
    if labels_array.dtype.type is np.str_:
        # Keep string labels as-is
        labels_tensor = labels_array
    else:
        labels_tensor = torch.tensor(labels_array, dtype=torch.float32)

    return_dict = {
        "cell_embedding": cell_embeddings_tensor,
        "cell_type_idx": cell_type_idx_tensor,
        "labels": labels_tensor,
        "donor_id": donor_ids,
    }
    
    if return_cell_expr:
        return_dict["cell_expr"] = torch.stack(return_cell_expr_list)
    
    return return_dict


def extract_donor_embeddings_from_h5ad(
    adata,
    model: Optional[torch.nn.Module] = None,
    label_name: Optional[str] = None,
    donor_id_key: str = "donor_id",
    embedding_key: str = "X_uce",
    device: str = "cuda",
    sample_cell_num: int = 1024,
    resample_num: int = 1,
    use_tqdm: bool = True,
    batch_size: int = 10,
    collate_sampling: bool = True,
    replace: bool = True,
    seed: int = 0,
    max_length: int = 1024,
) -> Dict[str, Dict[str, Any]]:
    """
    Extract donor-level embeddings from AnnData object using a PULSAR model.
    
    Args:
        adata: AnnData object containing single-cell data
        model: Pre-trained PULSAR model for generating donor embeddings
        label_name: Column name in adata.obs for labels (optional)
        donor_id_key: Column name in adata.obs for donor IDs
        embedding_key: Key in adata.obsm for cell embeddings
        device: Device to run the model on ('cuda' or 'cpu')
        use_hvg: Whether to use highly variable genes
        use_mean_uce: Whether to use mean of cell embeddings (when model is None)
        sample_cell_num: Number of cells to sample per donor
        resample_num: Number of times to resample each donor
        use_tqdm: Whether to show progress bar
        batch_size: Batch size for DataLoader
        collate_sampling: Whether to resample in collate_fn
        use_expr: Whether to use expression data
        replace: Whether to sample with replacement
        seed: Random seed for reproducibility
        max_length: Maximum sequence length
        
    Returns:
        Dictionary mapping donor IDs to their embeddings and labels
        Format: {donor_id: {"embedding": [embedding_array], "label": label_value}}
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    

    # Create dataset
    dataset = DonorDatasetAnndata(
        adata,
        label_name=label_name,
        donor_id_key=donor_id_key,
        embedding_key=embedding_key,
        collate_sampling=collate_sampling,
        device="cpu",
        max_length=max_length,
    )
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(
            x, sample_cell_num, collate_sampling, replace, return_cell_expr=False
        ),
    )
    
    model.to(device)
    model.eval()
    
    donor_embedding_collection = {}
    
    with torch.no_grad():
        for resample_idx in range(resample_num):
            if use_tqdm:
                print(f"Resample {resample_idx} time")
                iterator = tqdm(enumerate(data_loader))
            else:
                iterator = enumerate(data_loader)
            
            for _, batch in iterator:
                input_embeddings = batch["cell_embedding"].to(device).to(torch.bfloat16)
                donor_ids = batch["donor_id"]
                
                # Get model output
                output = model(input_embeddings)
                
                # Extract CLS token (first token) as donor embedding
                donor_embeddings = output[0][:, 0, :].cpu().to(torch.float32).numpy()
                
                # Store embeddings for each donor
                for idx, donor_id in enumerate(donor_ids):
                    if donor_id not in donor_embedding_collection:
                        donor_embedding_collection[donor_id] = {
                            "embedding": [donor_embeddings[idx]],
                            "label": (
                                batch["labels"][idx].item()
                                if "labels" in batch
                                else None
                            ),
                        }
                    else:
                        donor_embedding_collection[donor_id]["embedding"].append(
                            donor_embeddings[idx]
                        )
    
    return donor_embedding_collection

