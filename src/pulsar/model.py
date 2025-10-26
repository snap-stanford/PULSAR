from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel




@dataclass
class UCEPretrainOutput(ModelOutput):
    """Output class for pretraining tasks with PULSAR."""
    
    loss: Optional[torch.FloatTensor] = None
    cls_embedding: Optional[torch.FloatTensor] = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    reconstruction_loss: Optional[torch.FloatTensor] = None
    mask: Optional[torch.FloatTensor] = None
    reconstructed_embeddings: Optional[torch.FloatTensor] = None


@dataclass
class UCERegressionOutput(ModelOutput):
    """
    Output class for regression tasks with PULSAR.

    Attributes:
        loss: Loss value if labels are provided.
        logits: Predicted regression values.
        cls_embedding: Embedding of the [CLS] token.
    """
    
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cls_embedding: Optional[torch.FloatTensor] = None


@dataclass
class UCEClassificationOutput(ModelOutput):
    """
    Output class for classification tasks with PULSAR.

    Attributes:
        loss: Loss value if labels are provided.
        logits: Classification logits.
        cls_embedding: Embedding of the [CLS] token.
    """
    
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cls_embedding: Optional[torch.FloatTensor] = None



class Projector(nn.Module):
    """
    Multi-layer projection network with ReLU activation.
    
    Args:
        input_size: Input dimension size
        output_size: Output dimension size
        expansion_factor: Factor to expand intermediate layer size
    """
    
    def __init__(self, input_size: int, output_size: int, expansion_factor: int = 1):
        super().__init__()
        self.proj_1 = nn.Linear(input_size, input_size * expansion_factor)
        self.act = nn.ReLU()
        self.proj_2 = nn.Linear(input_size * expansion_factor, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the projector.
        
        Args:
            x: Input tensor
            
        Returns:
            Projected tensor
        """
        x = self.proj_1(x)
        x = self.act(x)
        x = self.proj_2(x)
        return x




class PULSARConfig(PretrainedConfig):
    """
    Configuration class for PULSAR model.
    
    This class stores the configuration of a PULSAR model, including encoder/decoder
    architecture parameters, training settings, and task-specific configurations.
    """
    
    model_type = "PULSAR"

    def __init__(
        self,
        input_size: int = 1280,
        hidden_size: int = 768,
        seq_length: int = 1024,
        encoder_num_hidden_layers: int = 12,
        encoder_num_attention_heads: int = 12,
        encoder_intermediate_size: int = 4096,
        decoder_num_hidden_layers: int = 6,
        decoder_num_attention_heads: int = 12,
        decoder_intermediate_size: int = 4096,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        use_cache: bool = True,
        classifier_dropout: Optional[float] = None,
        cell_state_size: int = 100,
        frozen: bool = False,
        use_cell_state: bool = True,
        num_labels: int = 2,
        cls_transform: bool = False,
        feed_cls_only: bool = True,
        use_decoder: bool = True,
        element_mask_strategy: str = "random",
        element_mask_ratio: float = 0.5,
        noise_norm: bool = False,
        decoder_loop_num: int = 1,
        coordinated_masking_program_path: Optional[str] = None,
        contrastive_loss_weight: float = 1.0,
        recon_loss_weight: float = 1.0,
        model_name: Optional[str] = None,
        expansion_factor: int = 4,
        encoder_ratio: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.encoder_intermediate_size = encoder_intermediate_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_intermediate_size = decoder_intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.cell_state_size = cell_state_size
        self.seq_length = seq_length
        self.frozen = frozen
        self.use_cell_state = use_cell_state
        self.num_labels = num_labels  # Number of labels for classification tasks
        self.cls_transform = cls_transform
        self.feed_cls_only = feed_cls_only
        self.use_decoder = use_decoder
        self.element_mask_strategy = element_mask_strategy
        self.element_mask_ratio = element_mask_ratio
        self.noise_norm = noise_norm
        self.decoder_loop_num = decoder_loop_num
        self.coordinated_masking_program_path = coordinated_masking_program_path
        self.contrastive_loss_weight = contrastive_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.model_name = model_name
        self.expansion_factor = expansion_factor
        self.encoder_ratio = encoder_ratio  # Ratio of kept tokens in encoder

    def get_encoder_config(self) -> PretrainedConfig:
        """
        Returns encoder-specific configuration.
        
        Returns:
            PretrainedConfig for the encoder.
        """
        return PretrainedConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.encoder_num_hidden_layers,
            num_attention_heads=self.encoder_num_attention_heads,
            intermediate_size=self.encoder_intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            use_cache=self.use_cache,
        )
    
    def get_decoder_config(self) -> PretrainedConfig:
        """
        Returns decoder-specific configuration.
        
        Returns:
            PretrainedConfig for the decoder.
        """
        return PretrainedConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.decoder_num_hidden_layers,
            num_attention_heads=self.decoder_num_attention_heads,
            intermediate_size=self.decoder_intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            use_cache=self.use_cache,
        )



class PULSARPreTrainedModel(PreTrainedModel):
    """
    Base class for all PULSAR models.
    
    Handles weight initialization and provides common methods for freezing layers.
    """

    config_class = PULSARConfig
    base_model_prefix = "pulsar"

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize model weights.
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def freeze(self) -> None:
        """Freeze all encoder and decoder layers."""
        print("Freezing the encoder and decoder layers.")
        for param in self.pulsar.parameters():
            param.requires_grad = False
    
    def freeze_first_n_layers(self, n: int) -> None:
        """
        Freeze the first n layers of the encoder.
        
        Args:
            n: Number of layers to freeze.
        """
        self.pulsar.freeze_encoder_by_layer(n)


class PULSAREncoder(nn.Module):
    """
    Transformer-based encoder for PULSAR model.
    
    This encoder processes cell embeddings using a stack of transformer layers.
    """
    
    def __init__(self, config: PretrainedConfig):
        """
        Initialize the PULSAREncoder with a transformer encoder.

        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        self.config = config
        # Initialize transformer encoder layer
        self.layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True  # Ensures input shape is (batch_size, seq_length, hidden_size)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.layer,
            num_layers=config.num_hidden_layers
        )
    
    def num_parameters(self) -> int:
        """
        Return the total number of parameters in the encoder.

        Returns:
            Total number of parameters.
        """
        return sum(p.numel() for p in self.parameters())

    def freeze(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def freeze_by_layer(self, n: int) -> None:
        """
        Freeze the first n layers of the transformer encoder.

        Args:
            n: Number of layers to freeze.
        """
        for i, layer in enumerate(self.encoder.layers):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False
        total_layers = len(self.encoder.layers)
        print(f"Frozen the first {n} layers of the encoder. Total layers: {total_layers}")
        

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> list:
        """
        Forward pass through the transformer encoder.

        Args:
            hidden_states: Input embeddings of shape (batch_size, seq_length, hidden_size)
            attention_mask: Attention mask of shape (batch_size, seq_length)

        Returns:
            List containing output embeddings tensor for compatibility with the model
        """
        # Process attention_mask for TransformerEncoder
        if attention_mask is not None:
            # Invert attention_mask: Transformer expects True for positions to ignore
            src_key_padding_mask = (attention_mask == 0)  # True where padding
        else:
            src_key_padding_mask = None
        
        output = self.encoder(hidden_states, src_key_padding_mask=src_key_padding_mask)
        
        # Return as a list for compatibility with the rest of the model
        return [output]
    
        

class PULSAR(PULSARPreTrainedModel):
    """
    Main PULSAR model for encoding donor-level representations from cell embeddings.
    
    This model uses a transformer encoder to process cell embeddings and generate
    donor-level representations via a CLS token.
    """
    
    config: PULSARConfig

    def __init__(self, config: PULSARConfig):
        """
        Initialize the PULSAR model with encoder, decoder, and embeddings.

        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__(config)
        self.config = config
        self.encoder = PULSAREncoder(config.get_encoder_config())
        
        if config.use_decoder:
            self.decoder = PULSAREncoder(config.get_decoder_config())
        
        self.cell_state_embedding = nn.Embedding(config.cell_state_size, config.hidden_size)
        self.mask_embedding = nn.Parameter(torch.randn(config.hidden_size))
        self.cls_embedding = nn.Parameter(torch.randn(config.hidden_size))
        self.in_proj = Projector(config.input_size, config.hidden_size)
        self.out_proj = Projector(config.hidden_size, config.input_size)
        
        if config.cls_transform:
            self.cls_transform = Projector(config.hidden_size, config.hidden_size)
    
        self._init_weights(self)
        
        if config.frozen:
            self.encoder.freeze()

    def freeze_encoder_by_layer(self, n: int) -> None:
        """
        Freeze the first n layers of the encoder.

        Args:
            n: Number of layers to freeze.
        """
        self.encoder.freeze_by_layer(n)

    def forward(
        self, 
        cell_embeddings: torch.Tensor, 
        cell_state_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> list:
        """
        Forward pass through the encoder.

        Args:
            cell_embeddings: Input cell embeddings of shape (batch_size, seq_length, embedding_size)
            cell_state_ids: Cell state IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)

        Returns:
            Output from the encoder
        """
        return self.encode(cell_embeddings, cell_state_ids, attention_mask)

    def forward_with_cls_transform(
        self, 
        cell_embeddings: torch.Tensor, 
        cell_state_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> list:
        """
        Forward pass through encoder with CLS token transformation.

        Args:
            cell_embeddings: Input cell embeddings of shape (batch_size, seq_length, embedding_size)
            cell_state_ids: Cell state IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)

        Returns:
            Output from the encoder with transformed CLS token
        """
        encoder_return = self.encode(cell_embeddings, cell_state_ids, attention_mask)
        cls_output = encoder_return[0][:, 0, :]
        cls_output = self.cls_transform(cls_output)
        encoder_return[0][:, 0, :] = cls_output
        return encoder_return

    def encode(
        self, 
        cell_embeddings: torch.Tensor, 
        cell_state_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> list:
        """
        Encode input cell embeddings with CLS token prepended.

        Args:
            cell_embeddings: Input cell embeddings of shape (batch_size, seq_length, embedding_size)
            cell_state_ids: Cell state IDs (unused, kept for API compatibility)
            attention_mask: Attention mask of shape (batch_size, seq_length)

        Returns:
            Encoded output of shape (batch_size, seq_length + 1, hidden_size)
        """
        # Prepend CLS token to the sequence
        batch_size = cell_embeddings.size(0)
        cls_embeds = self.cls_embedding.unsqueeze(0).expand(batch_size, 1, -1)
        
        # Project cell embeddings to hidden size
        cell_embeddings = self.in_proj(cell_embeddings)
        
        # Combine CLS token with cell embeddings
        combined_embeds = torch.cat([cls_embeds, cell_embeddings], dim=1)
        
        # Pass through encoder
        encoder_output = self.encoder(hidden_states=combined_embeds)
        return encoder_output
        

    def decode(
        self, 
        input_cls_embed: torch.Tensor, 
        cell_embeddings: torch.Tensor,
        cell_state_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> list:
        """
        Decode input embeddings using the decoder.

        Args:
            input_cls_embed: Input CLS embedding of shape (batch_size, 1, hidden_size)
            cell_embeddings: Input cell embeddings of shape (batch_size, seq_length, embedding_size)
            cell_state_ids: Cell state IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask (unused, kept for API compatibility)

        Returns:
            Decoded output of shape (batch_size, seq_length + 1, hidden_size)
        """
        # Add cell state embeddings if configured
        if self.config.use_cell_state and cell_state_ids is not None:
            state_embeds = self.cell_state_embedding(cell_state_ids)
            cell_embeddings = cell_embeddings + state_embeds
        
        # Iterative decoding
        for _ in range(self.config.decoder_loop_num):
            combined_embeds = torch.cat([input_cls_embed, cell_embeddings], dim=1)
            decoder_output = self.decoder(hidden_states=combined_embeds)
            cell_embeddings = decoder_output[0][:, 1:, :]  # Remove CLS token
        
        # Project output back to original embedding space
        decoder_output[0] = self.out_proj(decoder_output[0])
        return decoder_output





class PULSARForRegression(PULSARPreTrainedModel):
    """
    PULSAR model with a regression head for predicting continuous values.
    
    This model extends PULSAR with a regression head that predicts continuous
    values from the CLS token representation.
    """
    
    config_class = PULSARConfig

    def __init__(self, config: PULSARConfig):
        """
        Initialize the PULSARForRegression model.

        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__(config)
        self.pulsar = PULSAR(config)
        self.regression_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * config.expansion_factor),
            nn.ReLU(),
            nn.Linear(config.hidden_size * config.expansion_factor, config.num_labels)
        )
        
        self._init_weights(self)
    

    def forward(
        self,
        cell_embedding: torch.Tensor,
        cell_type_idx: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> UCERegressionOutput:
        """
        Forward pass for the regression model.

        Args:
            cell_embedding: Input cell embeddings of shape (batch_size, seq_length, embedding_size)
            cell_type_idx: Cell state IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)
            labels: Ground truth labels for computing the loss

        Returns:
            UCERegressionOutput containing loss, logits, and CLS embedding
        """
        # Get encoder output
        encoder_output = self.pulsar(cell_embedding, cell_type_idx, attention_mask)
        cls_output = encoder_output[0][:, 0, :]  # Extract CLS token
        
        # Predict regression values
        logits = self.regression_head(cls_output).squeeze(-1)  # Shape: (batch_size)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            # Ensure dtype compatibility
            if logits.dtype != labels.dtype:
                labels = labels.to(logits.dtype)
            loss = loss_fn(logits, labels)

        return UCERegressionOutput(loss=loss, logits=logits, cls_embedding=cls_output)
    

class PULSARForClassification(PULSARPreTrainedModel):
    """
    PULSAR model with a classification head for predicting discrete labels.
    
    This model extends PULSAR with a classification head that predicts class
    labels from the CLS token representation.
    """
    
    config_class = PULSARConfig

    def __init__(self, config: PULSARConfig):
        """
        Initialize the PULSARForClassification model.

        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__(config)
        self.pulsar = PULSAR(config)
        self.num_labels = config.num_labels
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * config.expansion_factor),
            nn.ReLU(),
            nn.Linear(config.hidden_size * config.expansion_factor, self.num_labels)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_weights(self)


    def forward(
        self,
        cell_embedding: torch.Tensor,
        cell_type_idx: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> UCEClassificationOutput:
        """
        Forward pass for the classification model.

        Args:
            cell_embedding: Input cell embeddings of shape (batch_size, seq_length, embedding_size)
            cell_type_idx: Cell state IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)
            labels: Ground truth labels for computing the loss

        Returns:
            UCEClassificationOutput containing loss, logits, and CLS embedding
        """
        # Get encoder output
        encoder_output = self.pulsar(cell_embedding, cell_type_idx, attention_mask)
        cls_output = encoder_output[0][:, 0, :]  # Extract CLS token
        
        # Apply dropout and classify
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # Shape: (batch_size, num_labels)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Binary classification
                loss_fn = nn.BCEWithLogitsLoss()
                labels = labels.float()
                loss = loss_fn(logits.squeeze(-1), labels)
            else:
                # Multiclass classification
                loss_fn = nn.CrossEntropyLoss()
                labels = labels.long()
                loss = loss_fn(logits, labels)

        return UCEClassificationOutput(loss=loss, logits=logits, cls_embedding=cls_output)


