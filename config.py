config = {
    "patch_size": 8,
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 144,
    "num_classes": 37,
    "num_channels": 3,
    "qkv_bias": True,
    # Training-specific configs
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 3e-4,
    "weight_decay": 0.01
}