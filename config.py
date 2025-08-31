# config.py

Config = {
    
    "patch_size": 8,            # Patch size of 8x8 pixels
    "hidden_size": 192,         
    "num_hidden_layers": 6,     
    "num_attention_heads": 6,   # Number of attention heads
    "intermediate_size": 4 * 192, # 4x the hidden size
    "hidden_dropout_prob": 0.1, # Add dropout for regularization
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "image_size": 64,           
    "num_classes": 100,         # CIFAR-100 has 100 classes
    "num_channels": 3,
    "qkv_bias": True,
    
    
    "batch_size": 64,           
    "num_epochs": 10,           
    "learning_rate": 3e-4,
    "weight_decay": 0.05,       
}