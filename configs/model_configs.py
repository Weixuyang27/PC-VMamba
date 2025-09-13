model_configs = {
    'unet': {
        'num_classes': 1,
        'input_channels': 3,
        'base_channels': 64
    },

    'transunet': {
        'num_classes': 1,
        'input_channels': 3,
        'base_channels': 64,
        'img_size': 256,
        'patch_size': 16,
        'in_chans': 3,
        'embed_dim': 1024,
        'depth': 12,
        'num_heads': 8,
        'mlp_ratio': 4.0
    },

    'swinunet': {
        'num_classes': 1,
        'input_channels': 3,
        'img_size': 256,
        'patch_size': 4,
        'in_chans': 3,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7,
        'mlp_ratio': 4.0
    },

    'localmamba': {
        'num_classes': 1,
        'input_channels': 3,
        'd_model': 256,
        'n_layers': 4,
        'dropout': 0.1
    },

    'localvisionmamba': {
        'num_classes': 1,
        'input_channels': 3,
        'd_model': 256,
        'n_layers': 4,
        'dropout': 0.1
    }
}