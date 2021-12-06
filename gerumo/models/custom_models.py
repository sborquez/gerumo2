from .base import BaseModel, MODEL_REGISTRY


@MODEL_REGISTRY.register()
class CNN(BaseModel):

    _KWARGS = [
        'conv_kernel_sizes', 'compress_filters', 'compress_kernel_size',
        'latent_variables', 'dense_layer_units'
    ]

    def architecture(self,
                     conv_kernel_sizes=[5, 3, 3], compress_filters=256,
                     compress_kernel_size=3, latent_variables=200,
                     dense_layer_unitcs=[128, 128, 64]):
        self.conv = conv_kernel_sizes

    def forward(self, X, training=False):
        return super().forward(X, training=training)
