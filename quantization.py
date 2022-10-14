"""
Quantization Quickstart
=======================
Here is a four-minute video to get you started with model quantization.
..  youtube:: MSfV7AyfiA4
    :align: center
Quantization reduces model size and speeds up inference time by reducing the number of bits required to represent weights or activations.
In NNI, both post-training quantization algorithms and quantization-aware training algorithms are supported.
Here we use `QAT_Quantizer` as an example to show the usage of quantization in NNI.
"""

# %%
# Preparation
# -----------
#
# In this tutorial, we use a simple model and pre-train on MNIST dataset.
# If you are familiar with defining a model and training in pytorch, you can skip directly to `Quantizing Model`_.
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer,DoReFaQuantizer, LsqQuantizer
import torch
from torch import nn, optim
# from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT
from vae1 import VAE, trainer, evaluator, device

config_list0 = [ {
    'quant_types': ['input', 'weight'],
    'quant_bits': {'input': 8, 'weight': 8},
    'op_names': ['fc1', 'fc21','fc22','fc3']
}]

config_list1 = [ {
    'quant_types': ['input', 'weight'],
    'quant_bits': {'input': 4, 'weight': 4},
    'op_names': ['fc1', 'fc21','fc22','fc3']
}]
# DoReFa

config_list2 = [{
        'quant_types': ['weight'],
        'quant_bits': {
            'weight': 8,
        }, # you can just use `int` here because all `quan_types` share same bits length, see config for `ReLu6` below.
        'op_types':[ 'Linear']
    }]
config_list3 = [{
        'quant_types': ['weight'],
        'quant_bits': {
            'weight': 4,
        }, # you can just use `int` here because all `quan_types` share same bits length, see config for `ReLu6` below.
        'op_types':[ 'Linear']
    }]
config_list=[config_list0,config_list1,config_list2,config_list3]
quantizer_list=[QAT_Quantizer,DoReFaQuantizer,LsqQuantizer]
for q in quantizer_list:
    idx=0
    if q==DoReFaQuantizer:
        idx=2
    for i in range(idx,idx+2):
        # define the model
        model = VAE().to(device)

        print(model)
        # define the optimizer and criterion for pre-training

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # criterion = F.nll_loss

        epochs=1
        # pre-train and evaluate the model on MNIST dataset
        for epoch in range(epochs):
            trainer(model, epoch, optimizer)
            evaluator(epoch,model)

        model_path = "./log/vae_model_pre_compression.pth"
        torch.save(model.state_dict(), model_path)

        # %%
        # Quantizing Model
        # ----------------
        #
        # Initialize a `config_list`.
        # Detailed about how to write ``config_list`` please refer :doc:`compression config specification <../compression/compression_config_list>`.
        # QAT, LsqQuantizer

            
        # %%
        # finetuning the model by using QAT
        quantizer=None
        if idx>1:
            quantizer = q(model, config_list[idx], optimizer)
        else:
            dummy_input = torch.rand(32, 1, 28, 28).to(device)
            quantizer=q(model, config_list[idx], optimizer,dummy_input)

        quantizer.compress()

        # %%
        # The model has now been wrapped, and quantization targets ('quant_types' setting in `config_list`)
        # will be quantized & dequantized for simulated quantization in the wrapped layers.
        # QAT is a training-aware quantizer, it will update scale and zero point during training.

        for epoch in range(epochs):
            trainer(model, epoch, optimizer)
            evaluator(epoch,model)

        # %%
        # export model and get calibration_config
        model_path = "./log/vae_model.pth"
        calibration_path = "./log/vae_calibration.pth"
        calibration_config = quantizer.export_model(model_path, calibration_path)
        torch.save(model.state_dict(), "./log/vae_model1.pth")

        print("calibration_config: ", calibration_config)
        print(model)
    
