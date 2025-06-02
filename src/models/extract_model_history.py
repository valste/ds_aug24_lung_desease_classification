# -*- coding: utf-8 -*-
import pandas as pd


def save_model_architecture(model) -> None:
    """
    save_model_architecture function saves the architecture of a Keras model to a CSV file.

    Input:
    model: keras.Model: Keras model to save

    Output:
    None
    """
    # Prepare a list to store layer details
    layer_info = []

    for layer in model.layers:
        try:
            # Try getting the output shape directly
            output_shape = layer.output.shape
        except AttributeError:
            try:
                # Or compute it if needed
                output_shape = layer.compute_output_shape(layer.input_shape)
            except Exception as e:
                print(f"Error computing output shape for layer {layer.name}: {e}")
                output_shape = "Not Available"  # Fallback if something still fails

        inbound_layers = []
        try:
            if isinstance(layer.input, list):
                inbound_layers = [inp._keras_history[0].name for inp in layer.input]
            else:
                inbound_layers = [layer.input._keras_history[0].name]
        except AttributeError:
            inbound_layers = ["Input"]

        layer_info.append(
            {
                "Layer Name": layer.name,
                "Layer Type": layer.__class__.__name__,
                "Output Shape": output_shape,
                "Number of Parameters": layer.count_params(),
                "Connected Layers": ",\n".join(inbound_layers),
            }
        )

    # Create a DataFrame
    df = pd.DataFrame(layer_info)

    # Save to CSV
    df.to_csv("model_architecture.csv", index=False)
