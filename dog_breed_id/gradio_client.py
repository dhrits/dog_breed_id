# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/08_client.ipynb.

# %% auto 0
__all__ = ['predict_image']

# %% ../nbs/08_client.ipynb 2
from gradio_client import Client, handle_file
def predict_image(client, impath):
    """Given impath, returns a PIL annotated image, breed and confidence"""
    result = client.predict(
		img=handle_file(impath),
		api_name="/predict"
    )
    return result
