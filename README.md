# Florence2 in ComfyUI

> Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks. 
Florence-2 can interpret simple text prompts to perform tasks like captioning, object detection, and segmentation. 
It leverages our FLD-5B dataset, containing 5.4 billion annotations across 126 million images, to master multi-task learning. 
The model's sequence-to-sequence architecture enables it to excel in both zero-shot and fine-tuned settings, proving to be a competitive vision foundation model.

## Installation:

Clone this repository to 'ComfyUI/custom_nodes` folder.

Install the dependencies in requirements.txt, transformers version 4.38.0 minimum is required:

`pip install -r requirements.txt`

or if you use portable (run this in ComfyUI_windows_portable -folder):

`python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Florence2\requirements.txt`

Supports most Florence2 models, which can be automatically downloaded with the `DownloadAndLoadFlorence2Model` to `ComfyUI/models/LLM`:

Official:

https://huggingface.co/microsoft/Florence-2-base

https://huggingface.co/microsoft/Florence-2-base-ft

https://huggingface.co/microsoft/Florence-2-large

https://huggingface.co/microsoft/Florence-2-large-ft
