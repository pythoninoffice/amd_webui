# amd_webui

* Install Python 3. The only supported versions are Python 3.7.x, 3.8.x, 3.9.x, and 3.10.x. All other Python versions will not work
* Install git
* Double click on `start.bat` to install & run
* Code will crash with "diffusers" library not found error message, close the blackscreen
* Reopen blackscreen, a webui should appear
* Copy your Huggingface access token from Huggingface, paste into the first textbox on the "Settings" tab, click on login
* Once successfully logged in, head to Model Manager tab, enter the model name found on huggingface, for example: stabilityai/stable-diffusion-2
* Click on Download model. This step will download the diffuers model and then convert them into onnx format. This step can take up to 10-30 mins depends on your internet connection and CPU speed
* Once model is downloaded, close the blackscreen, then re-open it once more.
* You should see available models appear in the `Select a model` dropdown on the txt2img tab
* Enjoy :)
