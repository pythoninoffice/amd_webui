# amd_webui
## System Requirements
- AMD GPU with at least 8GB VRAM
- One of: Python 3.7, 3.8, 3.9, or 3.10 | Download from https://www.python.org/
- Git | Download from https://git-scm.com/downloads

## How To Install
- Download the whole repo, place in a drive with large enough space
- Double click on `start.bat` to install & run. This will create a new Python virtual environment and install all required libraries. The installation can take up to 5 mins depending on your machine. At the end of the installation, you'll see the program crash with the following error that NO module named "diffusers":

![image](https://user-images.githubusercontent.com/90436829/205534573-eb9c3b6c-5a3d-4a7a-b218-39cc286ca5f3.png)

- Close the black screen (command prompt), and re-open `start.bat`, a web-ui should appear shortly

## Huggingface login
- Copy your Huggingface access token from Huggingface, paste into the first textbox on the "Settings" tab, click on `Login Huggingface`. You should see "login successful" message on the next textbox on the same tab.
![image](https://user-images.githubusercontent.com/90436829/205535219-5c2b1e5f-5164-4d06-80c9-d3a40e2ef251.png)


## Downloading SD Models
- Once successfully logged into Huggingface, head to Model Manager tab, enter the model name found on huggingface, for example: `stabilityai/stable-diffusion-2`
- Click on Download model. This step will download the diffuers model and then convert them into onnx format. This step can take up to 10-30 mins depends on your internet connection and CPU speed
![image](https://user-images.githubusercontent.com/90436829/205535362-353580bc-6466-490c-8904-e7a0bfcfb1a7.png)
![image](https://user-images.githubusercontent.com/90436829/205535867-5fc356dd-fd40-4eaa-bbee-198a0102bfa0.png)

- Monitor the blackscreen, once the model download & conversion is done, you'll see "Uninstalling onnx....". Then close the command prompt, and re-open `start.bat` again.
![image](https://user-images.githubusercontent.com/90436829/205537004-fb7a3296-1bfe-4533-81a4-efb3013a1a87.png)

- You should see available models appear in the `Select a model` dropdown on the txt2img tab, select a model and wait for it to load
![image](https://user-images.githubusercontent.com/90436829/205537222-a2d5f18f-644a-4916-ad4b-cce9d5d19e24.png)

- During image generation, the blackscreen should show something like this:
![image](https://user-images.githubusercontent.com/90436829/205537336-88597e48-5f49-4010-95f1-e2626f234bd3.png)

- If the blackscreen show something about "DmlExecutionProvider" is not in available provider names, then you are using CPU for image generation and something has gone wrong, contact me if this happens to you
![image](https://user-images.githubusercontent.com/90436829/205537433-fa0c7794-7eaf-4b23-8ebf-9d4ba16ce0b4.png)

- Enjoy :)
