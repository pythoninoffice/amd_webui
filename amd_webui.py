import gradio as gr
from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline
from huggingface_hub import _login
from huggingface_hub.hf_api import HfApi, HfFolder
import subprocess
import sys
import pathlib
import importlib.util
import numpy as np
import random
#from modules import txt2img


python = sys.executable
#repositories = pathlib.Path().absolute() / 'repositories'

onnx_dir = pathlib.Path().absolute()/'onnx_models'

#from PIL import Image

#baseImage = Image.open(r"in.jpg").convert("RGB") # opens an image directly from the script's location and converts to RGB color profile
#baseImage = baseImage.resize((768,512))

prompt = "A fantasy landscape, trending on artstation"
#denoiseStrength = 0.8 # a float number from 0 to 1 - decreasing this number will increase result similarity with baseImage
#scale = 7.5
#pipe = None
##need to set up UI for downloading weights


def txt2img(prompt, negative_prompt, steps, height, width, scale, denoise_strength=0, seed=None, scheduler=None, num_image=None):
    try:
        seed = int(seed)
        if seed < 0:
            seed = random.randint(0,4294967295)
    except:
        seed = random.randint(0, 4294967295) # 2^32
        
    generator = np.random.RandomState(seed)
        
    #generator = torch.Generator(device='cpu')
    #generator = generator.manual_seed(seed)
    image = pipe(prompt,
                negative_prompt = negative_prompt,
                num_inference_steps=steps,
                height = height,
                width = width,
                guidance_scale=scale,
                #strength=denoise_strength,
                generator = generator,
                num_images_per_prompt = num_image,
                ).images[0]
    return image

#image.save("t2i.png")

def img2img(prompt, negative_prompt, image_input, steps, height, width, scale, denoise_strength, seed=None, scheduler=None, num_image=None):
    
    if seed == '':

        seed = random.randint(0,4294967295)
    elif seed != '':
        seed = int(seed)
        if seed < 0:
            seed = random.randint(0,4294967295)
    print(f'this is the seed {seed}')
        
    generator = np.random.RandomState(seed)
    
    image = pipe(prompt,
                init_image = image_input,
                strength=denoise_strength,
                num_inference_steps=steps,
                guidance_scale=scale,
                negative_prompt = negative_prompt,
                num_images_per_prompt = num_image,
                generator = generator,
                height = height,
                width = width
                ).images[0]
    return image
    
    
    
                                                                                  
                                                                                  
    pass

def huggingface_login(token):
    try:
        #output = _login._login(HfApi(), token = token)
        output = _login._login(token = token, add_to_git_credential = True)
        return "Login successful."
    except Exception as e:
        return str(e)
    


def pip_install(lib):
    subprocess.run(f'echo Installing {lib}...', shell=True)
    if 'ort_nightly_directml' in lib:
        subprocess.run(f'echo 1', shell=True)
        subprocess.run(f'echo "{python}" -m pip install {lib}', shell=True)
        subprocess.run(f'"{python}" -m pip install {lib} --force-reinstall', shell=True)
    else:
        subprocess.run(f'echo 2', shell=True)
        subprocess.run(f'echo "{python}" -m pip install {lib}', shell=True, capture_output=True)
        subprocess.run(f'"{python}" -m pip install {lib}', shell=True, capture_output=True)

def pip_uninstall(lib):
    subprocess.run(f'echo Uninstalling {lib}...', shell=True)
    subprocess.run(f'"{python}" -m pip uninstall -y {lib}', shell=True, capture_output=True)

def is_installed(lib):
    library =  importlib.util.find_spec(lib)
    return (library is not None)

def download_sd_model(model_path):
    pip_install('onnx')
    from repositories.diffusers.scripts import convert_stable_diffusion_checkpoint_to_onnx
    onnx_opset = 14
    onnx_fp16 = False
    try:
        model_name = model_path.split('/')[1]
    except:
        model_name = model_path
    onnx_model_dir = onnx_dir/model_name 
    if not onnx_dir.exists():
        onnx_dir.mkdir(parents=True, exist_ok=True)
        print(model_name)
    convert_stable_diffusion_checkpoint_to_onnx.convert_models(model_path, str(onnx_model_dir), onnx_opset, onnx_fp16)
    pip_uninstall('onnx')

    
#'CompVis/stable-diffusion-v1-4'


def display_onnx_models():
    if not onnx_dir.exists():
        onnx_dir.mkdir(parents=True, exist_ok=True)
    return [m.name for m in onnx_dir.iterdir() if m.is_dir()]


def load_onnx_model(model):
##    if is_installed('onnx'):
##        pip_uninstall('onnx')
##    onnx_nightly = pathlib.Path().absolute()/'repositories/ort_nightly_directml-1.13.0.dev20220908001-cp39-cp39-win_amd64.whl'
##    pip_install(str(onnx_nightly))
##    subprocess.run('echo installing onnx nightly built', shell=True)
    global pipe
    pipe = OnnxStableDiffusionPipeline.from_pretrained(str(onnx_dir/model),
                                                   provider="DmlExecutionProvider")
    
    return 'model ready'


def load_onnx_model_i2i(model):
##    if is_installed('onnx'):
##        pip_uninstall('onnx')
##    onnx_nightly = pathlib.Path().absolute()/'repositories/ort_nightly_directml-1.13.0.dev20220908001-cp39-cp39-win_amd64.whl'
##    pip_install(str(onnx_nightly))
##    subprocess.run('echo installing onnx nightly built', shell=True)
    global pipe
    pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(str(onnx_dir/model),
                                                                  provider="DmlExecutionProvider")
    
    return 'model ready'


def start_app():
    with gr.Blocks() as app:
        gr.Markdown('SATBLE DIFFUSION WEBUI FOR AMD')
        with gr.Tab('txt2img'):
            txt2img_prompt_input = gr.Textbox(label='Prompt')
            txt2img_negative_prompt_input = gr.Textbox(label='Negative Prompt')
            with gr.Row():
                with gr.Column(scale = 1):

                    with gr.Row():
                        txt2img_model_input = gr.Dropdown(label='Select a model', choices = display_onnx_models())
                        test_output = gr.Textbox(label='testing output')
                    inference_step_input = gr.Slider(label='Steps', value = 30, minimum = 0, maximum=200, step = 1)
                    with gr.Row():
                        image_height = gr.Slider(label='Height', value = 512, minimum = 0, maximum=1080, step = 64)
                        image_width = gr.Slider(label='Width', value = 512, minimum = 0, maximum=1080, step = 64)
                    with gr.Row():
                        scale = gr.Slider(label='Scale', value = 7.5, minimum = 0, maximum=15, step = 0.1)
                        denoise_strength = gr.Slider(label='Denoise Strength', value = 1, minimum = 0, maximum=1, step = 0.1)
                    with gr.Row():
                        seed_input = gr.Textbox(label='Enter Seed here')
                        scheduler_input = gr.Dropdown(['option 1', 'option 2'])
                        
                    num_image = gr.Slider(label='Num. of Images', value = 1, minimum = 1, maximum=10, step = 1)
                        
                    #prompt_input_negative = gr.Textbox()
                    with gr.Row():
                        #load_onnx_model_button = gr.Button('Load Model')
                        txt2img_button = gr.Button('Generate')
                    
                txt2img_output = gr.Image(label='Output Image')
        with gr.Tab('img2img'):
            img2img_prompt_input = gr.Textbox(label='Prompt')
            img2img_negative_prompt_input = gr.Textbox(label='Negative Prompt')
            with gr.Row():
                img2img_image_input = gr.Image()
                img2img_image_output = gr.Image()
            with gr.Row():
                    img2img_model_input = gr.Dropdown(label='Select a model', choices = display_onnx_models())
                    img2img_test_output = gr.Textbox(label='testing output')
                    img2img_inference_step_input = gr.Slider(label='Steps', value = 30, minimum = 0, maximum=200, step = 1)
            with gr.Row():
                img2img_image_height = gr.Slider(label='Height', value = 512, minimum = 0, maximum=1080, step = 64)
                img2img_image_width = gr.Slider(label='Width', value = 512, minimum = 0, maximum=1080, step = 64)
            with gr.Row():
                img2img_scale = gr.Slider(label='Scale', value = 7.5, minimum = 0, maximum=15, step = 0.1)
                img2img_denoise_strength = gr.Slider(label='Denoise Strength', value = 1, minimum = 0, maximum=1, step = 0.1)
            with gr.Row():
                with gr.Row():
                    img2img_seed_input = gr.Textbox(label='Enter Seed here')
                    img2img_num_image = gr.Slider(label='Num. of Images', value = 1, minimum = 1, maximum=10, step = 1)
                img2img_scheduler_input = gr.Dropdown(['option 1', 'option 2'])
                
                
            img2img_button = gr.Button('Generate')
        with gr.Tab('Model Manager'):
            gr.Markdown("Some of the models require you logging in to Huggingface and agree to their terms. Make sure to do that before downloading models!")
            model_download_input = gr.Textbox()
            model_download_button = gr.Button('Download Model')
            model_dir_refresh = gr.Button('Refresh')
            
        with gr.Tab('Settings'):
            gr.HTML("Click on this link to find your <a href='https://huggingface.co/settings/tokens' style='color:blue'>Huggingface Access Token</a>")
            hugginface_token_input = gr.Textbox()
            huggingface_login_message = gr.Textbox()
            huggingface_login_button = gr.Button('Login HuggingFace')
            
        txt2img_button.click(txt2img, inputs=[txt2img_prompt_input, txt2img_negative_prompt_input, inference_step_input,
                                                                                  image_height, image_width, scale, denoise_strength,
                                                                                  seed_input, scheduler_input,
                                                                                  num_image], outputs = txt2img_output)
        
        #img2img_button.click(img2img, inputs = [img2img_prompt_input, img2img_image_input], outputs= img2img_image_output, show_progress=True)
        
        img2img_button.click(img2img, inputs=[img2img_prompt_input, img2img_negative_prompt_input, img2img_image_input, img2img_inference_step_input,
                                                                                  img2img_image_height, img2img_image_width, img2img_scale, img2img_denoise_strength,
                                                                                  img2img_seed_input, img2img_scheduler_input, img2img_num_image
                                                                                  ], outputs = img2img_image_output, show_progress=True)


        huggingface_login_button.click(huggingface_login,
                                       inputs = hugginface_token_input,
                                       outputs = huggingface_login_message)
        
        model_download_button.click(download_sd_model, inputs = model_download_input)
        
        #load_onnx_model_button.click(load_onnx_model, inputs=txt2img_model_input, show_progress=True, outputs = test_output)
        txt2img_model_input.change(load_onnx_model, inputs=txt2img_model_input, show_progress=True, outputs = test_output)
        img2img_model_input.change(load_onnx_model, inputs=img2img_model_input, show_progress=True, outputs = img2img_test_output)

    app.launch(inbrowser = True)


if __name__ == "__main__":
    start_app()
