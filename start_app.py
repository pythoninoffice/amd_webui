import importlib.util
import platform
import subprocess
import sys
import pathlib
import time


python = sys.executable
if sys.version[:4] == "3.10":
    onnx_nightly = 'ort_nightly_directml-1.13.0.dev20220908001-cp310-cp310-win_amd64.whl'
elif sys.version[:3] == "3.9":
    onnx_nightly = 'ort_nightly_directml-1.13.0.dev20220908001-cp39-cp39-win_amd64.whl'
elif sys.version[:3] == "3.8":
    onnx_nightly = 'ort_nightly_directml-1.13.0.dev20220908001-cp38-cp38-win_amd64.whl'
elif sys.version[:3] == "3.7":
    onnx_nightly = 'ort_nightly_directml-1.13.0.dev20220908001-cp37-cp37-win_amd64.whl'
else:
    print('no support python version found, install any of the 3.7, 3.8, 3.9, or 3.10!')
print(f'You are using python version - {sys.version}')



required_lib = ['torch', 'onnxruntime', 'transformers', 'scipy', 'ftfy', 'gradio']
standard_onnx = 'onnx'
repositories = pathlib.Path().absolute() / 'repositories'
git_repos = ['https://github.com/huggingface/diffusers']
requirements = pathlib.Path().absolute()  /'requirements.txt'


def pip_install(lib):
    subprocess.run(f'echo Installing {lib}...', shell=True)
    subprocess.run(f'echo "{python}" -m pip install {lib}', shell=True)
    subprocess.run(f'"{python}" -m pip install {lib}', shell=True, capture_output=True)
    subprocess.run(f'"{python}" -m pip install {lib}', shell=True, capture_output=True)

def pip_install_requirements():
    subprocess.run(f'echo installing requirements', shell=True)
    subprocess.run(f'"{python}" -m pip install -r requirements.txt', shell=True, capture_output=True)

def is_installed(lib):
    library =  importlib.util.find_spec(lib)
    return (library is not None)

def git_clone(repo_url, repo_name):
    repo_dir = repositories/repo_name
    if not repo_dir.exists():
        repo_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(f'echo cloning {repo_dir}', shell=True)
        subprocess.run(f'git clone {repo_url} "{repo_dir}"', shell=True)
    else:
        subprocess.run(f'echo {repo_name} already exists!', shell=True)


#git_clone('https://github.com/huggingface/diffusers','diffusers')

#subprocess.run(rf'echo "{python}" -m pip install -e .\repositories\diffusers', shell=True)
#subprocess.run(rf'"{python}" -m pip install -e .\repositories\diffusers', shell=True, capture_output=True)

#for lib in required_lib:
 #   if not is_installed(lib):
 #       pip_install(lib)
pip_install_requirements()
subprocess.run(f'"{python}" -m pip install repositories/{onnx_nightly}', shell=True)
subprocess.run('echo Done installing', shell=True)



import amd_webui
amd_webui.start_app()
