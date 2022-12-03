import importlib.util
import platform
import subprocess
import sys
import pathlib


python = sys.executable

required_lib = ['torch', 'onnxruntime', 'transformers', 'scipy', 'ftfy', 'gradio']
standard_onnx = 'onnx'
onnx_nightly = 'ort_nightly_directml-1.13.0.dev20220908001-cp39-cp39-win_amd64.whl'
repositories = pathlib.Path().absolute() / 'repositories'
git_repos = ['https://github.com/huggingface/diffusers']

def pip_install(lib):
    subprocess.run(f'echo Installing {lib}...', shell=True)
    subprocess.run(f'echo "{python}" -m pip install {lib}', shell=True)
    subprocess.run(f'"{python}" -m pip install {lib}', shell=True, capture_output=True)


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


git_clone('https://github.com/huggingface/diffusers','diffusers')

subprocess.run(rf'echo "{python}" -m pip install -e .\repositories\diffusers', shell=True)
subprocess.run(rf'"{python}" -m pip install -e .\repositories\diffusers', shell=True, capture_output=True)

for lib in required_lib:
    if not is_installed(lib):
        pip_install(lib)
subprocess.run(f'"{python}" -m pip install repositories/ort_nightly_directml-1.13.0.dev20220908001-cp39-cp39-win_amd64.whl', shell=True)
subprocess.run('echo Done installing', shell=True)


import amd_webui
amd_webui.start_app()
