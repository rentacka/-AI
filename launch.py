#!/usr/bin/env python
# coding: utf-8

import subprocess
import os
import sys
import importlib.util
import shlex
import platform
import argparse
import json
import webbrowser

dir_repos = "repositories"
dir_extensions = "extensions"
python = sys.executable
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")
stored_commit_hash = None
skip_install = False


def check_python_version():
    is_windows = platform.system() == "Windows"
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro

    if is_windows:
        supported_minors = [10]
    else:
        supported_minors = [7, 8, 9, 10, 11]

    if not (major == 3 and minor in supported_minors):
        import modules.errors

        modules.errors.print_error_explanation(f"""
INCOMPATIBLE PYTHON VERSION

This program is tested with 3.10.6 Python, but you have {major}.{minor}.{micro}.
If you encounter an error with "RuntimeError: Couldn't install torch." message,
or any other error regarding unsuccessful package (library) installation,
please downgrade (or upgrade) to the latest version of 3.10 Python
and delete current Python and "venv" folder in 感想文AI's directory.

You can download 3.10 Python from here: https://www.python.org/downloads/release/python-3109/

{"Alternatively, use a binary release of 感想文AI"}

Use --skip-python-version-check to suppress this warning.
""")


def commit_hash():
    global stored_commit_hash

    if stored_commit_hash is not None:
        return stored_commit_hash

    try:
        stored_commit_hash = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        stored_commit_hash = "<none>"

    return stored_commit_hash


def extract_arg(args, name):
    return [x for x in args if x != name], name in args


def extract_opt(args, name):
    opt = None
    is_present = False
    if name in args:
        is_present = True
        idx = args.index(name)
        del args[idx]
        if idx < len(args) and args[idx][0] != "-":
            opt = args[idx]
            del args[idx]
    return args, is_present, opt


def run(command, desc=None, errdesc=None, custom_env=None, live=False):
    if desc is not None:
        print(desc)

    if live:
        result = subprocess.run(command, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def check_run(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return result.returncode == 0


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def repo_dir(name):
    return os.path.join(dir_repos, name)


def run_python(code, desc=None, errdesc=None):
    return run(f'"{python}" -c "{code}"', desc, errdesc)


def run_pip(args, desc=None):
    if skip_install:
        return

    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {args} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")


def check_run_python(code):
    return check_run(f'"{python}" -c "{code}"')


def git_clone(url, dir, name, commithash=None):
    # TODO clone into temporary dir and move if successful

    if os.path.exists(dir):
        if commithash is None:
            return

        current_hash = run(f'"{git}" -C "{dir}" rev-parse HEAD', None, f"Couldn't determine {name}'s hash: {commithash}").strip()
        if current_hash == commithash:
            return

        run(f'"{git}" -C "{dir}" fetch', f"Fetching updates for {name}...", f"Couldn't fetch {name}")
        run(f'"{git}" -C "{dir}" checkout {commithash}', f"Checking out commit for {name} with hash: {commithash}...", f"Couldn't checkout commit {commithash} for {name}")
        return

    run(f'"{git}" clone "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}")

    if commithash is not None:
        run(f'"{git}" -C "{dir}" checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")

def run_extension_installer(extension_dir):
    path_installer = os.path.join(extension_dir, "install.py")
    if not os.path.isfile(path_installer):
        return

    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.abspath(".")

        print(run(f'"{python}" "{path_installer}"', errdesc=f"Error running install.py for extension {extension_dir}", custom_env=env))
    except Exception as e:
        print(e, file=sys.stderr)


def list_extensions(settings_file):
    settings = {}

    try:
        if os.path.isfile(settings_file):
            with open(settings_file, "r", encoding="utf8") as file:
                settings = json.load(file)
    except Exception as e:
        print(e, file=sys.stderr)

    disabled_extensions = set(settings.get('disabled_extensions', []))

    return [x for x in os.listdir(dir_extensions) if x not in disabled_extensions]


def run_extensions_installers(settings_file):
    if not os.path.isdir(dir_extensions):
        return

    for dirname_extension in list_extensions(settings_file):
        run_extension_installer(os.path.join(dir_extensions, dirname_extension))


def prepare_environment():
    global skip_install


    torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117")
    gradio_command = os.environ.get('Gradio_COMMAND', "pip install gradio==3.16.2")

    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")
    commandline_args = os.environ.get('COMMANDLINE_ARGS', "")

    sys.argv += shlex.split(commandline_args)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ui-settings-file", type=str, help="filename to use for ui settings", default='config.json')
    args, _ = parser.parse_known_args(sys.argv)

    sys.argv, _ = extract_arg(sys.argv, '-f')
    sys.argv, skip_torch_cuda_test = extract_arg(sys.argv, '--skip-torch-cuda-test')
    sys.argv, skip_python_version_check = extract_arg(sys.argv, '--skip-python-version-check')
    sys.argv, reinstall_torch = extract_arg(sys.argv, '--reinstall-torch')
    sys.argv, update_check = extract_arg(sys.argv, '--update-check')
    sys.argv, run_tests, test_dir = extract_opt(sys.argv, '--tests')
    sys.argv, skip_install = extract_arg(sys.argv, '--skip-install')

    if not skip_python_version_check:
        check_python_version()

    commit = commit_hash()

    print(f"Python {sys.version}")
    print(f"Commit hash: {commit}")

    if reinstall_torch or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)


    if reinstall_torch or not is_installed("gradio"):
        run(f'"{python}" -m {gradio_command}', "Installing torch and torchvision", "Couldn't install gradio", live=True)

    if not skip_torch_cuda_test:
        run_python("import torch; assert torch.cuda.is_available(), 'Torch is not able to use GPU; add --skip-torch-cuda-test to COMMANDLINE_ARGS variable to disable this check'")

    os.makedirs(dir_repos, exist_ok=True)


    run_pip(f"install -r {requirements_file}", "requirements for 感想文AI")
    
    if "--exit" in sys.argv:
        print("Exiting because of --exit argument")
        exit(0)

    if run_tests:
        exitcode = tests(test_dir)
        exit(exitcode)

def tests(test_dir):
    if "--api" not in sys.argv:
        sys.argv.append("--api")
    if "--ckpt" not in sys.argv:
        sys.argv.append("--ckpt")
        sys.argv.append("./test/test_files/empty.pt")
    if "--skip-torch-cuda-test" not in sys.argv:
        sys.argv.append("--skip-torch-cuda-test")
    if "--disable-nan-check" not in sys.argv:
        sys.argv.append("--disable-nan-check")

    print(f"Launching 感想文AI in another process for testing with arguments: {' '.join(sys.argv[1:])}")

    os.environ['COMMANDLINE_ARGS'] = ""
    with open('test/stdout.txt', "w", encoding="utf8") as stdout, open('test/stderr.txt', "w", encoding="utf8") as stderr:
        proc = subprocess.Popen([sys.executable, *sys.argv], stdout=stdout, stderr=stderr)

    import test.server_poll
    exitcode = test.server_poll.run_tests(proc, test_dir)

    print(f"Stopping 感想文AI process with id {proc.pid}")
    proc.kill()
    return exitcode

def start():
    print(f"Launching {'感想文AI'} with arguments: {' '.join(sys.argv[1:])}")
    webbrowser.open("http://localhost:7860/")


# In[2]:


prepare_environment()


from transformers import pipeline
import torch
import gradio as gr
from gradio.inputs import Textbox
from gradio.outputs import Textbox


# In[4]:


device = -1 # cpu
if torch.cuda.is_available():
    device = 0

generator = pipeline("text-generation", model="abeja/gpt-neox-japanese-2.7b", torch_dtype=torch.float16, device=device)


# In[19]:

prompt="この小説の感想を書きます。私が思ったことは、"

def generate_text(input_text,sepal_length):
    generated_text = generator(input_text+prompt, max_length=len(input_text)+len(prompt)+sepal_length,    
    do_sample=True,
    num_return_sequences=1,
    top_p=0.95,
    top_k=50)[0]['generated_text']
    
    generated_text = generated_text.replace(input_text+prompt, '')
    
    return generated_text

sepal_length = gr.inputs.Slider(minimum=10, maximum=400,default=10,label='感想の長さ')

input_text = gr.Textbox(label="要約を入力しよう(Ctrl+V")
output_text = gr.Textbox(label="Generated Text")

title = "感想文ＡＩ"
description = "マネーコイコイの閃き一千万はYoutubeで面白い動画をやってるから、おおすすめだよｂ https://www.youtube.com/@moneykoikoi"
jrr = gr.Interface(fn=generate_text, inputs= [input_text,sepal_length], outputs=output_text, title=title, description=description)

# In[ ]:


if __name__ == "__main__":
    start()

jrr.launch()
