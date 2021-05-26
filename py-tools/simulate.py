import os
import subprocess
from typing import List


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUST_TOOLS_DIR = f'{ROOT}/tools'


def compile(file_path: str, name: str, **kwargs):
    command = ['g++', '-std=c++17', file_path, '-o', name] + \
        [f'-D{k}={v}' for k, v in kwargs.items()]
    print(command)
    subprocess.call(command)


def simulate(bin: str, input_file: str, print_stderr=False) -> int:
    os.chdir(RUST_TOOLS_DIR)

    cmd = ['cargo', 'run', '--release', '--bin', 'tester',
           input_file, bin]
    print('$ ' + (' '.join(cmd)))
    res = subprocess.check_output(
        cmd, stderr=None if print_stderr else subprocess.DEVNULL).decode()
    return int(res.rstrip())


if __name__ == '__main__':
    cpp = f'{ROOT}/main.cpp'
    bin = f'/tmp/test'
    compile(cpp, bin)

    test0 = f'{RUST_TOOLS_DIR}/in/0000.txt'
    test1 = f'{RUST_TOOLS_DIR}/in/0001.txt'
    print(simulate(bin, test0))
    print(simulate(bin, test1))
