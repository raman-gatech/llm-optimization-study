import os
import subprocess
import sys
import time
from pathlib import Path

LISTEN_PORT = 8000
OUTPUT_DIR_BASE = './bench_results'
BENCHMARK_CONFIGS = [
    ('default', {
        'dataset': 'random',
        'random_input_len': 128,
        'random_output_len': 128,
        'request_rate': 2,
        'num_prompts': 200,
    }),
    ('default', {
        'dataset': 'random',
        'random_input_len': 768,
        'random_output_len': 32,
        'request_rate': 2,
        'num_prompts': 200,
    }),
    ('default', {
        'dataset': 'random',
        'random_input_len': 128,
        'random_output_len': 512,
        'request_rate': 2,
        'num_prompts': 200,
    }),
    ('default', {
        'dataset': 'random',
        'random_input_len': 768,
        'random_output_len': 192,
        'request_rate': 2,
        'num_prompts': 200,
    }),
    ('default', {
        'dataset': 'random',
        'random_input_len': 512,
        'random_output_len': 128,
        'request_rate': 1,
        'num_prompts': 200,
    }),
    ('default', {
        'dataset': 'random',
        'random_input_len': 512,
        'random_output_len': 128,
        'request_rate': 2,
        'num_prompts': 200,
    }),
    ('default', {
        'dataset': 'random',
        'random_input_len': 512,
        'random_output_len': 128,
        'request_rate': 4,
        'num_prompts': 200,
    }),
    ('default', {
        'dataset': 'random',
        'random_input_len': 512,
        'random_output_len': 128,
        'request_rate': 8,
        'num_prompts': 200,
    }),
    ('default', {
        'dataset': 'random',
        'random_input_len': 512,
        'random_output_len': 128,
        'request_rate': 16,
        'num_prompts': 200,
    }),
]


def verify_dependencies(gpu_name):
    import platform
    python_version = platform.python_version()
    print('python version:', python_version)

    import torch
    torch_version = torch.__version__
    print('torch version:', torch_version)
    cuda_available = torch.cuda.is_available()
    print('torch cuda available:', cuda_available)
    cuda_version = torch.version.cuda
    print('torch cuda version:', cuda_version)
    num_gpu = torch.cuda.device_count()
    print('torch cuda number of gpu:', num_gpu)
    gpu_list = [torch.cuda.get_device_name(i) for i in range(num_gpu)]
    print('torch cuda gpu:', ', '.join(gpu_list))

    import vllm
    vllm_version = vllm.__version__
    print('vllm version:', vllm_version)

    if not cuda_available:
        print('abort: cuda is not available')
        exit(1)
    if num_gpu != 1:
        print('abort: number of gpu is not 1')
        exit(1)
    if gpu_name not in gpu_list[0]:
        print(f'abort: gpu name "{gpu_name}" does not match device name "{gpu_list[0]}"')
        exit(1)

    check_passed = True
    python_version_list = list(map(int, python_version.split('.')))
    if python_version_list[0] != 3 or python_version_list[1] < 10 or python_version_list[1] > 13:
        print('compatibility risks: python version should be between 3.10 and 3.13')
        check_passed = False
    if vllm_version == '0.18.0' and python_version_list[0] == 3 and python_version_list[1] <= 10:
        # bug introduced by pr: https://github.com/vllm-project/vllm/pull/36093
        # bug fix pr: https://github.com/vllm-project/vllm/pull/37158
        print('stability risks: python version <= 3.10 may crash if vllm version is 0.18.0')
        check_passed = False
    if vllm_version != '0.18.0':
        print('benchmark inconsistency risks: vllm version should be 0.18.0')
        check_passed = False
    if not torch_version.startswith('2.10.'):
        print('benchmark inconsistency risks: torch version should be 2.10.*')
        check_passed = False
    if cuda_version != '12.6':
        print('benchmark inconsistency risks: cuda version should be 12.6')
        check_passed = False

    if 'HF_HOME' not in os.environ:
        print('warning: environment variable HF_HOME not set, '
              'Hugging Face will use a directory under home directory as local cache for downloaded models, '
              'which may cause quota exceeded errors if your quota in the home partition is limited')
    if 'HF_TOKEN' not in os.environ or not os.environ['HF_TOKEN']:
        print('warning: environment variable HF_TOKEN not set, '
              'will not be able to use gated models from Hugging Face, '
              'unless they are already cached locally')

    return {
        'command': sys.argv,
        'python version': python_version,
        'torch version': torch_version,
        'torch cuda available': cuda_available,
        'torch cuda version': cuda_version,
        'torch cuda number of gpu': num_gpu,
        'torch cuda gpu': ', '.join(gpu_list),
        'vllm version': vllm_version,
        'dependencies compatibility': 'OK' if check_passed else 'WARNING',
    }


def dump_verify_results(verify_results, output_dir):
    with open(output_dir / 'env.txt', 'w') as f:
        for key, value in verify_results.items():
            f.write(f'{key}: {value}\n')


def launch_server(model_name, vllm_serve_args, output_file_name, output_dir_base):
    cmd = [
        'vllm',
        'serve',
        model_name,
        '--host',
        '127.0.0.1',
        '--port',
        str(LISTEN_PORT),
        '--dtype',
        'bfloat16',
    ]
    cmd.extend(vllm_serve_args)
    return subprocess.Popen(
        cmd,
        stdout=open(output_dir_base / 'serve_log' / f'{output_file_name}.out', 'w'),
        stderr=open(output_dir_base / 'serve_log' / f'{output_file_name}.err', 'w'),
    )


def wait_for_server(model_name, proc_server):
    import requests
    session = requests.Session()

    print('waiting for server')
    while True:
        try:
            r = session.get(f'http://127.0.0.1:{LISTEN_PORT}/health', timeout=1)
            r.raise_for_status()
            break
        except requests.exceptions.RequestException:
            if proc_server.poll() is not None:
                print(f'error: server process unexpectedly exited with code {proc_server.returncode}')
                exit(1)
            time.sleep(1)

    r = session.get(f'http://127.0.0.1:{LISTEN_PORT}/v1/models')
    r.raise_for_status()
    print('server models:', r.text)

    r = session.post(f'http://127.0.0.1:{LISTEN_PORT}/v1/completions', json={
        "model": model_name,
        "prompt": "Hello",
        "max_tokens": 16,
        "temperature": 0
    })
    r.raise_for_status()
    print('server completions:', r.text)

    print('server is alive')


def launch_monitor(output_file_name, output_dir_base):
    cmd = [
        'nvidia-smi',
        'dmon',
        '-s',
        'pucvmet',
        '-d',
        '1',
    ]
    proc_monitor = subprocess.Popen(
        cmd,
        stdout=open(output_dir_base / 'mon' / f'{output_file_name}.txt', 'w'),
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    return proc_monitor


def kill_process(process):
    if process is not None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def run_benchmark_impl(model_name, config, vllm_serve_args, output_dir_base):
    config_name, config_args = config
    config_name = str(config_name)
    dataset = str(config_args['dataset'])
    num_prompts = str(config_args['num_prompts'])
    request_rate = str(config_args['request_rate'])
    random_input_len = str(config_args['random_input_len'])
    random_output_len = str(config_args['random_output_len'])
    output_file_name = f'{config_name}-{dataset}-{random_input_len}x{random_output_len}-{request_rate}qps-{num_prompts}p'
    print('benchmark starts:', output_file_name)

    proc_server = None
    proc_monitor = None
    try:
        proc_server = launch_server(model_name, vllm_serve_args, output_file_name, output_dir_base)
        wait_for_server(model_name, proc_server)
        proc_monitor = launch_monitor(output_file_name, output_dir_base)
        cmd = [
            'vllm',
            'bench',
            'serve',
            '--backend',
            'vllm',
            '--model',
            model_name,
            '--endpoint',
            '/v1/completions',
            '--base-url',
            f'http://127.0.0.1:{LISTEN_PORT}',
            '--dataset-name',
            dataset,
            '--num-prompts',
            num_prompts,
            '--request-rate',
            request_rate,
            '--random-input-len',
            random_input_len,
            '--random-output-len',
            random_output_len,
            '--percentile-metrics',
            'ttft,tpot,itl,e2el',
            '--metric-percentiles',
            '50,95,99',
            '--save-result',
            '--result-dir',
            str(output_dir_base / 'result'),
            '--result-filename',
            f'{output_file_name}.json',
        ]
        subprocess.check_call(
            cmd,
            stdout=open(output_dir_base / 'bench_log' / f'{output_file_name}.out', 'w'),
            stderr=open(output_dir_base / 'bench_log' / f'{output_file_name}.err', 'w'),
        )
    finally:
        kill_process(proc_monitor)
        kill_process(proc_server)

    print('benchmark finished')


def run_benchmark(model_name, vllm_serve_args, output_dir_base):
    for config in BENCHMARK_CONFIGS:
        run_benchmark_impl(model_name, config, vllm_serve_args, output_dir_base)


def ensure_dirs(output_dir_base):
    os.makedirs(output_dir_base / 'serve_log', exist_ok=True)
    os.makedirs(output_dir_base / 'bench_log', exist_ok=True)
    os.makedirs(output_dir_base / 'mon', exist_ok=True)
    os.makedirs(output_dir_base / 'result', exist_ok=True)


def main():
    if len(sys.argv) < 4:
        print('usage: python bench.py <model_name> <gpu_name> <benchmark_name> [-- <vllm_serve_args> ...]')
        exit(2)

    model_name = sys.argv[1]
    gpu_name = sys.argv[2]
    benchmark_name = sys.argv[3]
    vllm_serve_args_index = -1
    for i in range(4, len(sys.argv)):
        if sys.argv[i] == '--':
            vllm_serve_args_index = i
    if vllm_serve_args_index >= 0:
        vllm_serve_args = sys.argv[vllm_serve_args_index+1:]
    else:
        vllm_serve_args = []

    verify_results = verify_dependencies(gpu_name)
    model_name_stem_index = model_name.rfind('/')
    if model_name_stem_index >= 0:
        model_name_stem = model_name[model_name_stem_index+1:]
    else:
        model_name_stem = model_name
    output_dir_base = Path(OUTPUT_DIR_BASE).resolve() / model_name_stem / gpu_name.replace(' ', '-') / benchmark_name
    print('output directory:', output_dir_base)
    ensure_dirs(output_dir_base)
    dump_verify_results(verify_results, output_dir_base)

    print(f'benchmark group starts (count: {len(BENCHMARK_CONFIGS)})')
    run_benchmark(model_name, vllm_serve_args, output_dir_base)
    print('benchmark group finished')


if __name__ == '__main__':
    main()
