"""Run the unchanged full synthetic clean as a free hosted-runner feasibility probe."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import h5py
    import psutil
    from eiffel.analysis.compare_metrics import RunSpec, read_run_metrics, read_per_family
    from eiffel.analysis.validate_round_state import validate
    from eiffel.toml_runner import load_profile, profile_to_overrides

    profile = load_profile(ROOT/'experiments/toml/synthetic_50k_clean.toml')
    assert profile['experiment']['rounds'] == 30
    assert profile['experiment']['num_clients'] == 10
    assert profile['dataset']['samples_per_client'] == 5000
    assert profile['dataset']['central_test_size'] == 12000
    out = ROOT/'cloud-evidence/full_synthetic_clean'
    out.mkdir(parents=True, exist_ok=True)
    (out/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    command = [sys.executable,'-m','eiffel',*profile_to_overrides(profile),
               f'hydra.run.dir={out.as_posix()}','hydra.output_subdir=.hydra']
    started = time.monotonic()
    peak_process_rss = 0
    minimum_free_ram = psutil.virtual_memory().available
    with (out/'training.log').open('w') as log:
        child = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            try:
                processes = [psutil.Process(child.pid),*psutil.Process(child.pid).children(recursive=True)]
                rss = 0
                for process in processes:
                    try: rss += process.memory_info().rss
                    except (psutil.NoSuchProcess,psutil.AccessDenied): pass
                peak_process_rss = max(peak_process_rss,rss)
                minimum_free_ram = min(minimum_free_ram,psutil.virtual_memory().available)
            except psutil.NoSuchProcess:
                pass
            if time.monotonic()-started > 3300:
                child.terminate()
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: child.kill();child.wait()
                break
            time.sleep(1)
    summary = {'scope':'full-size synthetic software QA; not real-dataset research',
               'seed':2026,'requested_rounds':30,'logical_clients':10,
               'training_rows':50000,'test_rows':12000,'features':32,
               'elapsed_seconds':round(time.monotonic()-started,3),
               'sampled_peak_process_tree_rss_bytes':peak_process_rss,
               'minimum_available_system_ram_bytes':minimum_free_ram,
               'runner_cpu_count':os.cpu_count(),'runner_total_ram_bytes':psutil.virtual_memory().total,
               'exit_code':child.returncode,'raw_hdf5_retained_after_job':False,
               'artifact_upload_enabled':False,'completed_rounds':0}
    hdf = out/'round_state.h5'
    if child.returncode == 0 and hdf.exists():
        errors = validate(hdf)
        summary['validator_errors'] = errors
        with h5py.File(hdf,'r') as h5:
            summary['completed_rounds'] = int(h5['meta'].attrs['last_complete_round'])
            expected = {f'round_{r:04d}' for r in range(1,31)}
            assert set(h5['clients']) == expected
            assert set(h5['rounds']) == expected
            assert set(h5['global']) == expected | {'round_0000'}
            assert all(len(h5['clients'][r])==10 for r in expected)
        assert not errors, errors
        assert summary['completed_rounds']==30
        metrics = read_run_metrics(RunSpec('clean',hdf),
                    ['accuracy','macro_f1','mcc','macro_class_recall','min_class_recall'],phase='evaluate')
        assert len(metrics)==30
        assert all(math.isfinite(float(row[k])) for row in metrics for k in ('accuracy','macro_f1','mcc'))
        summary['round_metrics'] = [{k:v for k,v in row.items() if k!='run'} for row in metrics]
        summary['final_per_family'] = [{k:v for k,v in row.items() if k!='run'}
            for row in read_per_family(RunSpec('clean',hdf),phase='evaluate') if int(row['round'])==30]
        summary['hdf5_bytes'] = hdf.stat().st_size
        summary['hdf5_sha256'] = hashlib.sha256(hdf.read_bytes()).hexdigest()
    else:
        print('TRAINING_FAILURE_TAIL\n'+(out/'training.log').read_text(errors='replace')[-12000:])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('EIFFEL_CLOUD_SUMMARY_BEGIN')
    print(json.dumps(summary,sort_keys=True,allow_nan=False))
    print('EIFFEL_CLOUD_SUMMARY_END')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        compact={k:v for k,v in summary.items() if k not in ('round_metrics','final_per_family')}
        with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a') as f:
            f.write('## Full synthetic 30-round feasibility probe\n\n```json\n'+json.dumps(compact,indent=2)+'\n```\n')
    return 0 if child.returncode==0 and summary['completed_rounds']==30 else 1


if __name__=='__main__':
    raise SystemExit(main())
