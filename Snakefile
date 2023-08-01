from glob import glob
from smiles_cl.constants import RE_CHECKPOINT, DEFAULT_EVALUATION_DATASETS


device = config.get('device', 'cuda')


def get_checkpoints_for_run(run_dir):
    run_dir = Path(run_dir)
    checkpoints = (run_dir / 'checkpoints').glob('**/epoch=*-step=*.ckpt')
    for ckpt in checkpoints:
        match = RE_CHECKPOINT.match(ckpt.name)

        if match is None:
            raise ValueError('Invalid checkpoint path: {}'.format(ckpt))
        
        step_id = match.group('step_id')
        yield 'evaluation/{run_dir}/smiles/steps/{ckpt_step}'.format(
            run_dir=run_dir.name,
            ckpt_step=step_id
        )

rule checkpoint_evaluation:
    input:
        lambda wildcards: glob(
            'runs/smiles-cl/{run_dir}/checkpoints/epoch=*-step={ckpt_step}.ckpt'.format(
                run_dir=wildcards.run_dir,
                ckpt_step=wildcards.ckpt_step
            )
        )
    output:
        directory('evaluation/{run_dir}/smiles/steps/{ckpt_step}')
    shell:
        "python evaluate.py --device {device} checkpoint {input}"

rule create_summary:
    input:
        lambda wildcards: get_checkpoints_for_run('runs/smiles-cl/{run_dir}'.format(run_dir=wildcards.run_dir))
    output:
        'evaluation/{run_dir}/smiles/summary.png'
    shell:
        "python evaluate.py create_summary evaluation/{wildcards.run_dir}/smiles"
