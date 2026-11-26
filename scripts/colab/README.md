# Colab TPU v5e + EC2 L40S

Use [tpu_ec2.ipynb](tpu_ec2.ipynb) for the current deployment. Upload it to Colab,
select a TPU v5e runtime, and follow the numbered cells. The notebook starts the
TPU server, opens a direct outgoing SSH tunnel to EC2, validates both backends,
profiles their latency from the router's location, and compares the baselines,
PROFILE and GAPSHIFT with repeated runs and optional ablations.

Start the EC2 GPU server first using [the EC2 guide](../ec2/README.md). Set
`BRANCH` to a branch containing the new code. If the changes have not been pushed,
upload the local checkout to `/content/hetero-llm-inference` and disable
`UPDATE_REPO`. The notebook never publishes code itself.

Set `MODEL` and `MAX_LEN` to match EC2. Enter the EC2 hostname and SSH username,
and provide a key file in the temporary Colab runtime. The tunnel maps local
`127.0.0.1:8001` to EC2's `127.0.0.1:8000`. Set `SAVE_TO_DRIVE=True` if you want
copies of experiment logs and profiles saved to your Google Drive.

Profiles must be regenerated after changing the model, context range, serving
settings or network path. When switching to long contexts, restart EC2 with the
new limit, restart the TPU, then repeat the backend-check and profiling cells.

The previous [tpu_psc_smoke.ipynb](tpu_psc_smoke.ipynb) and its copy under
`colab-tpu-psc/` are historical PSC notebooks. They are retained to reproduce the
November results; use the EC2 notebook for current experiments.
