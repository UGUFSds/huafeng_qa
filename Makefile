# Project: huafeng_qa
CONDA_ENV := huafeng_qa
PY := python

.PHONY: env install test run info clean

info:
	@echo "[info] conda version:" && conda --version
	@echo "[info] current envs:" && conda env list
	@echo "[info] target env: $(CONDA_ENV)"

# Create or update the conda env and install requirements
env:
	@conda env list | awk '{print $$1}' | grep -qx '$(CONDA_ENV)' \
		&& echo "[skip] env $(CONDA_ENV) exists" \
		|| (echo "[create] $(CONDA_ENV)"; conda create -y -n $(CONDA_ENV) python=3.13 -c conda-forge)
	@conda run -n $(CONDA_ENV) $(PY) -m pip install -U pip
	@conda run -n $(CONDA_ENV) $(PY) -m pip install -r requirements.txt

install:
	@conda run -n $(CONDA_ENV) $(PY) -m pip install -r requirements.txt

# Quick DeepSeek connectivity test
test:
	@conda run -n $(CONDA_ENV) $(PY) scripts/test_deepseek_langchain.py

# Placeholder "run" target; reuse the test script as a demo entrypoint
run:
	@conda run -n $(CONDA_ENV) $(PY) scripts/test_deepseek_langchain.py

clean:
	@echo "[clean] nothing to clean for now"
