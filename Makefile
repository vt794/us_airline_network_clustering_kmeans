# Downloads, preprocesses, aggregates DB1B ticket and coupon to 
# an non-directional OD-indexed feature space, then trains a KMeans 
# unsupervised clustering model grouping observations into similar groups. 
# 
# Script takes two mandatory arguments for the `make data` command, and
# an array of optional yet recommended arguments for the `make model`
# command. 
#
# Underlying data is available from 1990 Q1, inclusive.

VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	python3 -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

# 1. data pipeline executes $(VENV)/bin/activate download.py preprocessing.py
data: $(VENV)/bin/activate 
	$(PYTHON) src/data_pipeline.py --year=$(year) --quarter=$(quarter)

# 2. model pipeline executes filter_aggregate.py and modeling.py
model: src/model_pipeline.py
	python src/model_pipeline.py --state=$(state) --airport=$(airport) --distance_min=$(distance_min) --distance_max=$(distance_max) --roundtrip=$(roundtrip) --rpcarrier=$(rpcarrier) --n_clusters=$(n_clusters) 
	
clean: clean_raw clean_processed clean_output clean_temp clean_data
clean_raw:
	rm -rf data/raw
clean_processed:
	rm -rf data/processed
clean_output:
	rm -rf data/output
clean_temp:
	rm -rf *.tmp
	rm -rf __pycache__
	rm -rf $(VENV)
clean_data:
	rm -rf data