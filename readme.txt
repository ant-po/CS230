CS230 Project - Anton Ponomarev
"Ranking financial assets using neural networks"


To get the data:
1) define required parameters in ./data/data_params
2) run "build_dataset.py"
	- the raw data will be stored in ./data/raw_data
	- the processed data will be stored in ./data/processed_data
3) copy files from ./data/processed_data/data_set_YYYY_MM_DD HH_MM_SS to ./data/processed_data/latest_dataset

To run the model:
1) run "model_.py"
	- log will be saved in ./logs to be viewed from TensorBoard