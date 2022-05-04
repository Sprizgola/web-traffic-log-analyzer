# web-traffic-log-analyzer
This repo aim to analyze the web traffic data exploiting the Apache access log using an unsupervised approach.
The approach used here is inspired by [this](https://www.sciencedirect.com/science/article/pii/S1084804520300515) paper. 

## Processing data
To process the data and extract the features run:

 `python preprocess_data.py --input_path PATH_TO_RAW_LOG --output_path PATH_TO_SAVE_DATA`
 
 There is a logfile sample in the `raw_data` directory.
 
 ## Analyze data
 To start the log analyzer run:
 
 `python main.py --data_path PATH_TO_PROCESSED_LOG --plot_output_path PATH_TO_SAVE_PLOT`.
 
 `plot_output_path` is an optional argument and can be ignored if you don't want to save the plot.
