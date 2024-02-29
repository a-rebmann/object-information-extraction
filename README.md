# Uncovering Object-centric Data in Classical Event Logs
<sub>
written by <a href="mailto:rebmann@informatik.uni-mannheim.de">Adrian Rebmann</a><br />
</sub>

## About
This repository contains the prototype of the approach, evaluation, and data as described in

*Rebmann, A., Rehse, J.-R., & van der Aa, H.: Uncovering Object-centric Data in Classical Event Logs for the Automated Transformation from XES to OCEL* accepted for the 20th International Conference on Business Process Management (BPM22)

## Setup and Usage

### Installation instructions
**The project was built with python 3.9**

1. create a virtual environment for the project 
2. install the dependencies in requirements.txt: using pip <code> pip install -r requirements.txt </code>
3. The approach uses the POS-tagger from spacy => run <code>python -m spacy download en_core_web_lg</code>

### Input, Output and Model directories
The following default directories are used for input and output, as well as trained models used by the approach:
* DEFAULT_INPUT_DIR = 'input/logs/'
* DEFAULT_OUTPUT_DIR = 'output/logs/' 
* DEFAULT_MODEL_DIR = '.model/main/'
* DEFAULT_RESOURCE_DIR = 'resources/'

You'll have to adapt DEFAULT_INPUT_DIR and DEFAULT_OUTPUT_DIR in <code>main.py</code>, if you want to load your input (and write your output) from (to) a different location

Place your input into DEFAULT_INPUT_DIR.

Note that all the serialized model and configuration files in <code>'.model/main/'</code> are necessary for the project to run (the model for semantic extraction used in Step 1)

### Usage
1. Configure the parameters in  <code>main.py</code> (if needed)
2. Run the project using <code>python main.py</code>

## References
* [pm4py](https://pm4py.fit.fraunhofer.de)
* [Semantic extraction approach](https://pm4py.fit.fraunhofer.de)

## Evaluation
Our approach was evaluated on a real-life event logs and a public OCEL log.
Specifically, the logs in the table below were used.

| Log               | Link |
|-------------------| ------ |
| OCEL log          | http://ocel-standard.org/1.0/running-example.jsonocel.zip |
| BPI Challenge '17 | https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b |
| BPI Challenge '19 | https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1 | 

To run it, see <code>evaluate.py</code> and adapt the global <code>REAL</code> and <code>MASKED</code> parameters.

### Data
For reproducibility, we included the flattened logs based on OCEL data  here: <code>input/evaluation/raw/</code> and the gold standard in this repository, here: <code>input/evaluation/gold/</code><br>
We fine-tuned the language model of the semantic extraction approach leaving out the data of the respective evaluation log, therefore we used 3 BERT models in the evaluation, one for the OCEL log and one for each real-life log, each has a size of approx. 500MB, so we do not upload all of them in the repository. You can train them yourself using the
code here:
[Fine-tuning BERT for semantic labeling](https://gitlab.uni-mannheim.de/processanalytics/fine-tuning-bert-for-semantic-labeling)
But you can also used the model provided in <code>'.model/main/'</code>.

### Results
The full results can be found here: <code>output/evaluation/</code>
