# Delay Compensation in Multi-Input Nonlinear Systems via Neural Operator Approximate Predictors

## Introduction

TODO: Abstract from the paper (potentially modified to suits the needs)

## Getting Started - Installation, Package Versioning, Datasets, and Models.
- To get started, please setup your virtual envrionment ([Virtual Env tutorial](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)) and install the corresponding packages and versions given in `requirements.txt`.
- Additionally, we have published both the dataset and models on huggingface. Please clone the repositories below following the instructions on huggingface and place the resulting
files in the dataset and models folders (See below for structure). 
  - [Hugging face: Dataset](https://huggingface.co/datasets/FilipBajraktari/MultiInputPredictorFeedbackNeuralOperator)
  - [Hugging face: Models](https://huggingface.co/FilipBajraktari/MultiInputPredictorFeedbackNeuralOperator)
 
    
<br>

   >The repository directory should look like this:
  ```
  Multi-Input-Predictor-Feedback-Control/
  ├── config/
  ├── data/ # Place the cloned datasets here
  ├── media/
  ├── src/
  ├── models/ # Place the cloned models here
  ├── src/
  requirements.txt
  README.md
  ```


## Assistance / Troubleshooting
If you have any issues with any of the notebooks or models in this repo, feel free to create an issue in this github repo or email bajraktarifilip@gmail.com and I am more than happy to assist! 

### Citation 
If you found this work useful or interesting for your own research, we would appreciate if you could cite our work:
```
@misc{bajraktari2025multiinputpredictorfeedbackcontrol,
      title={Delay Compensation in Multi-Input Nonlinear Systems via Neural Operator Approximate Predictors}, 
      author={Filip Bajraktari and Luke Bhan and Miroslav Krstic and Yuanyuan Shi},
      year={2025},
      eprint={},
      archivePrefix={},
      primaryClass={},
      url={},
}
```

### Licensing
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.