<br/><br/>
<p align="center">
  <img width="250" src="https://github.com/annahedstroem/GEF/blob/52092119790e4666ee364189f16c38a5fa73eb60/logo.png">
<h3 align="center"><b>Task-agnostic Interpretability Evaluator</b></h3>
<p align="center">PyTorch</p>
<br/><br/>

This repository contains the code and experiments for the paper **["Evaluating Interpretable Methods via Geometric Alignment of Functional Distortions"]([Link](https://openreview.net/pdf?id=ukLxqA8zXj))** (with Survey Certification!) by Hedström et al., 2025. 

[![Getting started!](https://colab.research.google.com/assets/colab-badge.svg)](anonymous)
<!--![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)-->
<!--[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)-->
<!--[![PyPI version](https://badge.fury.io/py/metaquantus.svg)](https://badge.fury.io/py/metaquantus)-->
<!--[![Python package](https://github.com/annahedstroem/MetaQuantus/actions/workflows/python-publish.yml/badge.svg)](https://github.com/annahedstroem/MetaQuantus/actions/workflows/python-publish.yml/badge.svg)-->
<!--[![Launch Tutorials](https://mybinder.org/badge_logo.svg)](anonymous)-->

Please note that this repository is under active development!

## Citation

If you find this work interesting or useful in your research, use the following Bibtex annotation to cite us:

```bibtex
@article{hedstrom2025explanation,
    title={Evaluating Interpretable Methods via Geometric Alignment of Functional Distortions},
    author={
      Hedstr{\"o}m, Anna and
      Bommer, Philine Lou and
      Tom, Burns and
      Lapuschkin, Sebastian and
      Samek, Wojciech and
      H{\"o}hne, Marina M-C
    },
    journal={Transactions on Machine Learning Research},
    year={2025},
    url={https://openreview.net/forum?id=ukLxqA8zXj},
}
```

<!--This work has been published ...........-->

## Repository overview

The repository is organised as follows:
- The `src/` folder contains all necessary functions.
- The `nbs/` folder includes notebooks for generating the plots in the paper and for benchmarking experiments.
- The `assets/` folder contains all files to reproduce the experiments.
- The `tests/` folder contains the tests.

All evaluation metrics used in these experiments are implemented in [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus), a widely-used toolkit for metric-based XAI evaluation. Benchmarking is performed with tools from [MetaQuantus](https://github.com/annahedstroem/MetaQuantus/), a specialised framework for meta-evaluating metrics in explainability.

## Paper highlights 📚

</p>
<p align="center">
  <img width="800" src="https://github.com/annahedstroem/GEF/blob/bab5fd3d3e3c916028ec3a7719cb9720dc7e2e7b/algorithm.png"> 
</p>


Overview of the three-step GEF evaluation method (Algo. 1) for estimating GEF (Def. 5). 
1. First, model parameter scaling is performed to obtain a perturbed model used for evaluation.
2. Second, model and explanation distortions are computed, with the pullback operation. By integrating along the path, as represented by the blue and red areas, we capture the continuous changes in the explanation, providing a more comprehensive measure of how perturbations affect the explanation.
3. Third, distortion vectors are constructed and correlated to obtain a final GEF quality estimate.

## Installation

Install the necessary packages using the provided [requirements.txt](https://annahedstroem/gef/blob/main/requirements.txt):

```bash
pip install -r requirements.txt
```

## Package requirements 

Required packages are:

```setup
python>=3.10.1
torch>=2.0.0
quantus>=0.5.0
metaquantus>=0.0.5
captum>=0.6.0
```

### Thank you

We hope our repository is beneficial to your work and research. If you have any feedback, questions, or ideas, please feel free to raise an issue in this repository. Alternatively, you can reach out to us directly via email for more in-depth discussions or suggestions. 

📧 Contact us: 
- Anna Hedström: [hedstroem.anna@gmail.com](mailto:hedstroem.anna@gmail.com)

Thank you for your interest and support!


