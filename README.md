# RPEval: Role-Playing Evaluation for Large Language Models

<p align="center">
  <a href="https://huggingface.co/spaces/yelboudouri/RPEval">
    <img src="https://img.shields.io/badge/HuggingFace-Leaderboard-orange" alt="HuggingFace Leaderboard">
  </a>
</p>

*This repository contains code and data referenced in: ["Role-Playing Evaluation for Large Language Models"](https://arxiv.org/abs/2505.13157).*

Large Language Models (LLMs) demonstrate a notable capacity for adopting personas and engaging in role-playing. However,
evaluating this ability presents significant challenges, as human assessments are resource-intensive and automated
evaluations can be biased. To address this, we introduce Role-Playing Eval (RPEval), a novel benchmark designed to
assess LLM role-playing capabilities across four key dimensions: emotional understanding, decision-making, moral
alignment, and in-character consistency.

## Getting Started

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yelboudouri/RPEval.git
cd RPEval
pip install -r requirements.txt
```


## Reproducing Paper Results

To reproduce the evaluation results from the paper:

```bash
python eval.py --responses-file=data/responses_gpt_4o_2024_08_06.jsonl
```

To test other models, simply change the `--responses-file` argument to the appropriate file under the `data/` directory.


## Evaluating a New Model

To run RPEval on a different model:

```bash
python eval.py --provider="<provider_name>" --model="<model_name>"
```

RPEval uses [SwitchAI](https://github.com/yelboudouri/SwitchAI) under the hood. Ensure your API key is properly configured and the target model is supported.


## Reference

If you use this code in your research, please cite the following paper:

```bibtex
@misc{boudouri2025roleplayingevaluationlargelanguage,
      title={Role-Playing Evaluation for Large Language Models}, 
      author={Yassine El Boudouri and Walter Nuninger and Julian Alvarez and Yvan Peter},
      year={2025},
      eprint={2505.13157},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.13157}, 
}
```
