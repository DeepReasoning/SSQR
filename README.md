This is the source code for the paper **Self-supervised Quantized Representation for Seamlessly Integrating Knowledge Graphs with Large Language Models**. The guidelines for our codes are as follows:

1.Prepare the dataset in the data folder, which should have the same format as FB15k-237 and WN18RR, including files of entities.dict, relations.dict, train.txt, valid.txt, and test.txt.

2. Train the SSQR for quantization encoder learning: python run.py. You can revise the configuration of the run.py file for different datasets.
  
3. Generate the finetuning instructions for LLMs: python gendata4llm/gen_adaprop.py, which can be configured with different model settings and datasets.
  
4. Finetuning the LLMs with an open-instruct framework (https://github.com/allenai/open-instruct) based on the generated JSON data.


If the code is useful for you, please cite the following paper:
```
@article{DBLP:journals/corr/abs-2501-18119,
  author       = {Qika Lin and
                  Tianzhe Zhao and
                  Kai He and
                  Zhen Peng and
                  Fangzhi Xu and
                  Ling Huang and
                  Jingying Ma and
                  Mengling Feng},
  title        = {Self-supervised Quantized Representation for Seamlessly Integrating Knowledge Graphs with Large Language Models},
  booktitle    = {Proceedings of the 63st Annual Meeting of the Association for Computational Linguistics (ACL)},
  year         = {2025}
}
```
