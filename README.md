# VulCT: Scalable Vulnerability Detection by Enhanced Tree Analysis
In recent years, with the increasing scale and complexity of software, traditional vulnerability detection methods are difficulty meeting the growing demand, and vulnerability detection is gradually advancing toward automation and intelligence. Numerous deep learning-based methods for detecting vulnerabilities have been proposed, which can achieve relatively satisfactory results in real datasets. However, these methods face difficulties in ensuring both efficiency and accuracy of vulnerability detection simultaneously and cannot be applied to large-scale real software. 

In this paper, we propose a novel enhanced tree-based vulnerability detection method, VulCT, which enables fast detection while preserving semantic features. We enrich the AST with data flow and control flow information, preserving the syntactic and semantic details of the program. Additionally, we introduce Markov chains to represent the AST in a simpler manner while maintaining its structural information. To examine the effectiveness of VulCT, we evaluate it on two widely used datasets namely FFmpeg+Qemu and Reveal. Experimental results indicate that VulCT is superior to seven state-of-the-art vulnerability detection tools (i.e., TokenCNN, VulDeePecker, SySeVR, ASTGRU, CodeBERT, Devign, and VulCNN). In terms of scalability, VuCT is ten times faster than VulCNN and 68 times faster than Devign.

# Design of VulCT
VulCT is divided into three phases: Static Analysis, Image Generation, and Classification.
1. Static Analysis: 
  The purpose of this step is to use static analysis to generate the corresponding Enhanced-AST. 
  The input of this step is the source code of a function and the output is an Enhanced-AST.
2. Image Generation: 
  The purpose of this step is to convert the Enhanced-AST into a grayscale image. 
  The input of this step is an Enhanced-AST and the output is a gray image.
3. Classification:
  This step aims to judge whether the input code is vulnerable. 
  The input to this step is a gray image and the output reports the detection result.

# Dataset
We choose two datasets FFMPeg+Qemu and Reveal obtained from the real world as the experimental datasets to evaluate the performance of VulCT. 
The dataset FFMPeg+Qemu provided by Devign is a balanced dataset obtained from the two open-source projects FFMPeg and Qemu, including about 12k vulnerable functions and 14k non-vulnerable functions. 
The ReVeal dataset proposed by ReVeal is an unbalanced dataset collected from two open-source projects Chrome and Debian. 
It consists of about 2k vulnerable functions and 20k non-vulnerable functions. 

A significant number of samples of known vulnerabilities in real software systems are present in the two datasets mentioned above. 
These vulnerability samples come from multiple sources and cover a wide range of vulnerability types (e.g., buffer overflows, format string vulnerabilities). 
For each dataset, we randomly divide it in the ratio of 8:1:1 to generate disjoint training, validation and test sets. 

# Source Code  
## Step 1: Language library build
Build a C Parser Library for Tree-sitter (We implement Tree-sitter for static analysis generates AST.)
Refer to [Tree-sitter](https://tree-sitter.github.io/tree-sitter/)
```
cd parse
bash build.sh
```

## Step2: ImageGeneration
Implementation of static analysis and image generation steps
- Input: dataset with source codes
- Output: gray images 
```
python ImageGeneration.py -i ../data/sard -o ../images/sard -t faast
```

### Step3: Classification
Implementing multiple CNN models to classify grayscale images
- Input: gray images of dataset
- Output: recall, precision, and F1 scores of CNN models
```
python Classification.py -i ../images -o ../result -t sard -m densenet
```
