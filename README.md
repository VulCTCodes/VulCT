# VulCT: Scalable Vulnerability Detection by Enhanced Tree Analysis
VulCT is a fast and accurate large-scale vulnerability detection system. 
To ensure the accuracy of detection, we extend the AST by adding extra information to generate the Enhanced-AST and transform it into a straightforward state matrix using Markov chains. 
Subsequently, we convert the generated state matrix into a grayscale image and classify the image based on the traditional CNN model. 

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

The source code and dataset of Amain are published here.

# Project Structure  
  
```shell  
VulCT 
|-- ImageGeneration.py     	       // implement the first two phases:  Static Analysis and Image Generation 
|-- Classification_models.py       // implement the Classification phase  
```

### ImageGeneration.py
- Input: dataset with source codes
- Output: gray images 
```
python ImageGeneration.py -i 
```

### Classification_models.py
- Input: gray images of dataset
- Output: recall, precision, and F1 scores of CNN models
```
python Classification_models.py
```
