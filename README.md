# VulCT: Scalable Vulnerability Detection by Enhanced Tree Analysis
VulCT is a fast and accurate large-scale vulnerability detection system. 
To ensure the accuracy of detection, we extend the AST by adding extra information to generate the Enhanced-AST and transform it into a straightforward state matrix using Markov chains. 
Subsequently, we convert the generated state matrix into a grayscale image and classify the image based on the traditional CNN model. 

VulCT is divided into three phases: Static Analysis, Image Generation, and Classification.
1. Static Analysis: 
  The purpose of this step is to use static analysis to generate the corresponding Enhanced-AST. 
  The input of this step is a code fragment and the output is an Enhanced-AST.
  
2. Image Generation: 
  The purpose of this step is to convert the Enhanced-AST into a grayscale image. 
  The input of this step is an Enhanced-AST and the output is a gray image.
  
3. Classification:
  This phase aims to extract the feature of each node with code and structure information.
  The input of this phase is the merged PDG and the output is the vectors corresponding to each node in the merged PDG.
  
The source code and dataset of VulCT will be published here after the paper is accepted.

