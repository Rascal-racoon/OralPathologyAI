# OralPathologyAI
Applications to support oral pathology diagnosis
 
# DEMO

 
# Features
Designed specially for AI pathology of oral epithelial dysplasia, but can be applied to other organs or lesions. A unique feature is the usage of thin columnar patch images, which is convenient for training the AI by oral epithelium images of various thickness. 
You can create your original patch image data set, train your AI, and let it analyze your virtual slide images. The AI prediction is displayed as a linear heatmap. 
 
# Requirement
Necessary modules: tensorflow 2.3.0, imagesize (not essential), pysimplegui, sklearn, scipy, numpy, pandas, pillow, matplotlib, and others (os, sys, ctypes, warnings, io, re, math, random, glob, pickle, time, datetime, collections)
 
My environment: 
Core i7-8700, 3.2 GHz, 64 GB, GeForce RTX 2080
Windows 10

# Installation
Create the virtual environment "hoge" (your favorite name) using Anaconda.
Copy OralPathologyAI.py to "fuga" (your favorite folder name).

# Usage
1. Reformat and rename all your image files to jpeg files with names like "000_o.jpg", "001_o.jpg", "002_o.jpg",.... and place them in your favorite folder. 
AI reads only jpeg image files. Your virtual slide images must be converted to jpeg images of high resolution. The filenames MUST have a "_o" additive followed by ".jpg".
In this version, I expect the filenames have three digit sequential numbers such as "000", "001", "010" and "999". Therefore, all the files should be renamed like "000_o.jpg", "001_o.jpg",..., "999_o.jpg".
This is the original image ('o' for "original"). Later, when you annotate the lesions by color lines, jpeg files with additive "_i" (for "inked") are generated in the same folder.
When you draw guidelines for epithelium, jpeg files with additive "_t" (for "trace") are generated in the same folder.  

2. Run OralPathologyAI.py under the virtual environment.
Open Anaconda promt and type; 
conda activate hoge
Move to fuga folder and type;
python OralPathologyAI.py

3. At the initial run, you are prompted to select the working folder. Browse to the folder where all the virtual slide jpeg files are placed and press OK.

4. You are ready to work on the first file, presumably "000_o.jpg".

5. I am sorry but the manual is only in Japanese so far, but there are two modes "GENERATOR" and "ANNOTATOR". THe GENERATOR is to draw lines along epithelium and extract patch images perpendicular to the trace.
These patches can be stored as training data (after you finish annotation) or can be analyzed using the trained AI. The ANNOTATOR is to draw lines to label the lesions.
Black line is used to mark the epithelium, to confirm that generated patches really contain epithelium. Red, green blue lines mark three different lesions.
In my case, I used blue to mark cancer, red to mark high grade dysplasia and green to mark low grade dysplasia.  
 
# Note
If the PC have only 4 GB memory, virtual slide images of huge size cannot be loaded on memory, and the run may be stalled due to MemoryError.
This is unavoidable unless more memory is equipped on the PC. I did not experience this trouble with a PC with 8 GB memory.   

# Author
Kei Sakamoto
E-mail: s-kei.mpa@tmd.ac.jp
 
# License
None
