# OralPathologyAI
Applications to support oral pathology diagnosis
 
# DEMO
Drawing guidelines to extract patches containing epithelium.  
![demo1](https://user-images.githubusercontent.com/38546255/133545516-4c359c4a-b98a-4a4e-a2b7-8275e496f31c.png)
  
Training AI with your choice of hyperparameters.  
![demo2](https://user-images.githubusercontent.com/38546255/133545910-ea9e61e8-d52c-40dc-84f0-c62559243e67.png) 
  
Analyzing a tongue biopsy specimen using pretrained AI and display as an atypia heatmap.  
![demo3](https://user-images.githubusercontent.com/38546255/133545972-8a668f40-b9a1-4d1c-8e50-4c2a06911df5.png)

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

2. Run oralPathologyAI.py under the virtual environment.
Open Anaconda promt and type; 
conda activate hoge
Move to fuga folder and type;
python oralPathologyAI.py

3. At the initial run, you are prompted to select the working folder. Browse to the folder where all the virtual slide jpeg files are placed and press OK.

4. You are ready to work on the first file, presumably "000_o.jpg".

5. I am sorry but the manual has not been finished so far. There are two modes "Generator" and "Annotator". THe Generator is to draw lines along epithelium and extract patch images perpendicular to the trace.
These patches can be stored as training data (after you finish annotation) or can be analyzed using the trained AI. The Annotator is to draw lines to label the lesions.
Black line is used to mark the epithelium, to confirm that generated patches really contain epithelium. Red, green blue lines mark three different lesions.
In my case, I used blue to mark cancer, red to mark high grade dysplasia and green to mark low grade dysplasia.You can enlarge the image by a wheel turn but this can be very slow in case of a huge virtual slide image.
Instead use the subwindow that pops up by right click and select the magnification. 1x is the largest where 1 dot on the memory is shown on 1 dot on the screen (actually it should be shown 1:1 but I just used x1).
You can enlarge or reduce on the subwindow by wheel turn.    
 
# Note
If the PC have only 4 GB memory, virtual slide images of huge size cannot be loaded on memory, and the run may be stalled due to MemoryError.
This is unavoidable unless more memory is equipped on the PC. I did not experience this trouble with a PC with 8 GB memory.   

# Author
Kei Sakamoto
E-mail: s-kei.mpa@tmd.ac.jp
 
# License
None
