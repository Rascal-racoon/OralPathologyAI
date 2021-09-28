# OralPathologyAI
Application to support oral pathology diagnosis
 
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

Since epithelial dysplasia have a polarity and various thickeness, thin columnar patch images are more convenient than square patches.  
![slide1](https://user-images.githubusercontent.com/38546255/135052671-d540c58e-5e3c-4702-a788-7183939de318.png)  
Lesions are annotated simply by color lines prior to patch collection, reducing the labor for annotation.  
![slide2](https://user-images.githubusercontent.com/38546255/135054602-33c0d365-9eb4-43b4-9bf1-fc5e640e886a.png)  
AI prediction is displayed as a heatmap.  
![slide3](https://user-images.githubusercontent.com/38546255/135054618-104baa3e-5295-4acd-a8c9-bc94fd864f99.png)  
This is an example using the weight matrix [0, 1, 2, 3].  
![slide4](https://user-images.githubusercontent.com/38546255/135054629-880a7c78-8950-400e-9c73-30eb64546755.png)  
Weight matrix [0, 1, 2, 3] is not good for grading dysplasia.  
![slide5](https://user-images.githubusercontent.com/38546255/135054635-5a488f38-44df-4ab8-9171-08e7a83051d3.png)  
Weight matrix is adjusted to focus on the difference of LGD/HGD and HGD/Cancer.  
![slide6](https://user-images.githubusercontent.com/38546255/135054659-2fbf7b43-fe79-4b51-a58b-f22b93067332.png)  
Strong signal appears in the lower heatmap (W = [0,0,0,1]) in cancer.  
![slide7](https://user-images.githubusercontent.com/38546255/135054670-d49dd737-646b-486c-a6fc-202a538ca6d5.png)  
In high grade dysplasia, strong signal appears in the upper heatmap, while no signal appears in the lower heatmap.  
![slide8](https://user-images.githubusercontent.com/38546255/135054677-59e9ddb4-71c2-431e-b543-697882d6d590.png)  
In low grade dysplasia, only weak signal appears in the upper heatmap.  
![slide9](https://user-images.githubusercontent.com/38546255/135054700-3e100596-67f8-4180-b907-a965d0d61fcf.png)  
In our analysis, concordance rate between heatmap diagnosis (diagnosis determined only by a heatmap) and the real diagnosis was 89%.  



 
# Requirement
Python 3.8  
Necessary modules: tensorflow 2.3.0, imagesize (not essential), pysimplegui, sklearn, scipy, numpy, pandas, pillow, matplotlib, and others (os, sys, ctypes, warnings, io, re, math, random, glob, pickle, time, datetime, collections)
 
My environment: 
Core i7-8700, 3.2 GHz, 64 GB, GeForce RTX 2080
Windows 10

# Installation
Create the virtual environment "hoge" (your favorite name) using Anaconda.
Copy OralPathAI.py to "fuga" (your favorite folder name).

# Usage
1. Reformat and rename all your image files to jpeg files with names like "000_o.jpg", "001_o.jpg", "002_o.jpg",.... and place them in your favorite folder.   
AI reads only jpeg image files. Your virtual slide images must be converted to jpeg images of high resolution. The filenames MUST have a "_o" additive followed by ".jpg".
In this version, I expect the filenames have three digit sequential numbers such as "000", "001", "010" and "999". Therefore, all the files should be renamed like "000_o.jpg", "001_o.jpg",..., "999_o.jpg".
This is the original image ('o' for "original"). Later, when you annotate the lesions by color lines, jpeg files with additive "_i" (for "inked") are generated in the same folder.
When you draw guidelines for epithelium, jpeg files with additive "_t" (for "trace") are generated in the same folder.  

2. Run oralPathAI.py under the virtual environment.
Open Anaconda promt and type;  
conda activate hoge  
Move to fuga folder and type;  
python oralPathAI.py  

3. At the initial run, you are prompted to select the working folder. Browse to the folder where all the virtual slide jpeg files are placed and press OK.

4. You are ready to work on the first file, presumably "000_o.jpg".

5. I am sorry but the manual has not been finished so far. There are two modes "Generator" and "Annotator". THe Generator is to draw lines along epithelium and extract patch images perpendicular to the trace.
These patches can be stored as training data (after you finish annotation) or can be analyzed using the trained AI. The Annotator is to draw lines to label the lesions.
Black line is used to mark the epithelium, to confirm that generated patches really contain epithelium. Red, green blue lines mark three different lesions.
In my case, I used blue to mark cancer, red to mark high grade dysplasia and green to mark low grade dysplasia.You can enlarge the image by a wheel turn but this can be very slow in case of a huge virtual slide image.
Instead use the subwindow that pops up by right click and select the magnification. 1x is the largest where 1 dot on the memory is shown on 1 dot on the screen (actually it should be shown 1:1 but I just used x1).
You can enlarge or reduce on the subwindow by wheel turn.    

6. AI_tongueBiopsy_60x1000.h5 is the model parameters of a neural network trained to recognize [normal, low grade dysplasia, high grade dysplasia, cancer] in 60x1000 pixel patches.
To use this AI, patch size must be set width=60, height=1000. 
 
# Note
If the PC have only 4 GB memory, virtual slide images of huge size cannot be loaded on memory, and the run may be stalled due to MemoryError.
This is unavoidable unless more memory is equipped on the PC. I did not experience this trouble with a PC with 8 GB memory.   

# Author
Kei Sakamoto  
E-mail: s-kei.mpa@tmd.ac.jp
 
# License
None
