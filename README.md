# Automated-Brazilian-license-plate-reader

![Generatedexample1-ezgif (1)](https://github.com/user-attachments/assets/f2bdd510-456f-429e-ad4c-d4a4def4a42d)

A license plate reader made for Brazilian license plates (working with both old and Mercosul style).

The system uses YOLO V8 for detecting the license plates and EasyOCR for reading the Alphanumeric characters.

In order to reduce computational cost, the code does not search for plates in all frames: there is a cooldown period. Also, the ROI (Region of Interest) is reduced and the GPU is not being used.

![Generatedexample1-ezgif (2)](https://github.com/user-attachments/assets/10244426-c1a1-4287-b3c9-7b47d9c181ad)

The code has two functions:
 
- **Display:** the reading happens in real time as the video is played;
- **Generate_video:** a file with the license plate values is created.

</div>
  
  ##
 
<div> 

The YOLO V8 model was trained on this dataset: https://github.com/raysonlaroca/rodosol-alpr-dataset

Video used for demonstration: https://www.youtube.com/watch?v=7N16o4hHd-s&t=157s&ab_channel=LucasSousa

