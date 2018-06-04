- install 'TotalImageConverter'

- install 'paint.net'

- To convert pdn file to png files of its layers, use this command:
PATH>TO>TOTAL-IMAGE-CONVERTER-FOLDER>pdn2png /split PATH>TO>PDN-FILE.pdn
for example:
C:\Program Files (x86)\CoolUtils\TotalImageConverter>pdn2png /split C:\Users\"Lee Twito"\Desktop\tag-tool\mytext07.pdn
* notice that I used here > instead of \ in pdn2png.exe path
* notice that sometimes spaces make problem in paths, try to put paths in quatation marks.

- use 'tag2mask' script in order to convert the tag layer to binary mask. it also organaizes the different layers in 'images' and 'tags' folders

- use the following line in cmd to loop over all files in current folder:
for /r %i in (*) do PATH>TO>TOTAL-IMAGE-CONVERTER-FOLDER>pdn2png /split "%i"
for example:
for /r %i in (*) do "C:\Program Files (x86)\CoolUtils\TotalImageConverter\pdn2png" /split "%i"
* make sure you add quatation marks surrounding the %i

- use bmp2png.py to convert bmp files (usually healthy tissue). modify path

- move all png files to new folder using this command:
mkdir conversion-output
mv *.png conversion-output
