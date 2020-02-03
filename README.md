# fcMGM

The program reqiores the existence of a file descirbing were the cytofluorometer outpufiles are placed. 

```
folder1/
AutoFl:dataAF.fcs
0:data00.fcs
folder2/
18:data18.fcs
24:data24.fcs
folder3/
42:data42.fcs
48:data48.fcs
```
The name of the file my be someting like `file.dat`.

At the first execution we will run the program with the preprocessing flag:
```
python runFcMGM.py --preprocessing --dim 3 -i file.dat
```
The flag `-dim` set the number of dimation which can be 1 to 3.
