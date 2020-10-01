---
title: 'OCR Pytesseract'
date: 2020-01-17
permalink: /posts/2020/01/OCR-Pytesseract/
excerpt: Optical character recognition or optical character reader (OCR) is the electronic or mechanical conversion of images of typed, handwritten or printed text into machine-encoded text, whether from a scanned document, a photo of a document, a scene-photo (for example the text on signs and billboards in a landscape photo) or from subtitle text superimposed on an image. 
tags:
  - OCR
  - Computer Vision
  - Pytesseract
  - Python
header:
  teaser: "/images/2020-01-17-OCR-Pytesseract_files/header.png"
---



Optical character recognition or optical character reader (OCR) is the electronic or mechanical conversion of images of typed, handwritten or printed text into machine-encoded text, whether from a scanned document, a photo of a document, a scene-photo (for example the text on signs and billboards in a landscape photo) or from subtitle text superimposed on an image. 
Source: [Wikipedia](https://en.wikipedia.org/wiki/Optical_character_recognition)


In layman terms, OCR is a process of extracting text from a document instead of typing the whole thing yourself. 

Document can be in form of PDF's ,Images or scanned documents .

<img src="/images/2020-01-17-OCR-Pytesseract_files/OCR_DOCS.PNG">

[Image Source](https://nanonets.com/blog/content/images/2019/11/OCR.jpg)

If we have data stored in form of documents then to do any form of analytics on the text data we need to make use of OCR to extract the data and perform text analytics on that data.

I worked on a [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) problem, where I had a bunch of employee contracts as PDF documents and I needed to extract the below entities form the documents

- Employee Name
- Employer Name
- Salary
- Location

I used OCR to extract the data in text form and applied NER on that data.

Similiarly other data science approcahes could be applied if the data is avaliable in text form.

#### Py-Tesseract

We will be using python library [pytesseract](https://pypi.org/project/pytesseract/) for OCR.

Python-tesseract (pytesseract) is a python wrapper for Google's Tesseract-OCR.

Python-tesseract is an optical character recognition (OCR) tool for python. That is, it will recognize and “read” the text embedded in images.




```python
from wand.image import Image as IM
from PIL import Image as PIM
from os import listdir
from os.path import isfile, join
import pytesseract
import argparse
import cv2
import os
import pandas as pd
import re
import time
from PyPDF2 import PdfFileWriter, PdfFileReader
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


start_time = time.time()
```

Create two folder , one for PDF file and other for images


```python
pdfPath=r"D:\works\OOCR\split_pdf"
imgPath=r"D:\works\OOCR\img_jpg"
```

Either add tesseract location in an environment variable or pass the location as below


```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
os.environ['PATH'] = os.environ['PATH'].encode('utf-8')

```

#### OCR on Image


```python
from IPython.display import Image
Image(filename='testOCR.jpg') 
```




![jpeg](/images/2020-01-17-OCR-Pytesseract_files/2020-01-17-OCR-Pytesseract_12_0.jpg)



**Tesseract in Action**


```python
img1=PIM.open("testOCR.jpg")
text = pytesseract.image_to_string(img1, lang='eng')
print(text)

```

    Now | am become
    Death, the destroyer
    of worlds
    

#### OCR on PDF file

We will be extract text data from PDF document by following below steps:-
    
1. Read PDF file.
2. Split PDF pages into seperate PDF files.
3. Convert these indivisual PDF files into images(jpg).
4. Apply OCR (pytesseract) on these image and stor these text data in CSV file.

**Download the PDF file from web**


```python
!curl "https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf" >> embeddings.pdf
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
     29  109k   29 32768    0     0  32768      0  0:00:03  0:00:01  0:00:02 24674
    100  109k  100  109k    0     0   109k      0  0:00:01  0:00:01 --:--:-- 64588
    

**Step 1 : Read PDF file by PDFFileReader**


```python
fname="embeddings.pdf"
inputpdf = PdfFileReader(open(fname,"rb"))
print('Number of PDF Pages = '+str(inputpdf.numPages))
```

    Number of PDF Pages = 9
    

**Step 2: Split these pdf pages into separate PDF files**


make sure to change the split_f_name variable based on your folder location

The below code will split the pdf file into separate PDF files


```python

for i in range(inputpdf.numPages):
    output = PdfFileWriter()
    output.addPage(inputpdf.getPage(i))
    with open(pdfPath+"\\"+fname[:fname.find('.')]+'_'+str(i)+'.pdf', "wb") as outputStream:
        output.write(outputStream)
```

List of created PDF


```python
[f for f in listdir(pdfPath) if isfile(join(pdfPath, f))]
```




    ['embeddings_0.pdf',
     'embeddings_1.pdf',
     'embeddings_2.pdf',
     'embeddings_3.pdf',
     'embeddings_4.pdf',
     'embeddings_5.pdf',
     'embeddings_6.pdf',
     'embeddings_7.pdf',
     'embeddings_8.pdf']



**Step 3: Convert PDF files to images**


```python
%%time
pdffiles = [f for f in listdir(pdfPath) if isfile(join(pdfPath, f))]
for pdfName in pdffiles:
    print "Converting PDF file" ,pdfName,"to Image..." 
    imgfname=imgPath+"\\"+pdfName[:pdfName.find('.')]+".jpg"
    print "Creating Image" ,imgfname,"..." 
    print("="*50)
    with(IM(filename=pdfPath+"\\"+pdfName,resolution=200)) as source:
        images=source.sequence
        pages=len(images)
        IM(images[i]).save(filename=imgfname)
        
            
```

    Converting PDF file embeddings_0.pdf to Image...
    Creating Image D:\works\OOCR\img_jpg\embeddings_0.jpg ...
    ==================================================
    Converting PDF file embeddings_1.pdf to Image...
    Creating Image D:\works\OOCR\img_jpg\embeddings_1.jpg ...
    ==================================================
    Converting PDF file embeddings_2.pdf to Image...
    Creating Image D:\works\OOCR\img_jpg\embeddings_2.jpg ...
    ==================================================
    Converting PDF file embeddings_3.pdf to Image...
    Creating Image D:\works\OOCR\img_jpg\embeddings_3.jpg ...
    ==================================================
    Converting PDF file embeddings_4.pdf to Image...
    Creating Image D:\works\OOCR\img_jpg\embeddings_4.jpg ...
    ==================================================
    Converting PDF file embeddings_5.pdf to Image...
    Creating Image D:\works\OOCR\img_jpg\embeddings_5.jpg ...
    ==================================================
    Converting PDF file embeddings_6.pdf to Image...
    Creating Image D:\works\OOCR\img_jpg\embeddings_6.jpg ...
    ==================================================
    Converting PDF file embeddings_7.pdf to Image...
    Creating Image D:\works\OOCR\img_jpg\embeddings_7.jpg ...
    ==================================================
    Converting PDF file embeddings_8.pdf to Image...
    Creating Image D:\works\OOCR\img_jpg\embeddings_8.jpg ...
    ==================================================
    Wall time: 7.77 s
    

List of created Images


```python
[f for f in listdir(imgPath) if isfile(join(imgPath, f))]

```




    ['embeddings_0.jpg',
     'embeddings_1.jpg',
     'embeddings_2.jpg',
     'embeddings_3.jpg',
     'embeddings_4.jpg',
     'embeddings_5.jpg',
     'embeddings_6.jpg',
     'embeddings_7.jpg',
     'embeddings_8.jpg']



**Step 4: Applying OCR on Images**


```python
%%time
contentDict={}
Imgfiles = [f for f in listdir(imgPath) if isfile(join(imgPath, f))]
for ImgName in Imgfiles:
    print "Reading Image",ImgName,'...'
    img1=PIM.open(imgPath+"\\"+ImgName)
    text = pytesseract.image_to_string(img1,lang='eng')
    contentDict[ImgName]=text
```

    Reading Image embeddings_0.jpg ...
    Reading Image embeddings_1.jpg ...
    Reading Image embeddings_2.jpg ...
    Reading Image embeddings_3.jpg ...
    Reading Image embeddings_4.jpg ...
    Reading Image embeddings_5.jpg ...
    Reading Image embeddings_6.jpg ...
    Reading Image embeddings_7.jpg ...
    Reading Image embeddings_8.jpg ...
    Wall time: 3min 31s
    


```python
contentDict["embeddings_0.jpg"][:500]
```




    u'Distributed Representations of Words and Phrases\nand their Compositionality\n\nTomas Mikolov Ilya Sutskever Kai Chen\nGoogle Inc. Google Inc. Google Inc.\nMountain View Mountain View Mountain View\nmikolov@google.com ilyasu@google.com kai@google.com\nGreg Corrado Jeffrey Dean\nGoogle Inc. Google Inc.\nMountain View Mountain View\ngcorrado@google.com jeff@google.com\nAbstract\n\nThe recently introduced continuous Skip-gram model is an efficient method for\nlearning high-quality distributed vector representati'




```python
from IPython.display import Image
Image(filename=imgPath+"\\"+"embeddings_0.jpg") 
```




![jpeg](/images/2020-01-17-OCR-Pytesseract_files/2020-01-17-OCR-Pytesseract_32_0.jpg)



Reading data in a dataframe


```python
df=pd.DataFrame(contentDict.items(), columns=['FileName', 'TextContent'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FileName</th>
      <th>TextContent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>embeddings_5.jpg</td>
      <td>Newspapers\nNew York New York Times Baltimore ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>embeddings_7.jpg</td>
      <td>Model Redmond Havel ninjutsu graffiti capitula...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>embeddings_1.jpg</td>
      <td>Input projection output\n\n \n\nw(t-2)\nif\n/ ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>embeddings_3.jpg</td>
      <td>Country and Capital Vectors Projected by PCA\n...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>embeddings_8.jpg</td>
      <td>References\n\n1] Yoshua Bengio, Réjean Ducharm...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>embeddings_4.jpg</td>
      <td>Method Time [min] | Syntactic[%] Semantic [%] ...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>embeddings_6.jpg</td>
      <td>NEG-15 with 10~° subsampling\n\nHS with 10~° s...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>embeddings_0.jpg</td>
      <td>Distributed Representations of Words and Phras...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>embeddings_2.jpg</td>
      <td>training time, The basic Skip-gram formulation...</td>
    </tr>
  </tbody>
</table>
</div>



Let's try to build a word cloud to quickly understand the document 


```python
def fxn_wordcloud(inp_val):
    text = inp_val
    wordcloud = WordCloud().generate(text)
    figure(num=None, figsize=(15, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()
```


```python
data=df.loc[df.FileName=='embeddings_0.jpg','TextContent'].values[0]
data[:300]
```




    u'Distributed Representations of Words and Phrases\nand their Compositionality\n\nTomas Mikolov Ilya Sutskever Kai Chen\nGoogle Inc. Google Inc. Google Inc.\nMountain View Mountain View Mountain View\nmikolov@google.com ilyasu@google.com kai@google.com\nGreg Corrado Jeffrey Dean\nGoogle Inc. Google Inc.\nMount'




```python
fxn_wordcloud(data)
```


![png](/images/2020-01-17-OCR-Pytesseract_files/2020-01-17-OCR-Pytesseract_38_0.png)


#### Conclusion

In this post, we learned how to apply OCR on images and PDF documents and use that data in text analytics like word cloud to quickly gain insight about the document. At times there might be some spelling mistake or gibberish output text, so please check the data before using it. On some occasions OCR was unable to read some PDF files.

#### References

[https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/](https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/)


[https://pypi.org/project/pytesseract/](https://pypi.org/project/pytesseract/)

[https://stackoverflow.com/questions/13984357/pythonmagick-cant-find-my-pdf-files/](https://stackoverflow.com/questions/13984357/pythonmagick-cant-find-my-pdf-files/)

[https://glenbambrick.com/tag/pythonmagick/](https://glenbambrick.com/tag/pythonmagick/)

[https://stackoverflow.com/questions/41353360/unable-to-install-pythonmagick-on-windows-10/](https://stackoverflow.com/questions/41353360/unable-to-install-pythonmagick-on-windows-10/)


```python

```
