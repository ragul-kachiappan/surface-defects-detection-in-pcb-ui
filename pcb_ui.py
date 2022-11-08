from tkinter import *
import cv2
from PIL import Image
from PIL import ImageTk
from tkinter import ttk
import tkinter.font as fnt
import numpy as np
import os
import asyncio
from argparse import ArgumentParser
import mmdet
import mmcv
import cv2
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass


Files = []
Dir = "Experiment_1/data/test"
for root, dir_name, file_name in os.walk(Dir): 
#root store address till directory, dir_name stores directory name # file_name stores file name
    for name in file_name:
        fullName = root+'/'+name
        Files.append(fullName)


inimg = Files[0]
count = 0
countp =0
defectcount = [0] * 6
config = "work_dirs/frfpnconfigaerial/frfpnconfig.py"
checkpoint = "work_dirs/frfpnconfigaerial/epoch_12.pth"
device = "cuda:0"
score_thresh = 0.5

initimg = "flaskapi/images/whitebg.jpg"


# build the model from a config file and a checkpoint file
model = init_detector(config, checkpoint, device)


def fetch():
    # grab a reference to the image panels
    global panelA, panelB
    global inimg
    global count
    global Files
    inimg = Files[count]
    tkinimg = ImageTk.PhotoImage(Image.open(inimg).resize((300,300)))
    panelT1.configure(image=tkinimg)
    panelT1.image = tkinimg

    inimg = Files[count+1]
    tkinimg = ImageTk.PhotoImage(Image.open(inimg).resize((300,300)))
    panelT2.configure(image=tkinimg)
    panelT2.image = tkinimg

    inimg = Files[count+2]
    tkinimg = ImageTk.PhotoImage(Image.open(inimg).resize((300,300)))
    panelT3.configure(image=tkinimg)
    panelT3.image = tkinimg

    inimg = Files[count+3]
    tkinimg = ImageTk.PhotoImage(Image.open(inimg).resize((300,300)))
    panelT4.configure(image=tkinimg)
    panelT4.image = tkinimg

    inimg = Files[count+4]
    tkinimg = ImageTk.PhotoImage(Image.open(inimg).resize((300,300)))
    panelT5.configure(image=tkinimg)
    panelT5.image = tkinimg

    inimg = Files[count+5]
    tkinimg = ImageTk.PhotoImage(Image.open(inimg).resize((300,300)))
    panelT6.configure(image=tkinimg)
    panelT6.image = tkinimg
    
    count = count + 6   
            

           



def predict():
    global inimg
    global countp
    global score_thresh
    global defectcount
    global mh,mb,oc,sh,sp,spc
    defectcount = [0] * 6
    width = 300
    height = 300
    dim = (width, height)
#11111111111111111111111111111111111111111111111111    
    inimg = Files[countp]
    result = inference_detector(model, inimg)
    #label finding
    bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    scores = bboxes[:, -1]
    img = model.show_result(
        inimg,
        result,
        score_thr=score_thresh,
        show=False,
        wait_time=0,
        win_name='result',
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241))

    for i in range(len(labels)):
        if scores[i]>score_thresh:
            if labels[i] == 0:
                defectcount[0]+=1
            elif labels[i] == 1:
                defectcount[1]+=1
            elif labels[i] == 2:
                defectcount[2]+=1
            elif labels[i] == 3:
                defectcount[3]+=1
            elif labels[i] == 4:
                defectcount[4]+=1
            else:
                defectcount[5]+=1

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, dim)
    image = Image.fromarray(resized)
    tkimg = ImageTk.PhotoImage(image)
    panelP1.configure(image=tkimg)
    panelP1.image = tkimg
#22222222222222222222222222222222222222222222222222
    inimg = Files[countp+1]
    result = inference_detector(model, inimg)
    #label finding
    bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    scores = bboxes[:, -1]
    img = model.show_result(
        inimg,
        result,
        score_thr=score_thresh,
        show=False,
        wait_time=0,
        win_name='result',
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241))

    for i in range(len(labels)):
        if scores[i]>score_thresh:
            if labels[i] == 0:
                defectcount[0]+=1
            elif labels[i] == 1:
                defectcount[1]+=1
            elif labels[i] == 2:
                defectcount[2]+=1
            elif labels[i] == 3:
                defectcount[3]+=1
            elif labels[i] == 4:
                defectcount[4]+=1
            else:
                defectcount[5]+=1
                
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, dim)
    image = Image.fromarray(resized)
    tkimg = ImageTk.PhotoImage(image)
    panelP2.configure(image=tkimg)
    panelP2.image = tkimg

#3333333333333333333333333333333333333333333333333333
    inimg = Files[countp+2]
    result = inference_detector(model, inimg)
    #label finding
    bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels) 
    scores = bboxes[:, -1]
    img = model.show_result(
        inimg,
        result,
        score_thr=score_thresh,
        show=False,
        wait_time=0,
        win_name='result',
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241))

    for i in range(len(labels)):
        if scores[i]>score_thresh:
            if labels[i] == 0:
                defectcount[0]+=1
            elif labels[i] == 1:
                defectcount[1]+=1
            elif labels[i] == 2:
                defectcount[2]+=1
            elif labels[i] == 3:
                defectcount[3]+=1
            elif labels[i] == 4:
                defectcount[4]+=1
            else:
                defectcount[5]+=1
                
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, dim)
    image = Image.fromarray(resized)
    tkimg = ImageTk.PhotoImage(image)
    panelP3.configure(image=tkimg)
    panelP3.image = tkimg

#444444444444444444444444444444444444444444444444444444444
    inimg = Files[countp+3]
    result = inference_detector(model, inimg)
    #label finding
    bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    scores = bboxes[:, -1]
    img = model.show_result(
        inimg,
        result,
        score_thr=score_thresh,
        show=False,
        wait_time=0,
        win_name='result',
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241))

    for i in range(len(labels)):
        if scores[i]>score_thresh:
            if labels[i] == 0:
                defectcount[0]+=1
            elif labels[i] == 1:
                defectcount[1]+=1
            elif labels[i] == 2:
                defectcount[2]+=1
            elif labels[i] == 3:
                defectcount[3]+=1
            elif labels[i] == 4:
                defectcount[4]+=1
            else:
                defectcount[5]+=1
                
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, dim)
    image = Image.fromarray(resized)
    tkimg = ImageTk.PhotoImage(image)
    panelP4.configure(image=tkimg)
    panelP4.image = tkimg

#5555555555555555555555555555555555555555555555555555555
    inimg = Files[countp+4]
    result = inference_detector(model, inimg)
    #label finding
    bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    scores = bboxes[:, -1]
    img = model.show_result(
        inimg,
        result,
        score_thr=score_thresh,
        show=False,
        wait_time=0,
        win_name='result',
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241))

    for i in range(len(labels)):
        if scores[i]>score_thresh:
            if labels[i] == 0:
                defectcount[0]+=1
            elif labels[i] == 1:
                defectcount[1]+=1
            elif labels[i] == 2:
                defectcount[2]+=1
            elif labels[i] == 3:
                defectcount[3]+=1
            elif labels[i] == 4:
                defectcount[4]+=1
            else:
                defectcount[5]+=1
                
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, dim)
    image = Image.fromarray(resized)
    tkimg = ImageTk.PhotoImage(image)
    panelP5.configure(image=tkimg)
    panelP5.image = tkimg
    
#666666666666666666666666666666666666666666666666666
    inimg = Files[countp+5]
    result = inference_detector(model, inimg)
    #label finding
    bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    scores = bboxes[:, -1]
    img = model.show_result(
        inimg,
        result,
        score_thr=score_thresh,
        show=False,
        wait_time=0,
        win_name='result',
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241))

    for i in range(len(labels)):
        if scores[i]>score_thresh:
            if labels[i] == 0:
                defectcount[0]+=1
            elif labels[i] == 1:
                defectcount[1]+=1
            elif labels[i] == 2:
                defectcount[2]+=1
            elif labels[i] == 3:
                defectcount[3]+=1
            elif labels[i] == 4:
                defectcount[4]+=1
            else:
                defectcount[5]+=1
                
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, dim)
    image = Image.fromarray(resized)
    tkimg = ImageTk.PhotoImage(image)
    panelP6.configure(image=tkimg)
    panelP6.image = tkimg
    
    

    textvariable1.set(str(defectcount[0]))
    textvariable2.set(str(defectcount[1]))
    textvariable3.set(str(defectcount[2]))
    textvariable4.set(str(defectcount[3]))
    textvariable5.set(str(defectcount[4]))
    textvariable6.set(str(defectcount[5]))
    
    countp = countp + 6   


def changemodel():
    global config,checkpoint
    global device,model
    check = int(storage_variable.get())
    if check == 1:
        config = "work_dirs/gaconfignew/gaconfig.py"
        checkpoint = "work_dirs/gaconfignew/epoch_12.pth"
        model = init_detector(config, checkpoint, device)
    if check == 2:
        config = "work_dirs/yoloconfig/yoloconfig.py"
        checkpoint = "work_dirs/yoloconfig/latest.pth"
        model = init_detector(config, checkpoint, device)

def changescore():
    global score_thresh
    score_thresh = float(score_valuechange.get())
    score_value.set(str(score_thresh))
#Initialize the main window
root = Tk()
root.configure(background='black')
root.title("PCB inspection system")
root.tk.call('lappend', 'auto_path', 'flaskapi/awthemes-10.4.0')
root.tk.call('package', 'require', 'awdark')
style = ttk.Style(root)
print(style.theme_use("awdark"))
panelT1 = None
panelT2 = None
panelT3 = None
panelT4 = None
panelT5 = None
panelT6 = None


panelP1 = None
panelP2 = None
panelP3 = None
panelP4 = None
panelP5 = None
panelP6 = None




tkinimg = ImageTk.PhotoImage(Image.open(initimg).resize((300,300)))
panelT1 = Label(image=tkinimg, text="Test - 1", bg="grey")
panelT1.image = tkinimg

panelT2 = Label(image=tkinimg, text="Test - 2", bg="grey")
panelT2.image = tkinimg

panelT3 = Label(image=tkinimg, text="Test - 3", bg="grey")
panelT3.image = tkinimg

panelT4 = Label(image=tkinimg, text="Test - 4", bg="grey")
panelT4.image = tkinimg

panelT5 = Label(image=tkinimg, text="Test - 5", bg="grey")
panelT5.image = tkinimg

panelT6 = Label(image=tkinimg, text="Test - 6", bg="grey")
panelT6.image = tkinimg



panelP1 = Label(image=tkinimg, text="Predict - 1", bg="grey")
panelP1.image = tkinimg

panelP2 = Label(image=tkinimg, text="Predict - 2", bg="grey")
panelP2.image = tkinimg

panelP3 = Label(image=tkinimg, text="Predict - 3", bg="grey")
panelP3.image = tkinimg

panelP4 = Label(image=tkinimg, text="Predict - 4", bg="grey")
panelP4.image = tkinimg

panelP5 = Label(image=tkinimg, text="Predict - 5", bg="grey")
panelP5.image = tkinimg

panelP6 = Label(image=tkinimg, text="Predict - 6", bg="grey")
panelP6.image = tkinimg


panelT1.grid(column=0,row=1, padx=5, pady=5)
panelT2.grid(column=0,row=2, padx=5, pady=5)
panelT3.grid(column=0,row=3, padx=5, pady=5)
panelT4.grid(column=2,row=1, padx=5, pady=5)
panelT5.grid(column=2,row=2, padx=5, pady=5)
panelT6.grid(column=2,row=3, padx=5, pady=5)

panelP1.grid(column=1,row=1, padx=5, pady=5)
panelP2.grid(column=1,row=2, padx=5, pady=5)
panelP3.grid(column=1,row=3, padx=5, pady=5)
panelP4.grid(column=3,row=1, padx=5, pady=5)
panelP5.grid(column=3,row=2, padx=5, pady=5)
panelP6.grid(column=3,row=3, padx=5, pady=5)


lbt1 = ttk.Label(root,borderwidth=2, relief="groove",font=("Segoe UI", 10))
lbt1.grid(column=4, row=1,sticky='w'+'e'+'n'+'s')



storage_variable = StringVar()

option_one = ttk.Radiobutton(
	lbt1,
	text="Faster RCNN with FPN",
	variable=storage_variable,
	value="0"
)

option_two = ttk.Radiobutton(
	lbt1,
	text="Faster RCNN with FPN and GARPN                 ",
	variable=storage_variable,
	value="1"
)

option_three = ttk.Radiobutton(
	lbt1,
	text="YOLOv3",
	variable=storage_variable,
	value="2"
)


ilb1 = ttk.Label(lbt1, text="Model Options",anchor=CENTER,font=("Segoe UI", 10))
ilb1.grid(row=0,sticky='w'+'e'+'n'+'s',pady=15,padx=3)
option_one.grid(row=1,sticky='w'+'e'+'n'+'s',pady=15,padx=3)
option_two.grid(row=2,sticky='w'+'e'+'n'+'s',pady=15,padx=3)
option_three.grid(row=3,sticky='w'+'e'+'n'+'s',pady=15,padx=3)


lbt2 = ttk.Label(root,borderwidth=2, relief="groove",font=("Segoe UI", 10))
lbt2.grid(column=4, row=2,sticky='w'+'e'+'n'+'s')
ilb2 = ttk.Label(lbt2, text="Defects",anchor=CENTER,font=("Segoe UI", 10))
ilb2.grid(row=0,column=0,columnspan=2,sticky='w'+'e'+'n'+'s',padx=13,pady=3)

textvariable1 = StringVar(value="0")
mhlabel = ttk.Label(lbt2, text="Missing Hole",font=("Segoe UI", 10))
mh = ttk.Label(lbt2, textvariable=textvariable1,borderwidth=2, relief="sunken",background="black",width=3,font=("Segoe UI", 10))
mhlabel.grid(row=1,column=0,sticky='w'+'e'+'n'+'s',padx=7,pady=7)
mh.grid(row=1,column=1,sticky='w'+'e'+'n'+'s',padx=7,pady=7)

textvariable2 = StringVar(value="0")
mblabel = ttk.Label(lbt2, text="Mouse Bite",font=("Segoe UI", 10))
mb = ttk.Label(lbt2, textvariable=textvariable2,borderwidth=2, relief="sunken",background="black",width=3,font=("Segoe UI", 10))
mblabel.grid(row=2,column=0,sticky='w'+'e'+'n'+'s',padx=7,pady=7)
mb.grid(row=2,column=1,sticky='w'+'e'+'n'+'s',padx=7,pady=7)

textvariable3 = StringVar(value="0")
oclabel = ttk.Label(lbt2, text="Open Circuit",font=("Segoe UI", 10))
oc = ttk.Label(lbt2, textvariable=textvariable3,borderwidth=2, relief="sunken",background="black",width=3,font=("Segoe UI", 10))
oclabel.grid(row=3,column=0,sticky='w'+'e'+'n'+'s',padx=7,pady=7)
oc.grid(row=3,column=1,sticky='w'+'e'+'n'+'s',padx=7,pady=7)

textvariable4 = StringVar(value="0")
shlabel = ttk.Label(lbt2, text="Short Circuit",font=("Segoe UI", 10))
sh = ttk.Label(lbt2, textvariable=textvariable4,borderwidth=2, relief="sunken",background="black",width=3,font=("Segoe UI", 10))
shlabel.grid(row=4,column=0,sticky='w'+'e'+'n'+'s',padx=7,pady=7)
sh.grid(row=4,column=1,sticky='w'+'e'+'n'+'s',padx=7,pady=7)

textvariable5 = StringVar(value="0")
splabel = ttk.Label(lbt2, text="Spur",font=("Segoe UI", 10))
sp = ttk.Label(lbt2, textvariable=textvariable5,borderwidth=2, relief="sunken",background="black",width=3,font=("Segoe UI", 10))
splabel.grid(row=5,column=0,sticky='w'+'e'+'n'+'s',padx=7,pady=7)
sp.grid(row=5,column=1,sticky='w'+'e'+'n'+'s',padx=7,pady=7)

textvariable6 = StringVar(value="0")
spclabel = ttk.Label(lbt2, text="Spurious Copper",font=("Segoe UI", 10))
spc = ttk.Label(lbt2, textvariable=textvariable6,borderwidth=2, relief="sunken",background="black",width=3,font=("Segoe UI", 10))
spclabel.grid(row=6,column=0,sticky='w'+'e'+'n'+'s',padx=7,pady=7)
spc.grid(row=6,column=1,sticky='w'+'e'+'n'+'s',padx=7,pady=7)

lb1 = ttk.Label(root, text="Test",anchor=CENTER,borderwidth=2, relief="groove",font=("Segoe UI", 12))
lb1.grid(column=0, row=0,sticky='w'+'e'+'n'+'s')
lb1 = ttk.Label(root, text="Prediction",anchor=CENTER,borderwidth=2, relief="groove",font=("Segoe UI", 12))
lb1.grid(column=1, row=0,sticky='w'+'e'+'n'+'s')
lb1 = ttk.Label(root, text="Test",anchor=CENTER,borderwidth=2, relief="groove",font=("Segoe UI", 12))
lb1.grid(column=2, row=0,sticky='w'+'e'+'n'+'s')
lb1 = ttk.Label(root, text="Prediction",anchor=CENTER,borderwidth=2, relief="groove",font=("Segoe UI", 12))
lb1.grid(column=3, row=0,sticky='w'+'e'+'n'+'s')
lb1 = ttk.Label(root, text="Tools and Measures",anchor=CENTER,borderwidth=2, relief="groove",font=("Segoe UI", 12))
lb1.grid(column=4, row=0,sticky='w'+'e'+'n'+'s')


lbt3 = ttk.Label(root,borderwidth=2, relief="groove",font=("Segoe UI", 10))
lbt3.grid(column=4, row=3,sticky='w'+'e'+'n'+'s')

score_value = StringVar(value="0.5")
score_valuechange = StringVar()
scorelabel = ttk.Label(lbt3, text="Current score threshold : ",font=("Segoe UI", 10))
score = ttk.Label(lbt3, textvariable=score_value,font=("Segoe UI", 10))
scorelabel.grid(row=0,column=0,sticky='w'+'e'+'n'+'s',padx=7,pady=7)
score.grid(row=0,column=1,sticky='w'+'e'+'n'+'s',padx=7,pady=7)

scorechangelabel = ttk.Label(lbt3, text="Change Threshold Score : ",font=("Segoe UI", 10))
scorechange = ttk.Entry(lbt3, width=10, textvariable=score_valuechange)
scorechangelabel.grid(row=1,column=0,sticky='w'+'e'+'n'+'s',padx=7,pady=7)
scorechange.grid(row=1,column=1,sticky='w'+'e'+'n'+'s',padx=7,pady=7)
#User buttons

btn3 = ttk.Button(lbt1, text="Change Model", command=changemodel)
btn3.grid(row=4,sticky='w'+'e'+'n'+'s',pady=15,padx=5)
btn4 = ttk.Button(lbt3, text="Change Score", command=changescore)
btn4.grid(row=2,columnspan=2,sticky='w'+'e'+'n'+'s',pady=15,padx=5)
btn2 = ttk.Button(lbt3, text="Run pipeline", command=predict)
btn2.grid(row=4, columnspan=2,sticky='w'+'e'+'n'+'s',pady=15,padx=5)
btn = ttk.Button(lbt3, text="Fetch Board", command=fetch)
btn.grid(row=3, columnspan=2,sticky='w'+'e'+'n'+'s',pady=15,padx=5)

root.resizable(0,0)
# kick off the GUI
root.mainloop()
