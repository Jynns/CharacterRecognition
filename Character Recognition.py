from tkinter import *
from PIL import ImageTk, Image
from PIL import Image, ImageGrab
import cv2 as cv
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras

class Window(Frame):
    def __init__(self, master = None):
        Frame.__init__(self, master)
        
        self.master = master
        self.init_window()
        
    def init_window(self):
        self.isdrawing = False
        
        self.model = keras.models.load_model('model')
        self.table = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z' }
        
        self.master.title("GUI")
        self.pack(fill=BOTH, expand=1)
        self.canvas = Canvas(self)
        
        clearButton = Button(self, text = "clear", command = self.clearCanvas)
        clearButton.place(x=0, y=0)  
        
        saveButton = Button(self, text = "predict", command = self.saveCanvas)
        saveButton.place(x=50, y=0)  
        
        self.text = StringVar()
        self.predictLabel = Label( self, textvariable=self.text, relief=RAISED )
        self.text.set("")
        self.predictLabel.pack()
        
        root.bind( "<Button>", self.eMouseClicked )
        root.bind("<ButtonRelease>", self.eMouseReleased)
        root.bind("<Motion>", self.eMotion)
        
    def eMouseClicked(self, event):
        self.isdrawing = True
        
    def eMouseReleased(self,event):   
        self.isdrawing = False
        
    def eMotion(self,event):
        x, y = event.x, event.y
        if self.isdrawing:
            thickness = 20
            self.canvas.create_oval(x-thickness, y-thickness, x+thickness, y+thickness, outline="#000", fill="#000", width=2)
            self.canvas.pack(fill=BOTH, expand=1)
        
    def clientExit(self):
        root.destroy()
        
    def clearCanvas(self): 
        self.canvas.delete("all")
        
    def saveCanvas(self):
        filename = PhotoImage(file = "test123.png")
        image = self.canvas.create_image(50, 50, anchor=NE, image=filename)
        x1 = self.canvas.winfo_rootx()
        y1 = self.canvas.winfo_rooty()
        x2 = self.canvas.winfo_rootx()+self.canvas.winfo_width()
        y2 = self.canvas.winfo_rooty() + self.canvas.winfo_height()
        box = (x1,y1,x2,y2)
        im2 = ImageGrab.grab(bbox = box)
        cvImg = cv2.cvtColor(np.array(im2), cv2.COLOR_RGB2BGR)
        img = cvImg.copy()
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.GaussianBlur(img, (7,7), 0)
        #THRESH_TRUNC
        _, img = cv.threshold(img,100, 255, cv.THRESH_BINARY_INV)
        img = cv.resize(img, (28,28))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = np.reshape(img, (1,28,28,1))
        img_pred = self.table[np.argmax(self.model.predict(img))]
        self.text.set(img_pred)
        
root = Tk()
root.geometry("400x400")
app = Window(root)
root.title("Draw Letter")
root.mainloop()