import tkinter as tk 
from tkinter import Message, Text 
import cv2 
import os 
import shutil 
import csv 
import numpy as np 
from PIL import Image, ImageTk 
import pandas as pd 
import datetime 
import time 
import tkinter.ttk as ttk 
import tkinter.font as font 
from urllib.request import urlopen 
from ssl import SSLContext, PROTOCOL_TLSv1 
window = tk.Tk() 
window.title("Face_Recognizer Attendance System")
dialog_title = 'QUIT'
window.geometry('1366x768')
window.configure() #background='grey')
#window.configure(background='Red')
#window.attributes('-fullscreen', True)
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
message = tk.Label(window, text="Face-Recognition-Attendance-System", bg="grey",
                   fg="black", width=50, height=3, font=('arial', 30, 'italic bold underline'))
message.place(x=80, y=20)
lbl = tk.Label(window, text="EMP ID", width=20, height=2, 
               fg="white", bg="green", font=('times', 15, 'bold'))
lbl.place(x=400, y=200)
txt = tk.Entry(window, width=20, bg="green",
               fg="white", font=('times', 15, 'bold'))
txt.place(x=700, y=215)
lb12 = tk.Label(window, text="Employee Name", width=20, fg="white",
                bg="green", height=2, font=('times', 15, 'bold'))
lb12.place(x=400, y=300)
txt2 = tk.Entry(window, width=20, bg="green",
                fg="white", font=('times', 15, 'bold'))
txt2.place(x=700, y=315)
lb13 = tk.Label(window, text="Notification: ", width=20, fg="white", 
                bg="green", height=2, font=('times', 15, 'bold'))
lb13.place(x=400, y=400)
message = tk.Label(window, text="", bg="green", fg="white", width=30, height=2,
                   activebackground="yellow", font=('times', 15, 'bold'))
message.place(x=700, y=400)
lb13 = tk.Label(window, text="Attendance: ", width=20, fg="white", bg="green", 
                height=2, font=('times', 15, 'bold underline'))
lb13.place(x=400, y=600)
message2 = tk.Label(window, text="", fg="white", bg="green", 
                    activeforeground="green", width=30, height=2, font=('times', 15, 'bold'))
message2.place(x=700, y=600)
def clear():
    txt.delete(0, 'end')
    res = ""
    message.configure(text=res)
def clear2():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text=res)
def is_number(s):
    try:
        float(s)
        return True 
    except ValueError:
        pass 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True 
    except (TypeError, ValueError):
        pass 
    return False 
def TakeImages():
    Id = (txt.get())
    name = (txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0 
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
                sampleNum = sampleNum+1
                cv2.imwrite("TrainingImage\ "+name+ "."+Id + '.' + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
                cv2.imshow('frame', img)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break 
                elif sampleNum > 100:
                    break 
            cam.release()
            cv2.destroyAllWindows()
            res = "Images Saved for ID: "+ Id + "Name: " + name 
            row = [Id, name]
            with open('EmployeeDetails\EmployeeDetails.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
                csv.File.close()
                message.configure(text=res)
    else: 
            if(is_number(Id)):
                res = "Enter Alphabetical Name"
                message.configure(text=res)
            if(name.isalpha()):
                res = "Enter Numeric Id"
                message.configure(text=res)
def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("image")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"
    message.configure(text=res)
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
        return faces, Ids 
def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("EmployeeDetails\EmployeeDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    while(True):
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for(x,y,w,h) in faces:
            cv2.rectangle(im, (x,y), (x+w, y+h), (255, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x: x+w])
            if(conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id']] == Id['Name'].values 
                tt = str(Id)+"-"+aa 
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
            else:
                Id = 'Unknown'
                tt = str(Id)
                if(conf > 75):
                    noOfFile = len(os.listdir("ImagesUnknown"))+1 
                    cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h, x: x+w])
                    cv2.putText(im, str(tt), (x,y+h), font, 1, (255, 255, 255), 2)
                attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
                cv2.imshow('im', im)
                if (cv2.waitKey(10000)):
                    break 
                if (cv2.waitKey(1) == ord('q')):
                    break 
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            Hour, Minute, Second = timeStamp.split(":")
            fileName = "Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
            attendance.to_csv(fileName, index=False)
            cam.release()
            cv2.destroyAllWindows()
            res = attendance 
            message2.configure(text=res)
clearButton = tk.Button(window, text="Clear", command=clear, fg="red", bg="yellow",
                        width=20, height=2, activebackground="Red", font=('times', 15, ' bold '))
clearButton.place(x=950, y=200)
clearButton2 = tk.Button(window, text="Clear", command=clear2, fg="red", bg="yellow",
                         width=20, height=2, activebackground="Red", font=('times', 15, ' bold '))
clearButton2.place(x=950, y=300)
takeImg = tk.Button(window, text="Take Images", command=TakeImages, fg="red", bg="yellow",
                    width=20, height=3, activebackground="Red", font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)
trainImg = tk.Button(window, text="Train Images", command=TrainImages, fg="red",
                     bg="yellow", width=20, height=3, activebackground="Red", font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Track Images", command=TrackImages, fg="red",
                     bg="yellow", width=20, height=3, activebackground="Red", font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=quit, fg="red", bg="yellow",
                       width=20, height=3, activebackground="Red", font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)
window.mainloop()