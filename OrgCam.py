"""
**** OrgCam ***
developed by Ashraf Minhaj
mail me at- ashraf_minhaj@yahoo.com
"""
"""
Version: 0.1 
"""

"""
I'll make and executable .exe file so that this can be run on any computer.
right now it can be used by the people who has python, numpy, opencv pyautogui
installed in their pc.
"""

import cv2  
import numpy as np
import pyautogui
from time import sleep
from PIL import Image  #pip install packages
import os

#location of opencv haarcascade <change according to your file location>
face_cascade = cv2.CascadeClassifier('F:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml') 
cap = cv2.VideoCapture(0)   # 0 = main camera , 1 = extra connected webcam and so on.
rec = cv2.face.LBPHFaceRecognizer_create()

#the path where the code is saved
pathz = "C:\\Users\\HP\\cv_practice\\OrgCam" #Change this


#recogizer module
def recog():
    """ Recognizes people from the pretrained .yml file """
    #print("Starting")
    rec.read(f"{pathz}\\orgcam.yml")  #yml file location <change as yours>
    id = 0  #set id variable to zero
    num = 10000

    font = cv2.FONT_HERSHEY_COMPLEX 
    col = (255, 0, 0)
    strk = 2 

    while True:  #This is a forever loop
        ret, frame = cap.read() #Capture frame by frame 
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #change color from BGR to Gray
        faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)

        face_num = int(len(faces)) #Count faces in an image

        #print(faces)
        for(x, y, w, h) in faces:
            #print(x, y, w, h)
            roi_gray = gray[y: y+h, x: x+w]  #region of interest is face

            #*** Drawing Rectangle ***
            color = (0, 255, 0)
            stroke = 2
            end_cord_x = x+w
            end_cord_y = y+h
            cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

            #***detect
            id, conf = rec.predict(roi_gray)
            #cv2.putText(np.array(roi_gray), str(id), font, 1, col, strk)
            #print(id) #prints the id's

        
        cv2.imshow('OrgCam', frame)

        #define Person with IDS
        if id == '1' or '5':
            user = "minhaj"

        """  #uncomment and add lines for new users
        if id == '2':
            user = "Hiba nawab"

        """

        #check if user wants to quit the program (pressing 'q')
        if cv2.waitKey(10) == ord('q'):
            op = pyautogui.confirm("Close the Program 'OrgCam'?") 
            if op == 'OK':
                print("Out")
                break
         
                
        if cv2.waitKey(10) == ord('c'):   #user presses C to capture 
            if face_num < 2 and id == 5:  #check how many faces at the cam and ID
                print(f"Taking picture of {id}") 
                cv2.imwrite(f'{pathz}\\Saved_photos\\{user}.{id}.{num}.jpg', frame)
                num = num -1 
          
                

    cap.release()
    cv2.destroyAllWindows() #remove all windows we have created




#create dataset and train the model
def data_Train():
    sampleNum = 0
    #print("Starting training")
    id = pyautogui.prompt(text="""
    Enter User ID.\n\nnote: numeric data only 1 2 3 etc.""", title='OrgCam', default='none')
    #check for user input
    

    #if user input is 1 2 or 3  
    if int(id) > 0:
        pyautogui.alert(text='WRONG INPUT',title='OrgCam',button='Back')
    

    else:
        #let, the input is okay
        while True:
            
            
            ret, img = cap.read()  
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for(x, y, w, h) in faces: #find faces
                sampleNum = sampleNum + 1  #increment sample num till 21
                cv2.imwrite(f'{pathz}\\dataSet\\User.{id}.{sampleNum}.jpg', gray[y: y+h, x: x+w])
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 4)
                cv2.waitKey(100)

            cv2.imshow('faces', img)  #show image while capturing
            cv2.waitKey(1)
            if(sampleNum > 20): #21 sample is collected
                break   
            
    trainer()  #Train the model based on new images

    recog() #start recognizing
            


#Trainer 
def trainer():
    faces = []   #empty list for faces
    Ids = [] #empty list for IDs

    path = (f'{pathz}\\dataSet')

    #gets image id with path
    def getImageWithID(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        #print(f"{imagePaths}\n")
    
        for imagePath in imagePaths:
            faceImg = Image.open(imagePath).convert('L')
            #cv2.imshow('faceImg', faceImg)
            faceNp = np.array(faceImg, 'uint8')
            ID = int(os.path.split(imagePath)[-1].split('.')[1])
            #print(ID)
            faces.append(faceNp)
            Ids.append(ID)

            cv2.waitKey(10)

        return Ids, faces


    ids, faces = getImageWithID(path)

    print(ids, faces)
    rec.train(faces, np.array(ids))
    #create a yml file at the folder. WIll be created automatically.
    rec.save(f'{pathz}\\orgcam.yml')   
    pyautogui.alert("Done Saving.\nPress OK to continue")
    cv2.destroyAllWindows()
    



#Options checking
opt =pyautogui.confirm(text= 'Chose an option', title='OrgCam', buttons=['START', 'Train', 'Exit'])
if opt == 'START':
    #print("Starting the app")
    recog()
    
if opt == 'Train':
    opt = pyautogui.confirm(text="""
    Please look at the Webcam.\nTurn your head a little while capturing.\nPlease add just one face at a time.
    \nClick 'Ready' when you're ready.""", title='OrgCam', buttons=['Ready', 'Cancel'])
        
    if opt == 'Ready':
            #print("Starting image capture + Training")
            data_Train()
    if opt == 'Cancel':
        print("Cancelled")
        recog()
        

if opt == 'Exit':
    print("Quit the app")
    
