import numpy as np
import  cv2

class Prava:
    def __init__(self,x1,y1,x2,y2,k):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.k = k

    def __str__(self):
        return "Prava([x1 = {0},y1 = {1}, x2 ={2}, y2 = {3} , k = {4}])".format(self.x1, self.y1, self.x2, self.y2, self.k)

def izracunajK(x1,y1,x2,y2):
    return round((float(y2) -y1) / (x2-x1),3)

def dajLinije(ko,img):
    kernel = np.ones((3,3),np.uint8)
    gray = cv2.dilate(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),kernel)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 100, 6)
    svePrave = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            k = izracunajK(x1,y1,x2,y2)
            svePrave.append(Prava(x1,y1,x2,y2,k))

    A = sorted(svePrave, key=lambda x: x.k)
    razlika = 0
    indeks1 = 0
    indeks2=0

    for i in range(len(A) -1):
        r = A[i].k - A[i+1].k
        if r < razlika:
            razlika = r
            indeks1 = i
            indeks2 = i+1

    prvi = A[:indeks1+1]
    drugi = A[indeks2:]

    P1 = sorted(prvi, key=lambda b: b.x1)
    P2 = sorted(prvi, key=lambda b: b.x2)
    D1 = sorted(drugi, key=lambda b: b.x1)
    D2 = sorted(drugi, key=lambda b: b.x2 )

    k = izracunajK(P1[0].x1,P1[0].y1,P2[len(P2) - 1].x2,P2[len(P2) - 1].y2)
    sabiranje = Prava(P1[0].x1,P1[0].y1,P2[len(P2) - 1].x2,P2[len(P2) - 1].y2,k)
    k1 = izracunajK(D1[0].x1,D1[0].y1,D2[len(D2) - 1].x2,D2[len(D2) - 1].y2)
    oduzimanje = Prava(D1[0].x1,D1[0].y1,D2[len(D2) - 1].x2,D2[len(D2) - 1].y2,k1)

    return  sabiranje,oduzimanje

for vidNum in np.arange(0, 10):
    suma = 0
    k = 0
    n = 0
    frameNum = 0
    cap = cv2.VideoCapture("klipovi/video-" + str(vidNum) + ".avi")
    print("Trenutni video",vidNum)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    framebr = 0
    while(cap.isOpened()):
        framebr +=1
        ret, frame = cap.read()
        img = frame
        if framebr == 1:

            sabiranje , oduzimanje = dajLinije(vidNum,frame)
            print("SABIRANJE",str(sabiranje))
            print("ODUZIMANJE",str(oduzimanje))
            h,w,_ = frame.shape
            for i in range(h):
                for j in range(w):
                    bgr = frame[i][j]
                    if (bgr[0] < 200 or bgr[1] < 200 or bgr[2] < 200):
                        frame[i][j] = [0,0,0]

            rez = cv2.dilate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).copy(), np.ones((3, 3)))

            a,b,c = cv2.findContours(rez,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            height, width = rez.shape
            min_x, min_y = width, height
            max_x = max_y = 0

            for pravougaonik in b:
                if ( cv2.contourArea(pravougaonik) > 55 ):
                    (x,y,w,h) = cv2.boundingRect(pravougaonik)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)

        if ret == True:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
