import numpy as np
import  cv2
import cProfile
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from keras.models import model_from_json
from random import randint
from math import sqrt
from sympy import Point, Line

class Prepoznat:
    def __init__(self,frejm,vrednost,x,y):
        self.frejm = frejm
        self.vrednost = vrednost
        self.x = x
        self.y = y

class Prava:
    def __init__(self,x1,y1,x2,y2,k):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.k = k

    def __str__(self):
        return "Prava([x1 = {0},y1 = {1}, x2 ={2}, y2 = {3} , k = {4}])".format(self.x1, self.y1, self.x2, self.y2, self.k)

    def dist(self):
        return sqrt( (self.x2-self.x1)*(self.x2-self.x1)  + (self.y2-self.y1) * (self.y2-self.y1)     )

def izracunajK(x1,y1,x2,y2):
    return round((float(y2) -y1) / (x2-x1),3)


def dajLinije(ko,img):
    kernel = np.ones((3,3),np.uint8)
    gray = cv2.dilate(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),kernel)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=10, lines=np.array([]), minLineLength=200, maxLineGap=20)
    svePrave = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            k = izracunajK(x1,y1,x2,y2)

            p = Prava(x1,y1,x2,y2,k)

            svePrave.append(p)

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

    P1 = sorted(prvi, key=lambda b: b.y2)

    D1 = sorted(drugi, key=lambda b: b.y2)

    sabiranje = P1[len(P1)-1 ]

    oduzimanje =D1[len(D1) - 1]

    cv2.line(img, (sabiranje.x1,sabiranje.y1), (sabiranje.x2,sabiranje.y2),(0,0,255),2)
    cv2.line(img, (oduzimanje.x1,oduzimanje.y1), (oduzimanje.x2,oduzimanje.y2),(0,0,255),2)
    return  sabiranje,oduzimanje

# izvor # https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html
def pnpoly(vertx, verty, testx, testy):
   nvert = 4

   nalazi = False

   for i in range(0,nvert):
       j = (i+nvert-1) % nvert
       if ( ((verty[i]>testy) != (verty[j]>testy)) and (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) ):
           nalazi = not(nalazi)

   return nalazi



def distance_to_line(p, p1, p2):

        x_diff = p2[0] - p1[0]
        y_diff = p2[1] - p1[1]
        num = abs(y_diff*p[0] - x_diff*p[1] + p2[0]*p1[1] - p2[1]*p1[0])
        den = sqrt(y_diff**2 + x_diff**2)
        d = num / den

        return d


def nijeObradjen(brojevi):
    if len(brojevi) == 1:
        return True
    poslednji = brojevi[ len(brojevi) -1 ]

    for i in range(0,len(brojevi) - 1):
        t = brojevi[i]
        if (t.frejm + 10 >=poslednji.frejm):
            if (abs(t.x-poslednji.x) < 10 and abs(t.y-poslednji.y) < 10):
                if t.vrednost == poslednji.vrednost:
                    return False

    return True

def main():
    start = timer()


    json_file = open('Mmodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("Mmodel.h5")

    Svi = []
    for vidNum in np.arange(0, 10):
        prepoznatiS = []
        prepoznatiO = []
        suma = 0
        cap = cv2.VideoCapture("klipovi/video-" + str(vidNum) + ".avi")
        #print("Trenutni video",vidNum)
        framebr = 0
        sabiranje = []
        oduzimanje = []

        while(cap.isOpened()):
            framebr +=1
            ret, frame = cap.read()

            if framebr == 1:

                sabiranje , oduzimanje = dajLinije(vidNum,frame)
               # print("SABIRANJE",str(sabiranje))
                #print("ODUZIMANJE",str(oduzimanje))
                k = izracunajK(sabiranje.x1,sabiranje.y1,sabiranje.x2,sabiranje.y2)
                n = - (k*sabiranje.x1) + sabiranje.y1

                ko = izracunajK(oduzimanje.x1,oduzimanje.y1,oduzimanje.x2,oduzimanje.y2)
                no = - (ko*oduzimanje.x1) + oduzimanje.y1

            lower = np.array([200, 200, 200])
            upper = np.array([255, 255, 255])
            if (framebr < 1200):
                shapeMask = cv2.inRange(frame, lower, upper)
            else:
                #print("Kraj videa")
                break

            rez = cv2.dilate(shapeMask, (3, 3))

            a,b,c = cv2.findContours(rez,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            for pravougaonik in b:

                (x,y,w,h) = cv2.boundingRect(pravougaonik)

                if ( cv2.contourArea(pravougaonik) > 53):

                    tackaGore = [x,y]
                    tackaDesno = [x+w,y]
                    tackaLevo = [x,y+h]
                    tackaDole = [x+w,y+h]
                    offset = 15

                    oooo = 5

                    A = [sabiranje.x1-offset -5 ,sabiranje.y1-offset+5]
                    B = [sabiranje.x2-offset + 5 ,sabiranje.y2-offset - 5]

                    C = [sabiranje.x1-oooo -5,sabiranje.y1-oooo + 5]
                    D = [sabiranje.x2-oooo + 5 ,sabiranje.y2-oooo - 5 ]

                    Ao = [oduzimanje.x1+offset,oduzimanje.y1+offset]
                    Bo = [oduzimanje.x2+offset ,oduzimanje.y2+offset ]

                    Co = [oduzimanje.x1,oduzimanje.y1]
                    Do = [oduzimanje.x2 ,oduzimanje.y2]

                    d = 3

                    if  pnpoly([A[0],B[0],D[0],C[0]],[A[1],B[1],D[1],C[1]],tackaDole[0],tackaDole[1]):

                        izvucenBroj = rez[y - d:y + d + h, x - d: x + d + w]
                        sredjenBroj = cv2.erode(cv2.resize(izvucenBroj,(28,28)), (3,3))
                        noviElement = sredjenBroj[np.newaxis,:]
                        predvidjen = loaded_model.predict(np.array([noviElement]))

                        prepoznatiS.append(Prepoznat(framebr,np.argmax(predvidjen),x,y))
                        if nijeObradjen(prepoznatiS):
                            suma += np.argmax(predvidjen)

                    if pnpoly([Ao[0],Bo[0],Do[0],Co[0]],[Ao[1],Bo[1],Do[1],Co[1]],tackaGore[0],tackaGore[1]):

                        izvucenBroj = rez[y - d:y + d + h, x - d: x + d + w]

                        sredjenBroj = cv2.erode(cv2.resize(izvucenBroj,(28,28)), (3, 3))
                        noviElement = sredjenBroj[np.newaxis,:]
                        predvidjen = loaded_model.predict(np.array([noviElement]))

                        prepoznatiO.append(Prepoznat(framebr,np.argmax(predvidjen),x,y))
                        if nijeObradjen(prepoznatiO):
                            suma -= np.argmax(predvidjen)

        Svi.append(suma)
        cap.release()
        cv2.destroyAllWindows()
    end = timer()
    print("Vreme",end - start)

    f = open("out.txt","w")
    f.write("SW 40/2014 Miroslav Kospic\n")
    f.write("file   sum\n")
    for i in range(len(Svi)):
        f.write("video-" + str(i)+".avi" + "	" + str(Svi[i]) + "\n")
    exit(0)

main()

