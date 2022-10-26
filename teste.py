import cv2 as cv

cascade = cv.CascadeClassifier("dataset/treinamento/cascade.xml")
while True:
    imagem = cv.imread("dataset/fish/test/fishes/2J1RQ09RX7AU.jpg")
    gray = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
    objetos = cascade.detectMultiScale(gray, 1.25, 5)

    for (x,y,w,h) in objetos:
        cv.rectangle(imagem, (x,y),(x+w,y+h),(0,0,255),2)

    cv.imshow("Fish", imagem)
    k = cv.waitKey(60)
    if k == 27:
        break
    
cv.destroyAllWindows()
