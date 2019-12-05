import face_recognition
import cv2
import os


def draw_rectangle(img, rect):  # desenha um retangulo na imagem e nas coordenadas receebidas
    (x, y, w, h) = rect
    cv2.rectangle(img, (y, x), (h, w), (0, 180, 0), 2)



resolucaoPadrao = (640, 480)
endereco=r"D:\Eric\Documentos\Unesc\Iniciação Cientifica - Reconhecimento Facial\Piloto\Fotos de teste"
imagem=r"\4pessoas.jpg"
i=0
dirs = os.listdir(endereco)
#for images in dirs:
#    encoding = face_recognition.face_encodings(image)[i]
#    i=i+1

image = face_recognition.load_image_file(endereco+imagem)
#encoding = face_recognition.face_encodings(image)[i]

#results = face_recognition.compare_faces([obama_encoding])


face_locations = face_recognition.face_locations(image)

print(face_locations)
for location in face_locations:
    draw_rectangle(image,location)
image = cv2.resize(image, resolucaoPadrao)
cv2.imshow("camera", image)
cv2.waitKey(0)