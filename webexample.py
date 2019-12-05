import face_recognition
import cv2
import numpy as np
import os

# definição de variaveis globais
resolucaoPadrao = (640, 480)


# video_capture = cv2.VideoCapture(0)
def grava_aprendizado(names, encoding, arquivo="treinamento.dat"):
    arquivo = open(arquivo, 'w', encoding="utf8")
    string = ""
    for name in names:
        string += str(name) + ";"
    arquivo.writelines(string[:-1] + "\n")

    for info in encoding:
        string = ""
        for data in info:
            string += str(data) + ";";
        string += "\n"
        arquivo.writelines(string[:-2] + "\n")
    arquivo.close()


def le_aprendizado(arquivo="treinamento.dat"):
    arquivo = open(arquivo, 'r', encoding="utf8")
    string = arquivo.readline()
    names = []
    names = string[:-1].split(";")
    #print(names)
    encoding = []
    for string in arquivo:
        lines = string[:-1].split(";")
        data = np.array([])
        for i in range(len(lines)):
            data=np.append(data,float(lines[i]))
        encoding.append(data)
    #print(encoding)
    return names,encoding


def carrega_imagens_e_aprende(endereco):
    known_face_names = []
    known_face_encodings = []

    dirs = os.listdir(endereco)
    for i in range(len(dirs)):
        face = face_recognition.load_image_file(endereco + '\\' + dirs[i])  # load the image im memory
        try:
            face_find = face_recognition.face_encodings(face)[0]  # verifica se ha um rosto na imagem
            known_face_encodings.append(face_find)  # copia a imagem encontrada no vetor
            known_face_names.append(dirs[i][:-4])  # coloca o nome no vetor
            print(dirs[i][:-4])  # imprime o nome
        except:  # caso não encontre face ou tenha erro na imagem não carrega nadas e imprime mensagem de erro
            print("Face não encontrada em %s.\nImagem Ignorada" % dirs[i])  #
    return known_face_names, known_face_encodings

def reconhece_imagem(imagem,max_erro):
    small_frame = cv2.resize(imagem, resolucaoPadrao)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=max_erro)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings,
                                                        face_encoding)  # calcula o erro (distancia) da imagem testada com cada uma das imagens treinadas
        best_match_index = np.argmin(face_distances)  # Busca a imagem de menor distancia
        for indice in range(len(matches)):
            print("{0:.3f}-{1:d}  - ".format(face_distances[indice], matches[indice]), end="")

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        print(
            "{0:.3f}-{1:d} {2:d}".format(face_distances[best_match_index], matches[best_match_index], best_match_index))

        face_names.append(name)
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(small_frame, (right, top), (left, bottom), (0, 0, 255), 1)

        # Draw a label with a name below the face
        cv2.rectangle(small_frame, (left, bottom - 20), (right, bottom), (0, 128, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(small_frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        rgb_frame = small_frame[:, :, ::-1]
        cv2.imshow(dirs[i], rgb_frame)


# Initialize some variables
endereco = r"D:\Eric\Documentos\Unesc\Iniciação Cientifica - Reconhecimento Facial\Piloto\FotosTreinamento"
#known_face_names, known_face_encodings = carrega_imagens_e_aprende(endereco)
#grava_aprendizado(known_face_names, known_face_encodings)
known_face_names, known_face_encodings=le_aprendizado()
#print(type(known_face_encodings), type(known_face_encodings[0]))
#print(known_face_encodings)
endereco = r"D:\Eric\Documentos\Unesc\Iniciação Cientifica - Reconhecimento Facial\Piloto\FotosTeste"
dirs = os.listdir(endereco)
for i in range(len(dirs)):
    imagem = face_recognition.load_image_file(endereco + "\\" + dirs[i])
    reconhece_imagem(imagem,0.53)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)


    # Find all the faces and face encodings in the current frame of video



    print("-------------------------------------------")


cv2.waitKey(0)
