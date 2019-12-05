import face_recognition
import cv2
import numpy as np
import os
import time

# definição de variaveis globais
resolucaoPadrao = (640, 360)
video_capture = cv2.VideoCapture(0)  # seleciona a webcam escolhida
seletividade = 0.48


def inclui_registro():
    nome = input("Digite o nome do usuário:")
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, resolucaoPadrao)
    face_locations = face_recognition.face_locations(small_frame)
    print(face_locations)
    # capturePicture = small_frame[int(y - (factor - 1) * h):int(y + h * factor),int(x - w * (factor - 1)):int(x + w * factor)]


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
    # print(names)
    encoding = []
    for string in arquivo:
        lines = string[:-1].split(";")
        data = np.array([])
        for i in range(len(lines)):
            data = np.append(data, float(lines[i]))
        encoding.append(data)
    # print(encoding)
    return names, encoding


def carrega_imagens_e_aprende(endereco):
    known_face_names = []
    known_face_encodings = []
    total = 0
    processados = 0
    dirs = os.listdir(endereco)
    for dir in dirs:
        total += len(os.listdir(endereco + "\\" + dir))

    print(dirs)
    hora_inicio = time.time()

    for dir in (dirs):
        imagedir = os.listdir(endereco + "\\" + dir)
        # print(imagedir)
        for images in imagedir:
            hora_atual = time.time()
            if (processados > 0):
                tempoRestante = ((hora_atual - hora_inicio) / processados) * (total - processados)
                print("Status %.1f%% \nTempo estimado para Termino é de %s" % (
                processados / total * 100, time.strftime("%H:%M:%S", time.gmtime(tempoRestante))))
            processados += 1
            face = face_recognition.load_image_file(endereco + '\\' + dir + "\\" + images)  # load the image im memory
            try:
                face_find = face_recognition.face_encodings(face)[0]  # verifica se ha um rosto na imagem
                known_face_encodings.append(face_find)  # copia a imagem encontrada no vetor
                known_face_names.append(dir)  # coloca o nome no vetor
                # print("Face encontrada em %s\\%s." % (dir,images))
            except:  # caso não encontre face ou tenha erro na imagem não carrega nadas e imprime mensagem de erro
                print("Face não encontrada em %s\\%s.\nImagem Ignorada!" % (dir, images))
    return known_face_names, known_face_encodings


def reconhece_imagem(imagem, max_erro, known_face_names, known_face_encodings):
    small_frame = cv2.resize(imagem, resolucaoPadrao)
    # rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=max_erro)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings,
                                                        face_encoding)  # calcula o erro (distancia) da imagem testada com cada uma das imagens treinadas
        best_match_index = np.argmin(face_distances)  # Busca a imagem de menor distancia
        for indice in range(len(matches)):
            pass
            # print("{0:.3f}-{1:d}  - ".format(face_distances[indice], matches[indice]), end="")

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        print(
            "{3:-<20}{0:.3f}-{1:d} {2:d}".format(face_distances[best_match_index], matches[best_match_index],
                                                 best_match_index, name))

        face_names.append(name)
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(small_frame, (right, top), (left, bottom), (0, 0, 255), 1)

        # Draw a label with a name below the face
        cv2.rectangle(small_frame, (left, bottom - 20), (right, bottom), (0, 128, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(small_frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        # rgb_frame = small_frame[:, :, ::-1]
        # cv2.imshow("camera", rgb_frame)
    return small_frame


def menu():
    print("------------------------")
    print("----------MENU----------")
    print("1-Carregar Modelo")
    print("2-Treinar modelo")
    print("3-Inserir usuário")
    print("4-Reconhecer imagem")
    print("5-Sair")
    print("------------------------")
    opcao = int(input("opção:"))
    return opcao


def main():
    known_face_names=[]
    known_face_encodings=[]
    opcao = 0
    modelo_carregado = False
    while (opcao != 5):
        opcao = menu()
        if (opcao == 1):
            known_face_names, known_face_encodings = le_aprendizado()
            modelo_carregado = True
        elif (opcao == 2):
            endereco = r"D:\Eric\Documentos\Unesc\Iniciação Cientifica - Reconhecimento Facial\Piloto\Trainning"
            # endereco=r"D:\Eric\Documentos\Unesc\Iniciação Cientifica - Reconhecimento Facial\Piloto\Trainning"
            known_face_names, known_face_encodings = carrega_imagens_e_aprende(endereco)
            grava_aprendizado(known_face_names, known_face_encodings)
            modelo_carregado = True
        elif (opcao == 3):
            print("função não implementada")
        elif (opcao == 4):
                k = 0
                while (modelo_carregado) and (k != 27) and (k != 32):
                    ret, imagem = video_capture.read()
                    imagem = reconhece_imagem(imagem, seletividade, known_face_names, known_face_encodings)
                    cv2.imshow("camera", imagem)
                    k = cv2.waitKey(1)
                cv2.destroyAllWindows()

# chama main()
main()
video_capture.release()
