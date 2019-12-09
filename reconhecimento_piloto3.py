import os
import time

import cv2
import face_recognition
import numpy as np

enderecopadrao = r"D:\Eric\Documentos\Unesc\Iniciação Cientifica - Reconhecimento Facial\Piloto"
# definição de variaveis globais
resolucaoPadrao = (640, 360)  # para a webcam Razer
# resolucaoPadrao=(640,480) # para a webcam embutida
cameraEscolida = 0
espera = 1

seletividade = 0.48
scale = 1.3
numeroarquivos = 8


def define_recorte_figura(face_locations):
    print(face_locations)
    maior_area = 0
    indice = 0
    i = 0
    for retangles in face_locations:
        top, right, bottom, left = retangles
        area = ((bottom - top) * (right - left))
        if (area > maior_area):
            maior_area = area
            indice = i
        i += 1
    top, right, bottom, left = face_locations[indice]
    h = int((bottom - top) * scale / 2)
    l = int((right - left) * scale / 2)
    mh = int((bottom + top) / 2)
    ml = int((left + right) / 2)
    top = mh - h
    right = ml + l
    bottom = mh + h
    left = ml - l
    lmax, hmax = resolucaoPadrao
    if (top < 0):
        top = 0
    if left < 0:
        left = 0
    if right > lmax:
        right = lmax
    if bottom > hmax:
        bottom = hmax
    return top, bottom, left, right


def inclui_registro(endereco):
    pastaCriada = False
    video_capture = cv2.VideoCapture(cameraEscolida)  # seleciona a webcam escolhida
    ret, capture_picture = video_capture.read()
    while (pastaCriada == False):
        nome = input("Digite o nome do usuário:")
        nome = nome.replace(" ", "_")
        nome = nome.replace(";", "_")
        endereco = endereco + "\\" + str(nome)
        try:
            os.mkdir(endereco)
            pastaCriada = True
        except:
            print("Pasta ja existe, favor verificar e tentar novamente")
            pastaCriada = False
    for numero in range(numeroarquivos):
        k = 0
        while ((k != 27) and (k != 32)):
            ret, frame = video_capture.read()
            capture_picture = frame
            a, b = resolucaoPadrao
            a = int(a * 1.5)
            b = int(b * 1.5)
            resolucao = (a, b)
            small_frame = cv2.resize(frame, resolucao)
            face_locations = face_recognition.face_locations(small_frame)
            if (face_locations != []):
                top, bottom, left, rigth = define_recorte_figura(face_locations)
                print(top, bottom, left, rigth)
                cv2.rectangle(small_frame, (left, top), (rigth, bottom), (0, 128, 255), 2)
                capture_picture = small_frame[top:bottom, left:rigth]
            cv2.imshow("teste", small_frame)
            k = cv2.waitKey(1)
        nomearquivo = endereco + "\\" + str(numero) + ".jpg"
        print(nomearquivo)
        os.chdir(endereco)
        cv2.imwrite(str(numero) + ".jpg", capture_picture)  # escreve o arquivo no disco
        os.chdir(enderecopadrao)
    cv2.destroyAllWindows()
    video_capture.release()
    time.sleep(espera)
    return True


def grava_aprendizado(names, encoding, arquivo="treinamento.dat"):
    arquivo = open(arquivo, 'w', encoding="utf8")
    string = ""
    for name in names:
        string += str(name) + ";"
    arquivo.writelines(string[:-1] + "\n")

    for info in encoding:
        string = ""
        for data in info:
            string += str(data) + ";"
        string += "\n"
        arquivo.writelines(string[:-2] + "\n")
    arquivo.close()


def le_aprendizado(arquivo="treinamento.dat"):
    arquivo = open(arquivo, 'r', encoding="utf8")
    string = arquivo.readline()
    names = string[:-1].split(";")
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
            face = face_recognition.load_image_file(endereco + '\\' + dir + "\\" + images)  # load the image im memory
            try:
                face_find = face_recognition.face_encodings(face)[0]  # verifica se ha um rosto na imagem
                known_face_encodings.append(face_find)  # copia a imagem encontrada no vetor
                known_face_names.append(dir)  # coloca o nome no vetor
                # print("Face encontrada em %s\\%s." % (dir,images))
            except:  # caso não encontre face ou tenha erro na imagem não carrega nadas e imprime mensagem de erro
                print("Face não encontrada em %s\\%s.\nImagem Ignorada!" % (dir, images))
            hora_atual = time.time()
            processados += 1
            if (processados > 2):
                tempoRestante = ((hora_atual - hora_inicio) / processados) * (total - processados)
                print("Status %.1f%% \nTempo estimado para Termino é de %s" % (
                    processados / total * 100, time.strftime("%H:%M:%S", time.gmtime(tempoRestante))))
            elif (processados == 1):
                tempoRestante = ((hora_atual - hora_inicio) / processados) * (total - processados)
                print("Status %.1f%% \nTempo estimado para Termino é de %s" % (
                    processados / total * 200, time.strftime("%H:%M:%S", time.gmtime(tempoRestante))))

    return known_face_names, known_face_encodings


def calcula_confianca(face_distances, matches, known_face_names):
    repeticoes = 0
    encontrados = []
    erro = 1
    for i in range(len(matches)):
        if (matches[i]):
            erro *= face_distances[i]
            repeticoes += 1
            if not (known_face_names[i] in encontrados):
                encontrados.append(known_face_names[i])
    if (len(encontrados) == 1):
        status = True
    else:
        status = False
        erro = 1
    print(repeticoes)
    return status, (1 - erro) * 100


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
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        # calcula o erro (distancia) da imagem testada com cada uma das imagens treinadas
        best_match_index = np.argmin(face_distances)  # Busca a imagem de menor distancia
        confidence = 1

        status, confidence = calcula_confianca(face_distances, matches, known_face_names)

        if status:
            name = known_face_names[best_match_index]
        print("{3:-<20}{0:.3f}-{1:d}{2:d} confiança={4:.2f}".format(face_distances[best_match_index],
                                                                    matches[best_match_index],
                                                                    best_match_index, name, confidence))

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
    os.chdir(enderecopadrao)
    opcao = 0
    passou = False
    while not (0 < opcao < 6):
        if passou:
            print("------------------------")
            print("Digite uma opção válida")
            print("------------------------\n")

        passou = True
        print("------------------------")
        print("----------MENU----------")
        print("1-Carregar Modelo")
        print("2-Treinar modelo")
        print("3-Inserir usuário")
        print("4-Reconhecer imagem")
        print("5-Sair")
        print("------------------------")
        opcao = (input("opção:"))
        if opcao.isdigit():
            opcao = int(opcao)
        else:
            opcao = 0
    return opcao


def main():
    known_face_names = []
    known_face_encodings = []
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
            video_capture = cv2.VideoCapture(cameraEscolida)  # seleciona a webcam escolhida
            inclui_registro(
                endereco=r"D:\Eric\Documentos\Unesc\Iniciação Cientifica - Reconhecimento Facial\Piloto\Trainning")
        elif (opcao == 4):
            video_capture = cv2.VideoCapture(cameraEscolida)  # seleciona a webcam escolhida
            k = 0
            while (modelo_carregado) and (k != 27) and (k != 32):
                ret, imagem = video_capture.read()
                imagem = reconhece_imagem(imagem, seletividade, known_face_names, known_face_encodings)
                cv2.imshow("camera", imagem)
                k = cv2.waitKey(1)
            cv2.destroyAllWindows()
            try:
                video_capture.release()
                time.sleep(espera)
                print()
            except:
                print("Erro Geral!")
            print("Voltando ao menu....")


# chama main()
main()
