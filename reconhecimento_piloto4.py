import os
import time
import cv2
import face_recognition
import numpy as np

#costantes
enderecopadrao = r"D:\Eric\Documentos\Unesc\Iniciação Cientifica - Reconhecimento Facial\Piloto"
# definição de variaveis globais
resolucaoPadrao = (640, 360)  # para a webcam Razer
#resolucaoPadrao = (640, 240)  # para a webcam embutida
cameraEscolida = 1
espera = 1

seletividade = 0.48
scale = 1.3
numeroarquivos = 8


def define_recorte_figura(face_locations):
    # esta função recebe um vetor con as coordenadas de um retangulo onde pode haver um rosto
    # e retorna a coordenada do maior rosto para ser recortado durante o cadastramento ja considerando o fator scale

    #print(face_locations)
    maior_area = 0
    indice = 0
    indice= escolhe_maior_retangulo(face_locations)
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
    # função que recebe o endereço da imagem onde as imagens devem ser gravadas captura a imagem,
    # captura a imagem da webcam selecionada e recorta somente o rosto que ira ser gravado
    # para futuro treinamento
    pastaCriada = False
    video_capture = cv2.VideoCapture(cameraEscolida)  # seleciona a webcam escolhida
    ret, capture_picture = video_capture.read()# captura imagem
    while (pastaCriada == False): # se a pasta/usuario não exisitir
        nome = input("Digite o nome do usuário:")
        nome = nome.replace(" ", "_")
        nome = nome.replace(";", "_")
        endereco = endereco + "\\" + str(nome)
        try:
            os.mkdir(endereco)# cria pasta
            pastaCriada = True
        except: # caso de erro
            print("Pasta ja existe, favor verificar e tentar novamente")
            pastaCriada = False
    for numero in range(numeroarquivos): # tira o numero de fotos configurado por numeroarquivos
        k = 0
        while ((k != 27) and (k != 32)): # quando for pressionado esq ou espaço
            ret, frame = video_capture.read()
            capture_picture = frame
            a, b = resolucaoPadrao
            a = int(a * 1.5)
            b = int(b * 1.5)
            #pega a fota com resolução 50% maior que o rosto para garantir um boa imagem para o treinamento, pode ser reduzido para melhorar desempenho(velocidade) e
            # economizar espaço de armazenamento em discos
            resolucao = (a, b)
            small_frame = cv2.resize(frame, resolucao)
            face_locations = face_recognition.face_locations(small_frame) # localiza o rosto na fotografia
            if (face_locations != []):
                top, bottom, left, rigth = define_recorte_figura(face_locations) # define limites de recorte da foto
                #print(top, bottom, left, rigth)
                #cv2.rectangle(small_frame, (left, top), (rigth, bottom), (0, 128, 255), 2)
                capture_picture = small_frame[top:bottom, left:rigth] # captura imagem
            cv2.imshow("teste", small_frame)
            k = cv2.waitKey(1)
        nomearquivo = endereco + "\\" + str(numero) + ".jpg"
        print("Gravando arquivo", nomearquivo)
        os.chdir(endereco)
        cv2.imwrite(str(numero) + ".jpg", capture_picture)  # escreve o arquivo no disco
        os.chdir(enderecopadrao)
    cv2.destroyAllWindows()
    video_capture.release()
    time.sleep(espera)
    return True


def grava_aprendizado(names, encoding, arquivo="treinamento.dat"):
    # função recebe a matriz de nomes, a codificação da Ai e o nome do arquivo de treinamento e
    # faz a gravação  do apredizado feito num arquivo texto do tipo CSV para poder ser recuperado posteriomente
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
    # recebe o arquivo a ser lido do tipo CSV e retorna o vetor de nomes e a matriz de treinamente realizado
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

def carrega_imagens_e_aprende(endereco): # função de treinamento
    # recebe o endereço onde as fotos estão gravadas e realiza o treinamento
    # e retorna o vetor de nomes e a matriz de do treinamento realizado
    known_face_names = []
    known_face_encodings = []
    total = 0
    processados = 0
    dirs = os.listdir(endereco) # lista todas as pastas
    for dir in dirs:
        total += len(os.listdir(endereco + "\\" + dir)) # conta o numero de fotos por pasta

    #print(dirs)
    hora_inicio = time.time() # guarda hora inicio treinamento

    for dir in (dirs):

        imagedir = os.listdir(endereco + "\\" + dir)
        # print(imagedir)
        for images in imagedir:
            face = face_recognition.load_image_file(endereco + '\\' + dir + "\\" + images)  # carrega imagem da memoria
            try:
                face_find = face_recognition.face_encodings(face)[0]  # verifica se ha um rosto na imagem
                known_face_encodings.append(face_find)  # copia a imagem encontrada no vetor
                known_face_names.append(dir)  # coloca o nome no vetor
                # print("Face encontrada em %s\\%s." % (dir,images))
            except:  # caso não encontre face ou tenha erro na imagem não carrega nadas e imprime mensagem de erro
                print("Face não encontrada em %s\\%s.\nImagem Ignorada!" % (dir, images))
            hora_atual = time.time() # armazena hora atual
            processados += 1
            if (processados > 4):
                tempoRestante = ((hora_atual - hora_inicio) / processados) * (total - processados)# calcula tempo restante e imprime tela
                print("Status %.1f%% \nTempo estimado para Termino é de %s" % (
                    processados / total * 100, time.strftime("%H:%M:%S", time.gmtime(tempoRestante))))
            elif (processados <= 4):
                tempoRestante = ((hora_atual - hora_inicio) / processados) * (total - processados)
                print("Status %.1f%% \nTempo estimado para Termino é de %s" % (
                    processados / total * 200, time.strftime("%H:%M:%S", time.gmtime(tempoRestante))))

    return known_face_names, known_face_encodings # retorna vetor de nomes conhecidos e valores do trainamento


def calcula_confianca(face_distances, matches, known_face_names):
    # função que transforma a distancia euclidiana entre as imagens comparadas e retorna um fator em 0 e 100%
    # corresopondente ao valor percentual de confiança
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
    return status, (1 - erro) * 100


def escolhe_maior_retangulo(retangulos):
    # função recebe um lista de retangulos e retorna o indice do maior
    # pois em tese trata-se do rosto mais proximo da camera
    if retangulos==[]:
        return None
    max_area = 0
    indice = 0
    i = 0
    for x1, y1, x2, y2 in retangulos:
        area = (((x2 - x1)**2)**0.5) * (((y2 - y1)**2)**0.5)
        if (area > max_area):
            max_area = area
            indice = i
        i += 1
    return indice



def reconhece_imagem(imagem, max_erro, known_face_names, known_face_encodings):
    # função que recebe a imagem capturada da webcam, o erro maximo admissivel, o vetor de nomes e a matriz de aprendizado
    # retorna  imagem com um retangulo sobre o rosto com o nome do mesmo caso seja possivel identificar e o status do reconhecimento
    small_frame = cv2.resize(imagem, resolucaoPadrao)
    # rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(small_frame)
    maior_retangulo = escolhe_maior_retangulo(face_locations)
    face=[]



    if (maior_retangulo != None):
        face.append(face_locations[maior_retangulo])
        face_encodings = face_recognition.face_encodings(small_frame, face)
        face_names = []

        # Verifica se a a maior face encontrada é parecida com alguma conhecida
        matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0], tolerance=max_erro)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
        # calcula o erro (distancia) da imagem testada com cada uma das imagens treinadas
        best_match_index = np.argmin(face_distances)  # Busca a imagem de menor distancia
        confidence = 1

        status, confidence = calcula_confianca(face_distances, matches, known_face_names) # calcula o erro e retorna

        if status:
            name = known_face_names[best_match_index] # seleciona o nome da melhor identificação
        print("{3:-<20}{0:.3f}-{1:d}{2:d} confiança={4:.2f}".format(face_distances[best_match_index],
                                                                    matches[best_match_index],
                                                                    best_match_index, name, confidence))

        face_names.append(name) # coloca o nome num vetor
        # exibndo resultados
        (top, right, bottom, left), name =(face[0], face_names[0])

        if (confidence > 0):
            cv2.rectangle(small_frame, (right, top), (left, bottom), (0, 255, 255), 1) # desenha um retangulo no rosto


        cv2.rectangle(small_frame, (left, bottom - 20), (right, bottom), (0, 128, 255), cv2.FILLED) # desenha um retangulo laranja preenchido abaixo do retangulo de identificação
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(small_frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1) # escreve o nome em branco no retangulo laranja

    return small_frame


def menu():
    # função para mostrar o menu
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
    # função principal
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
                #imagem=cv2.resize(imagem, resolucaoPadrao)
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


# chama função principal main()
main()

