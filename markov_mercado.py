# Previsão de comportamento de mercado com cadeias de Markov.
#
# Simula a migração de clientes entre estados (exemplo implementado: provedores de nuvem),
# calcula a evolução após k passos, o estado estacionário e gera um gráfico.
# Teoria e a aplicação estão explicadas no README.md.

import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # salva o gráfico em arquivo, sem abrir janela

# ---> Funções básicas de álgebra linear

# Retorna a matriz identidade n x n


def identidade(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


# Retorna a transposta de A (linhas viram colunas).
def transposta(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


# Calcula o vetor linha x * P: y[j] = soma_i x[i] * P[i][j]
def mult_vetor_matriz(x, P):
    n = len(P)
    y = [0.0] * n
    for j in range(n):
        for i in range(n):
            y[j] += x[i] * P[i][j]
    return y


# Produto de duas matrizes quadradas: C[i][j] = soma_k A[i][k]*B[k][j]
def mult_matrizes(A, B):
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


# Determinante por eliminação de Gauss com pivoteamento parcial
# Cada troca de linhas inverte o sinal; o resultado é o produto da diagonal
def determinante(A):
    n = len(A)
    M = [linha[:] for linha in A]          # cópia para não alterar a original
    det = 1.0
    for c in range(n):
        # escolhe como pivô o maior valor absoluto da coluna (estabilidade)
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        
        if abs(M[p][c]) < 1e-12:
            return 0.0                      # coluna nula -> det = 0
        if p != c:
            M[c], M[p] = M[p], M[c]
            det = -det                      # troca de linhas inverte o sinal
            
        det *= M[c][c]
        for r in range(c + 1, n):           # zera abaixo do pivô
            f = M[r][c] / M[c][c]
            for j in range(c, n):
                M[r][j] -= f * M[c][j]

    return det


# Resolve A x = b por eliminação de Gauss com pivoteamento parcial.
# Retorna None se o sistema for singular
def resolver_sistema(A, b):
    n = len(A)

    # matriz aumentada [A | b]
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))

        if abs(M[p][c]) < 1e-12:
            return None
        
        M[c], M[p] = M[p], M[c]

        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for j in range(c, n + 1):
                M[r][j] -= f * M[c][j]

    # substituição regressiva
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - s) / M[i][i]
    return x


# --> Validação

# Verifica se P é quadrada e estocástica e se x0 é vetor de probabilidade
# Levanta ValueError com mensagem clara em caso de problema
def validar_entrada(P, x0, tol=1e-9):
    n = len(P)

    if any(len(l) != n for l in P):
        raise ValueError("A matriz de transição deve ser quadrada.")
    
    if len(x0) != n:
        raise ValueError("O vetor inicial deve ter tamanho n = %d." % n)
    
    for i, linha in enumerate(P):
        if any(v < 0 for v in linha):
            raise ValueError("Probabilidades negativas na linha %d." % (i + 1))
        if abs(sum(linha) - 1) > tol:
            raise ValueError(
                "A linha %d soma %.6f (deve somar 1)." % (i + 1, sum(linha)))
        
    if any(v < 0 for v in x0) or abs(sum(x0) - 1) > tol:
        raise ValueError("O vetor inicial deve ser não negativo e somar 1.")

# --> Evolução do sistema

# Aplica x_{t+1} = x_t * P por k passos. Retorna a lista [x_0, ..., x_k]
def evoluir(P, x0, k):
    historico = [x0[:]]
    x = x0[:]

    for _ in range(k):
        x = mult_vetor_matriz(x, P)
        historico.append(x)
    
    return historico


# Calcula P^k por multiplicações sucessivas (confere x_k = x_0 P^k)
def potencia_matriz(P, k):
    R = identidade(len(P))

    for _ in range(k):
        R = mult_matrizes(R, P)
    
    return R

# --> Estado estacionário

# Resolve π P = π com sum(π)=1
# Reescrevemos como (Pᵀ - I) π = 0. Como esse sistema é indeterminado (det = 0),
# troca-se a última equação pela normalização sum(π) = 1


def estado_estacionario_linear(P):
    n = len(P)
    Pt = transposta(P)
    A = [[Pt[i][j] - (1.0 if i == j else 0.0)
        for j in range(n)] for i in range(n)]
    b = [0.0] * n
    A[n - 1] = [1.0] * n        # última linha: π1 + ... + πn = 1
    b[n - 1] = 1.0

    return resolver_sistema(A, b)


# Método da convergência (potências): itera até a variação ser < tol
# Retorna (vetor, passos). Se não convergir, vetor é None
def estado_estacionario_convergencia(P, x0, tol=1e-10, max_passos=100000):
    x = x0[:]

    for t in range(1, max_passos + 1):
        novo = mult_vetor_matriz(x, P)
        if max(abs(novo[i] - x[i]) for i in range(len(x))) < tol:
            return novo, t
        x = novo
    
    return None, max_passos


# Testa regularidade: existe m com P^m estritamente positiva?
# (testar até m = (n-1)^2 + 1, limite de Wielandt)
def eh_regular(P):
    n = len(P)
    R = [linha[:] for linha in P]
    
    for _ in range((n - 1) ** 2 + 1):
        if all(R[i][j] > 1e-15 for i in range(n) for j in range(n)):
            return True
        R = mult_matrizes(R, P)
    
    return False


# Estima |λ2| pela razão entre distâncias sucessivas a π:
# ||x_{t+1}-π|| / ||x_t-π|| -> |λ2|
def estimar_lambda2(historico, pi):
    def dist(x):
        return sum(abs(x[i] - pi[i]) for i in range(len(pi)))
    
    for t in range(len(historico) - 2, 0, -1):   # pega último ponto ainda não-nulo
        d0, d1 = dist(historico[t]), dist(historico[t + 1])
        if d0 > 1e-12:
            return d1 / d0
    
    return None


# --> Gráfico

# Verifica o nome do arquivo para gerar nomes únicos
def gerar_nome_unico(arquivo):
    nome, extensao = os.path.splitext(arquivo)
    contador = 1

    while os.path.exists(arquivo):
        arquivo = f"{nome}_{contador}{extensao}"
        contador += 1

    return arquivo


# Plota a probabilidade de cada estado por passo e a linha do estado estacionário
def plotar(historico, nomes, pi, arquivo="grafico-evolucao/evolucao_markov.png"):
    passos = list(range(len(historico)))
    plt.figure(figsize=(9, 5))

    for i, nome in enumerate(nomes):
        serie = [h[i] for h in historico]
        linha, = plt.plot(passos, serie, marker="o", markersize=3, label=nome)

        if pi is not None:      # linha tracejada = limite
            plt.axhline(pi[i], color=linha.get_color(),
                        linestyle="--", alpha=0.5)

    plt.title(
        "Evolução da participação de mercado (tracejado = estado estacionário)")
    plt.xlabel("Passo (trimestre)")
    plt.ylabel("Probabilidade")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Cria a pasta caso ela ainda não exista
    pasta = os.path.dirname(arquivo)
    if pasta:
        os.makedirs(pasta, exist_ok=True)

    arquivo = gerar_nome_unico(arquivo)
    plt.savefig(arquivo, dpi=150)
    plt.close()

    return arquivo


# --> Relatório

# Formata vetor como porcentagens
def fmt(v):
    return "[" + ", ".join("%.2f%%" % (100 * a) for a in v) + "]"


# Imprime o relatório de interpretação do comportamento de longo prazo
def relatorio(P, x0, k, nomes, historico, pi_lin, pi_conv, passos_conv):
    n = len(P)
    I = identidade(n)
    P_menos_I = [[P[i][j] - I[i][j] for j in range(n)] for i in range(n)]
    print("-" * 30)
    print("RELATÓRIO – CADEIA DE MARKOV")
    print("-" * 30)
    print("Estados:", ", ".join(nomes))
    print("\nEstado inicial x0 :", fmt(x0))
    print("Estado após k=%d   :" % k, fmt(historico[-1]))

    # verificação: x_k calculado por P^k deve coincidir com a iteração
    xk = [sum(x0[i] * potencia_matriz(P, k)[i][j]
              for i in range(n)) for j in range(n)]
    erro = max(abs(xk[j] - historico[-1][j]) for j in range(n))
    print("Conferência x0·P^k vs iteração (erro máx): %.2e" % erro)

    print("\ndet(P - I) = %.2e  -> ≈ 0 confirma que λ = 1 é autovalor." %
        determinante(P_menos_I))
    regular = eh_regular(P)
    print("Cadeia regular (alguma potência de P é positiva)?",
          "SIM" if regular else "NÃO")

    if pi_lin is None:
        print("\nO sistema linear é singular: não há estado estacionário único.")
        return
    print("\nEstado estacionário (sistema linear):", fmt(pi_lin))
    if pi_conv is not None:
        print("Estado estacionário (convergência)  :", fmt(pi_conv),
            "(%d passos)" % passos_conv)
        print("Diferença entre os métodos: %.2e" %
            max(abs(pi_lin[i] - pi_conv[i]) for i in range(n)))

    lam2 = estimar_lambda2(evoluir(P, x0, 200), pi_lin)
    # Interpretação
    print("\nINTERPRETAÇÃO")
    print("-" * 70)
    dom = max(range(n), key=lambda i: pi_lin[i])
    print("- No longo prazo, '%s' domina o mercado com %.2f%%." %
          (nomes[dom], 100 * pi_lin[dom]))
    
    for i in range(n):
        delta = pi_lin[i] - x0[i]
        tend = "ganha" if delta > 0 else "perde"
        print("- %-14s %s %.2f p.p. em relação ao início (%.2f%% -> %.2f%%)."
              % (nomes[i], tend, abs(100 * delta), 100 * x0[i], 100 * pi_lin[i]))
    
    if regular:
        print("- Como a cadeia é regular, o resultado NÃO depende do estado inicial.")
    if lam2 is not None:
        print("- |λ2| ≈ %.3f: o erro cai por esse fator a cada passo (quanto menor, mais rápido)." % lam2)
    
    print("- Tempo médio entre visitas ao estado i no equilíbrio: 1/πi passos.")
    print("  " + "; ".join("%s: %.1f" %
          (nomes[i], 1 / pi_lin[i]) for i in range(n) if pi_lin[i] > 0))
    print("- Decisão: para mudar o equilíbrio é preciso alterar P (retenção/captação),")
    print("  e não apenas a participação inicial.")


# ----> Programa principal

# Lê um número inteiro e verifica se está dentro do limite permitido
def ler_inteiro(mensagem, minimo=None):
    while True:
        entrada = input(mensagem).strip()
        if not entrada:
            print("Erro: o campo não pode ficar vazio.")
            continue
        try:
            valor = int(entrada)
        except ValueError:
            print("Erro: digite um número inteiro válido.")
            continue
        if minimo is not None and valor < minimo:
            print("Erro: o valor deve ser maior ou igual a", minimo)
            continue
        
        return valor

# Lê uma quantidade específica de números decimais
def ler_numeros(mensagem, quantidade):
    while True:
        entrada = input(mensagem).strip()
        if not entrada:
            print("Erro: o campo não pode ficar vazio.")
            continue
        try:
            valores = [float(v) for v in entrada.split()]
        except ValueError:
            print("Erro: digite apenas números.")
            continue
        
        if len(valores) != quantidade:
            print("Erro: digite exatamente", quantidade, "números.")
            continue
        if not all(v == v and abs(v) != float("inf") for v in valores):
            print("Erro: os valores devem ser números finitos.")
            continue
        
        return valores

# Lê do teclado nomes, matriz P, vetor x0 e k

def ler_matriz():
    n = ler_inteiro("Número de estados n: ", minimo=1)

    nomes = []

    for i in range(n):
        while True:
            nome = input("Nome do estado %d: " % (i + 1)).strip()

            if nome:
                nomes.append(nome)
                break

            print("Erro: o nome não pode ficar vazio.")

    print("Digite cada linha de P (n números separados por espaço):")
    P = []

    for i in range(n):
        while True:
            linha = ler_numeros("Linha %d: " % (i + 1), n)
            
            if any(v < 0 for v in linha):
                print("Erro: probabilidades não podem ser negativas.")
                continue
            
            if abs(sum(linha) - 1) > 1e-9:
                print("Erro: a linha deve somar 1.")
                continue
            
            P.append(linha)
            break

    while True:
        x0 = ler_numeros("Vetor inicial (n números): ", n)

        if any(v < 0 for v in x0):
            print("Erro: probabilidades não podem ser negativas.")
            continue

        if abs(sum(x0) - 1) > 1e-9:
            print("Erro: o vetor inicial deve somar 1.")
            continue
        break

    k = ler_inteiro("Número de passos k: ", minimo=0)
    
    return nomes, P, x0, k


def main():
    # passo 1: recebe a matriz de transição e o vetor de estado inicial
    # Exemplo: migração trimestral de clientes entre provedores de nuvem
    nomes = ["AWS", "Azure", "Google Cloud"]
    P = [[0.85, 0.10, 0.05],     # de AWS
         [0.08, 0.84, 0.08],     # de Azure
         [0.10, 0.15, 0.75]]     # de Google Cloud
    x0 = [0.50, 0.30, 0.20]
    k = 12

    if input("Usar o exemplo de nuvem? (s/n): ").strip().lower() == "n":
        nomes, P, x0, k = ler_matriz()

    try:
        validar_entrada(P, x0)
    except ValueError as e:
        print("Erro de entrada:", e)
        return

    historico = evoluir(P, x0, k)    # passo 2
    pi_lin = estado_estacionario_linear(
        P)   # passo 3 (sistema linear)
    pi_conv, passos = estado_estacionario_convergencia(
        P, x0)  # passo 3 (convergência)

    print("\nEvolução passo a passo:")
    for t, h in enumerate(historico):
        print("  t=%2d  %s" % (t, fmt(h)))
    print()
    relatorio(P, x0, k, nomes, historico, pi_lin, pi_conv, passos)

    # passo 4: gráfico (usa mais passos para mostrar a convergência)
    arquivo = plotar(evoluir(P, x0, max(k, 40)), nomes, pi_lin)
    print("\nGráfico salvo em:", arquivo)


if __name__ == "__main__":
    main()
