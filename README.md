# Cadeias de Markov para Previsão de Mercado

Trabalho de Vetores, Geometria Analítica e Álgebra Linear aplicado a Sistemas de Informação. O programa simula a migração trimestral de clientes entre provedores de nuvem (AWS, Azure e Google Cloud) usando uma matriz de transição, prevê a participação de mercado após *k* passos e calcula o estado estacionário (o equilíbrio de longo prazo).
Operações de álgebra linear (produto de matrizes, determinante e eliminação de Gauss) implementadas manualmente, e matplotlib usado para gráfico.

## Como executar

```
pip install matplotlib
python markov_mercado.py
```

O programa pergunta se o usuário deseja utilizar o exemplo de provedores de nuvem ou informar sua própria matriz de transição e vetor inicial.

Ao final da execução, é gerado um gráfico com a evolução das probabilidades e o estado estacionário, salvo na pasta `grafico-evolucao`.

Para evitar sobrescrever gráficos anteriores, os arquivos recebem nomes únicos automaticamente (`evolucao_markov.png`, `evolucao_markov_1.png`, `evolucao_markov_2.png` etc.). A pasta é criada automaticamente caso não exista.

## Integrantes
- Beatriz Martins de Lucena
- Jessica Arruda de Jesus
- Lucas Lopes Rodrigues da Silva
