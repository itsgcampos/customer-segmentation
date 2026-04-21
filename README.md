# Customer Segmentation with K-Means

## 📌 Objetivo

Segmentar clientes com base em seu comportamento de compra utilizando técnicas de clustering, permitindo identificar perfis distintos e apoiar estratégias de negócio como marketing, retenção e gestão de risco.

---

## 🧠 Problema de Negócio

Empresas precisam entender melhor seus clientes para:

- Identificar clientes de alto valor
- Detectar padrões de comportamento de compra
- Criar estratégias personalizadas (marketing, crédito, retenção)
- Reduzir risco associado a determinados perfis

Este projeto responde:

- Quais são os diferentes perfis de clientes?
- Como os clientes se comportam em termos de frequência, valor e volume de compras?
- Como esses perfis podem ser utilizados para decisões estratégicas?

---

## 📊 Dataset

O conjunto de dados contém transações de compras realizadas por clientes em uma loja online.

### 📌 Descrição das Variáveis

- **InvoiceNo**
  Número da fatura (identificador da transação)

- **StockCode**
  Código único do produto

- **Description**
  Descrição do item comprado

- **Quantity**
  Quantidade de itens comprados

- **InvoiceDate**
  Data da compra

- **UnitPrice**
  Preço unitário do item

- **CustomerID**
  Identificador único do cliente

- **Country**
  País de residência do cliente

---

## ⚙️ Metodologia

### 1. Limpeza de Dados

- Remoção de valores nulos
- Tratamento de registros inválidos (quantidades negativas, preços inconsistentes)

---

### 2. Feature Engineering

Foi aplicada a técnica de **RFM (Recency, Frequency, Monetary)**:

- **Recency (R)**: Tempo desde a última compra
- **Frequency (F)**: Número de compras realizadas
- **Monetary (M)**: Valor total gasto pelo cliente

Essas variáveis permitem capturar o comportamento de consumo de cada cliente.

---

### 3. Preparação dos Dados

- Normalização das variáveis (StandardScaler)
- Remoção de outliers (opcional, mas recomendado)

---

### 4. Modelagem

- Algoritmo utilizado: **K-Means**
- Definição do número de clusters utilizando:
  - Elbow Method
  - (Opcional) Silhouette Score

---

### 5. Segmentação de Clientes

Após o clustering, os grupos foram interpretados com base nas variáveis RFM.

Exemplo de segmentos que espera-se encontrar:

- 🟢 Clientes de alto valor (alta frequência e alto gasto)
- 🟡 Clientes regulares
- 🔴 Clientes de baixo valor ou inativos
- 🔵 Novos clientes

---

## 💡 Insights de Negócio

- Clientes de alto valor podem ser priorizados com benefícios exclusivos
- Clientes inativos podem ser alvo de campanhas de reativação
- Clientes de baixo valor podem representar maior risco em estratégias de crédito
- Segmentação permite personalizar ofertas e limites de crédito

---

## 🚀 Aplicações

- Marketing direcionado
- Estratégias de retenção
- Definição de limites de crédito por perfil
- Identificação de clientes de alto risco

---

## 🛠️ Tecnologias Utilizadas

- Python
- Pandas
- Scikit-learn
- Matplotlib / Seaborn

## 🚀 Pipeline do modelo em produção

1. Executar main.py, responsável por ler a base de dados crua, fazer o pré-processamento e gerar a base tratada que treinará o modelo;
2. O arquivo src/models/predict.py é responsável por realizar as predições em produção;
3. Para testes isolados, executar o arquivo src/models/test_predict.py. O csv dos clientes para realizar a predição, deve ser salvo como o path: data/raw/clients_to_predict.csv. Os resultados estarão presentes em data/predictions/clients_to_predict.csv

---

## 🔮 Próximos Passos

- Conseguir coletar mais dados ou dados de maior qualidade
- Realizei testes rápidos com outros modelos, e o resultado também não foram satisfatórios, o K-means teve até melhor desempenho
- Uma outra alternativa seria trabalhar mais em cima dos dados já existente, explorando novas features e maior tratamento/pré-processamento dos dados

---
