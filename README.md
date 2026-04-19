# 🧠 Customer Segmentation com K-Means

Este projeto tem como objetivo realizar a **segmentação de clientes** utilizando técnicas de **clustering (K-Means)**, a partir de dados transacionais de compras.

A ideia é identificar **grupos de clientes com comportamentos semelhantes**, permitindo aplicações em:

- Marketing direcionado
- Personalização de ofertas
- Estratégias de retenção
- Análise de valor do cliente

---

# 📊 Dataset

O dataset utilizado contém transações de clientes de um e-commerce, com informações sobre compras realizadas.

## 📌 Descrição das Variáveis

| Coluna      | Descrição                                     |
| ----------- | --------------------------------------------- |
| InvoiceNo   | Número da fatura (identificador da transação) |
| StockCode   | Código único do produto                       |
| Description | Descrição do item comprado                    |
| Quantity    | Quantidade de itens comprados                 |
| InvoiceDate | Data da compra                                |
| UnitPrice   | Preço unitário do item                        |
| CustomerID  | Identificador único do cliente                |
| Country     | País de residência do cliente                 |

---

# 🎯 Objetivo do Projeto

O principal objetivo é:

> Agrupar clientes com base em seu comportamento de compra utilizando K-Means.

Para isso, serão criadas features que representem o comportamento do cliente, como:

- Frequência de compras
- Valor total gasto
- Recência da última compra

---

# 🧠 Metodologia

O projeto segue um pipeline estruturado:

## 1. Análise Exploratória (EDA)

- Entendimento dos dados
- Identificação de outliers e valores faltantes
- Análise de distribuição das variáveis

## 2. Pré-processamento

- Limpeza dos dados
- Tratamento de valores inconsistentes
- Conversão de tipos

## 3. Feature Engineering

Criação de variáveis agregadas por cliente, como:

- Total gasto
- Número de compras
- Ticket médio
- Recência

## 4. Modelagem (K-Means)

- Escolha do número ideal de clusters (Elbow Method)
- Avaliação com métricas como:
  - Silhouette Score
  - Inertia

## 5. Avaliação e Interpretação

- Análise dos clusters gerados
- Perfil de cada grupo de clientes
- Insights de negócio
