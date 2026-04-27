# Relatório de Drift — Parte 6

Gerado em: 2026-04-27 10:15:38

## Configuração da Simulação

| Parâmetro | Valor |
|-----------|-------|
| Referência histórica | train + val (25,470 registros) |
| Dados correntes simulados | test set (4,495 registros) |
| Nível de significância (α) | 0.05 |
| Teste features contínuas | Kolmogorov-Smirnov (KS) |
| Teste features categóricas | chi-quadrado (χ²) |

## Drift de Dados por Feature

| Feature | Teste | Estatística | p-value | Drift? |
|---------|-------|-------------|---------|--------|
| AGE | KS | 0.0183 | 0.1534 | ✅ NÃO |
| BILL_AMT1 | KS | 0.0121 | 0.6282 | ✅ NÃO |
| BILL_AMT2 | KS | 0.0163 | 0.2599 | ✅ NÃO |
| BILL_AMT3 | KS | 0.0151 | 0.3472 | ✅ NÃO |
| BILL_AMT4 | KS | 0.0158 | 0.2948 | ✅ NÃO |
| BILL_AMT5 | KS | 0.0121 | 0.6254 | ✅ NÃO |
| BILL_AMT6 | KS | 0.0154 | 0.3219 | ✅ NÃO |
| EDUCATION | CHI2 | 10.5584 | 0.1030 | ✅ NÃO |
| LIMIT_BAL | KS | 0.0109 | 0.7524 | ✅ NÃO |
| MARRIAGE | CHI2 | 3.8941 | 0.2731 | ✅ NÃO |
| PAY_0 | CHI2 | 8.2463 | 0.6048 | ✅ NÃO |
| PAY_2 | CHI2 | 6.8161 | 0.7427 | ✅ NÃO |
| PAY_3 | CHI2 | 11.1316 | 0.3474 | ✅ NÃO |
| PAY_4 | CHI2 | 8.4528 | 0.5847 | ✅ NÃO |
| PAY_5 | CHI2 | 3.6053 | 0.9354 | ✅ NÃO |
| PAY_6 | CHI2 | 6.5618 | 0.6826 | ✅ NÃO |
| PAY_AMT1 | KS | 0.0107 | 0.7737 | ✅ NÃO |
| PAY_AMT2 | KS | 0.0099 | 0.8437 | ✅ NÃO |
| PAY_AMT3 | KS | 0.0125 | 0.5839 | ✅ NÃO |
| PAY_AMT4 | KS | 0.0148 | 0.3673 | ✅ NÃO |
| PAY_AMT5 | KS | 0.0124 | 0.5961 | ✅ NÃO |
| PAY_AMT6 | KS | 0.0066 | 0.9962 | ✅ NÃO |
| SEX | CHI2 | 1.1698 | 0.2795 | ✅ NÃO |

## Resumo de Drift de Dados

- **Features com drift detectado:** 0 / 23
- **Features afetadas:** Nenhuma

## Drift de Modelo (Performance)

| Métrica | Baseline (Parte 5) | Corrente (simulado) | Delta | Drift? |
|---------|-------------------|---------------------|-------|--------|
| roc_auc | 0.7682 | 0.7682 | +0.0000 | ✅ NÃO |

> **Nota:** As métricas "correntes" nesta simulação são as mesmas do test set
> da Parte 5 (o modelo foi avaliado exatamente uma vez sobre esses dados).
> Em produção real, as métricas correntes seriam recomputadas periodicamente
> com dados rotulados acumulados.

## Plano de Retreinamento

### Critérios de Gatilho

| Critério | Limiar | Ação |
|----------|--------|------|
| Drift em features críticas (PAY_0, LIMIT_BAL) | p < 0.05 | Avaliar imediatamente |
| Drift em ≥ 5 features simultâneas | qualquer | Iniciar retreinamento |
| Queda de roc_auc | > 5 pp vs baseline | Retreinamento obrigatório |
| Queda de roc_auc | > 3 pp vs baseline | Alerta + monitoramento intensivo |
| Latência de inferência | > 500 ms (P95) | Investigar degradação |

### Frequência Mínima Sugerida

- **Drift de dados:** verificar semanalmente (ou a cada 10k novos registros)
- **Avaliação de modelo:** mensal (requer dados rotulados acumulados)
- **Retreinamento completo:** trimestral ou quando gatilho acionado

### Estimativa de Custo por Retreinamento

| Etapa | Tempo estimado | Custo computacional |
|-------|---------------|---------------------|
| Preparação de dados | ~5 min | Baixo |
| GridSearch (25 combinações) | ~15-30 min | Médio (CPU local) |
| Avaliação no test set | ~1 min | Baixo |
| Deploy da nova versão | ~5 min | Baixo |
| **Total** | **~25-40 min** | **Baixo (sem GPU)** |

> O custo é aceitável para um sistema de cartão de crédito onde erros
> de classificação têm impacto financeiro direto.
