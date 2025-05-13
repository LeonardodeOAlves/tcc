# Importando bibliotecas para Analise

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
#import statsmodels.api as sm
#from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
#from decimal import Decimal

# In[1]: Leitura da base

baseinicial = pd.read_excel("base-analise-atend-v2.xlsx")
baseinicial.info()
# baseinicial['Necessidade'] = baseinicial['Necessidade'].astype('int')
# baseinicial.info()

# Organizando em ordem crescente

baseanalise = baseinicial.sort_values(by=['DataAjustada'], ascending=True).reset_index(drop=True)
baseanalise.info()

# Lendo apenas as primeiras linhas das variaveis
print(baseanalise.head())

# Buscando necessidade
necessidade = baseanalise['NecessidadeAjustada']
necessidade.info()

# Lendo os dados iniciais brutos
print(necessidade.head())

# Análise descritiva
print("Resumo estatistico:")
print(necessidade.describe())
print("\nDesvio padrao:")
print(necessidade.std())
print("\nNumero de observacoes:")
print(len(necessidade))
# In[2]: Plotando o grafico bruto (selecione todos os comandos)

plt.figure(figsize=(10, 6))
plt.plot(necessidade)
plt.title('Necessidade Mensal de um Centro de Distribuição')
plt.xlabel('Tempo')
plt.ylabel('Quantidade')
plt.show()

# In[3]: Definindo necessidade como uma serie temporal
necessidade_ts = pd.Series(necessidade.values, index=pd.to_datetime(baseanalise['DataAjustada']))
necessidade_ts.info()

# Quantos registros tenho?
print(len(necessidade_ts)) 

# In[4]: Grafico como serie de tempo usando Plotly (Selecione todos os comandos)

plt.figure(figsize=(10, 6)) 
plt.plot(necessidade_ts)
plt.title('Itens Processados Centro de Distribuição - Serie Temporal')
plt.xlabel('Tempo (01/01/22 a 31/03/25)')
plt.ylabel('Quantidade (em MM Peças)')
#plt.yscale('linear')
#plt.yticks(np.arange(12,32,5))
plt.show() 

# In[5]:
############
# Decomposicao da serie temporal em suas componentes(Tendencia, Sazonal e 
# Residuos)

decompm = seasonal_decompose(necessidade_ts, model='multiplicative', period=12)

# observando os valores da decomposicao pelo modelo aditivo
print(decompm.trend)
print(decompm.seasonal) 
print(decompm.resid) 

# In[6]: Plotar a decomposicao (Selecionar todos os comandos)
plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(decompm.trend)
plt.title('Tendência')

plt.subplot(4, 1, 2)
plt.plot(decompm.seasonal)
plt.title('Componente Sazonal')

plt.subplot(4, 1, 3)
plt.plot(decompm.resid)
plt.title('Resí­duos')

plt.subplot(4, 1, 4)
plt.plot(necessidade_ts, label='Original')
plt.plot(decompm.trend * decompm.seasonal * decompm.resid, label='Reconstrui­da')
plt.title('Original vs. Reconstruí­da')
plt.legend()

plt.tight_layout()
plt.show()

# In[7]:

####################################################################
# Testes de Estacionariedade
####################################################################
# Teste de Dickey-Fuller
# H0: A série Não é Estacionária
# H1: A série é Estacionária


# Teste de Dickey-Fuller aumentado (ADF)
def dickey_fuller_test(series, title=''):
    result = adfuller(series)
    print(f'Teste de Dickey-Fuller para {title}')
    print(f'Estatística: {result[0]}')
    print(f'p-valor: {result[1]}')
    print('Critérios:')
    for key, value in result[4].items():
        print(f'{key}: {value}')
    print('Conclusão:', 'Estacionária' if result[1] < 0.05 else 'Não Estacionária')
    print()
    
# In[8]: Aplicando o teste de Dickey-Fuller
dickey_fuller_test(necessidade_ts, 'Necessidade')

# In[9]:
# Funções ACF e PACF
####################################################################
def plot_acf_pacf(series, lags=17, title=''):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plot_acf(series, lags=lags, ax=ax[0], title=f'ACF {title}')
    plot_pacf(series, lags=lags, ax=ax[1], title=f'PACF {title}', method='ywm')
    plt.show()

# In[10]: Plotando ACF e PACF das séries
plot_acf_pacf(necessidade_ts, title='Itens Processados') 

# In[11]: --208
#####################################################################################
# modelos ARIMA com Sazonalidade - SARIMA, possui os parâmetros P, D e Q Sazonais.
# Fica SARIMA(p,d,q)(P,D,Q)
#####################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from arch import arch_model

# In[12]: Dividindo em série de treino (serietreino) e teste (serieteste)

# Transformando a serie em um DataFrame e tornando a coluna DataAjustada como index
necessidade_df = pd.DataFrame({'DataAjustada':necessidade_ts.index, '0':necessidade_ts.values})
necessidade_df.set_index('DataAjustada', inplace=True)
#necessidade_df.columns={'DataAjustada','Necessidade'}

serietreino = necessidade_df[:'2024-09']
serieteste = necessidade_df['2024-10':'2025-03'] 

# In[13]: Plotando as séries de treino e teste juntas
plt.figure(figsize=(10, 6))
plt.plot(necessidade_df, label='Necessidade')
plt.plot(serietreino, label='Necessidade Treino')
plt.plot(serieteste, label='Necessidade Teste', color='blue')
plt.title("Série Treinada e Testada")
plt.xlabel('Data')
plt.ylabel('Quantidade (em MM Peças)')
plt.legend()
plt.grid(True)
plt.show()

# In[14]:
# Criar colunas 'Ano' e 'Mês' a partir do índice de datas
import seaborn as sns

necessidade_df['Ano'] = necessidade_df.index.year
necessidade_df['Mês'] = necessidade_df.index.month

# In[15]: Fazer o Gráfico com destaque para valores mensais
plt.figure(figsize=(10, 6))
sns.violinplot(x='Mês', y='0', data=necessidade_df, palette='viridis')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.title('Distribuição Mensal da Necessidade (2018-2025)')
plt.xlabel('Mês')
plt.ylabel('Quantidade (em MM Peças)')
plt.show()

# In[16]: Analisando a série com gráficos de ACF e PACF
plot_acf(serietreino)
plot_pacf(serietreino, method='ywm')
plt.show()

# In[17]: Teste de Estacionariedade - ADF (Dickey-Fuller)
result = adfuller(serietreino.dropna())
print(f'Resultado do Teste ADF: p-valor = {result[1]}')
if result[1] < 0.05:
    print("A série é estacionária.")
else:
    print("A série não é estacionária.")

# In[18]: Função para verificar quantas diferenciações são necessárias para tornar a série estacionária
import pmdarima as pm
def verificar_differenciacao(serie, nome):
    # Usar a função ndiffs do pmdarima
    d = pm.arima.ndiffs(serie, test='adf')  # Teste de Dickey-Fuller aumentado
    print(f"A série {nome} precisa de {d} diferenciação(ões) para ser estacionária.")
    return d

# In[18]: Verificar quantas diferenciacoes sao necessarias
verificar_differenciacao(serietreino, "Necessidade - Treinamento")

# Diferenciação para estacionariedade
necessidadetreino_diff = serietreino.diff().dropna()

# In[19]: Gráficos ACF e PACF da série diferenciada
fig, axes = plt.subplots(1, 2, figsize=(16,4))
plot_acf(necessidadetreino_diff, lags=24, ax=axes[0])
plot_pacf(necessidadetreino_diff, lags=24, ax=axes[1], method='ywm')
plt.show()

# In[20]: Ajuste do modelo ARIMA na série diferenciada (autoarima)
arimanecessidade = auto_arima(necessidadetreino_diff,
                         seasonal=True,
                         m=12,  # Periodicidade da sazonalidade
                         trace=True,
                         stepwise=True)

# Exibir o resumo do modelo ajustado
print(arimanecessidade.summary())

# In[21]: Validação e Diagnóstico

# Resíduos do modelo
residuos_arima = arimanecessidade.resid()
print(f"Resíduos do modelo: {residuos_arima}")

# In[22]: 1. Teste de Ljung-Box para verificar autocorrelação dos resíduos
from statsmodels.stats.diagnostic import acorr_ljungbox

ljung_box = acorr_ljungbox(residuos_arima, lags=[10], return_df=True)
print(f'Resultado do teste de Ljung-Box:\n{ljung_box}')

p_value_ljungbox = ljung_box['lb_pvalue'].values[0]
if p_value_ljungbox > 0.05:
    print("Não há evidências de autocorrelação significativa nos resíduos (não rejeitamos H0).")
else:
    print("Há evidências de autocorrelação nos resíduos (rejeitamos H0).")
    
# In[23]: 2. Teste de Normalidade dos Resíduos (Kolmogorov-Smirnov)
from scipy.stats import kstest
ks_stat, p_value = kstest(residuos_arima, 'norm', args=(np.mean(residuos_arima), np.std(residuos_arima)))
print(f'Teste de Kolmogorov-Smirnov para normalidade: p-valor = {p_value}')
if p_value > 0.01:
    print("Os resíduos seguem uma distribuição normal.")
else:
    print("Os resíduos não seguem uma distribuição normal.")
    
# In[24]: 3. Teste ARCH para verificar heterocedasticidade dos resíduos

from arch import arch_model

am = arch_model(residuos_arima, vol='ARCH', p=1)
test_arch = am.fit(disp='off')
print(test_arch.summary())
#se p-value > 0.05 - nao ha efeitos ARCH

# In[25]: Prever 24 passos à frente na série diferenciada
n_periods = 24
previsoes_diff = arimanecessidade.predict(n_periods=n_periods)
print(f"Previsões diferenciadas: {previsoes_diff}")

# In[26]: Índices das previsões (mesmo formato de data da série de treino e teste)
index_of_fc = pd.date_range(serietreino.index[-1], periods=n_periods+1, freq='MS')[1:]

# In[27]: Para voltar ao nível original:
# Iterar para reverter a diferenciação das previsões
ultimo_valor_original = serietreino.iloc[-1] # Último valor conhecido da série original (não diferenciada)
previsoes_nivel_original = [ultimo_valor_original]
print(ultimo_valor_original)
print(previsoes_nivel_original)

# In[28]: Somar as previsões diferenciadas ao último valor conhecido da série original
for previsao in previsoes_diff:
    novo_valor = previsoes_nivel_original[-1] + previsao
    previsoes_nivel_original.append(novo_valor)
    
# In[29]: Remover o primeiro valor, pois é o último valor conhecido da série original
previsoes_nivel_original = previsoes_nivel_original[1:]
print(previsoes_nivel_original)

# In[30]: Converter previsões de volta para uma Série Pandas com o índice correto
previsoes_nivel_original_series = pd.Series(previsoes_nivel_original, index=index_of_fc)
print(previsoes_nivel_original_series)

# In[31]: Plotando as previsões no nível original junto com a série de treino e teste
plt.figure(figsize=(10, 6))
plt.plot(serietreino, label='Treino')
plt.plot(serieteste, label='Teste', color='blue')
plt.plot(previsoes_nivel_original_series, label='Previsão ARIMA - Nível Original', color='orange')
plt.legend()
plt.title('Previsão ARIMA para Necessidade Mensal (24 Passos à Frente - Nível Original)')
plt.grid(True)
plt.show()

# In[32]: Garantir que as previsões e os valores reais estejam alinhados para o MAPE
previsoes_series_alinhadas = previsoes_nivel_original_series[:len(serieteste)].dropna()
necesidadeteste_alinhada = serieteste.loc[previsoes_series_alinhadas.index]

# In[33]: Calcular o MAPE
from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(necesidadeteste_alinhada, previsoes_series_alinhadas)*100
print(f'MAPE: {mape}')

# In[34]: Ajustar o modelo ETS (Holt-Winters Exponential Smoothing) - para serie varejotreino
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from arch import arch_model
from scipy.stats import kstest
from sklearn.metrics import mean_absolute_percentage_error as mape
import statsmodels.api as sm

ets_model = ExponentialSmoothing(serietreino, seasonal='add', trend='add', seasonal_periods=12).fit()

# In[205]: Previsões para os próximos 24 passos
ets_forecast = ets_model.forecast(steps=24)
print(f'Previsões ETS: {ets_forecast}')

# In[206]: Plotando os valores reais e as previsões
plt.figure(figsize=(10, 6))
plt.plot(serietreino, label='Treino')
plt.plot(serieteste, label='Teste', color='blue')
plt.plot(ets_forecast, label='Previsão ETS', color='orange')
plt.legend()
plt.title('Previsão ETS para Necessidade Mensal de um Centro de Distribuição - 24 Passos à Frente')
plt.grid(True)
plt.show()

# In[207]: Avaliação do desempenho do modelo usando MAPE
mape_ets = mape(serieteste, ets_forecast[:len(serieteste)])*100
print(f'MAPE ETS: {mape_ets}') 