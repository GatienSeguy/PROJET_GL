import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Série temporelle fictive
dates = pd.date_range(start='2025-01-01', periods=20, freq='D')
values = np.random.randn(20).cumsum()

split_index = 10

plt.plot(dates[:split_index+1], values[:split_index+1], color='blue', label='Avant barre')
plt.plot(dates[split_index:], values[split_index:], color='red', label='Après barre')

# Barre verticale
plt.axvline(dates[split_index], color='black', linestyle='--', label='Séparation')

plt.legend()
plt.xlabel('Date')
plt.ylabel('Valeur')
plt.title('Série temporelle reliée avec barre de séparation')
plt.show()

