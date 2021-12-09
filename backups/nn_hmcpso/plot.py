import json
hmcempso = json.load(open('loss_hmcempso.json'))
hmcrempso = json.load(open('loss_hmcrempso.json'))
empso = json.load(open('loss_empso.json'))
h2 = json.load(open('hmcpso5.json'))
from matplotlib import pyplot as plt
plt.plot(hmcempso,label='HMC+EMPSO')
plt.plot(hmcrempso,label='HMC+REMPSO')
plt.plot(empso,label='EMPSO')
plt.plot(h2,label='HMCPSO2')
plt.legend()
plt.show()
