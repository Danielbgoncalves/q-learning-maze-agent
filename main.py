"""
    Aplicação de aprendizado por reforço com Q-learning 
    para aprender como essa bagaça funciona.

    A tabela Q terá shape 16x4, 16 linhas uma pra cada estado com 4 colunas cada indicando 
    o quão bom seria ir em cada direção estando nesse estado.

     0  1  2  3  
     4  5  6  7  
     8  9 10 11  
    12 13 14 15

     0  1  2  3  4  5  6  7  8  9  
    10 11 12 13 14 15 16 17 18 19
    20 21 22 23 24 25 26 27 28 29
    30
    40
    50
    60
    70
    80
    90 91 92 93 94 95 96 97 98 99
"""

import numpy as np
from enum import IntEnum
import random


# ==== Configurações iniciais ====

TAM_MAPA = 10
NMR_ESTADOS = TAM_MAPA**2 # um grid 10x10
NMR_ACOES = 4 # CIMA=0, DIREITA=1, BAIXO=2, ESQUERDA=3

class action(IntEnum):
    CIMA = 0
    DIREITA = 1 
    BAIXO = 2
    ESQUERDA = 3

ACTIONS = [action.CIMA, action.DIREITA, action.BAIXO, action.ESQUERDA]

# ==== Funções ====

def set_final_states():
    global FINAL_STATES

    for i in range(TAM_MAPA):
        for j in range(TAM_MAPA):
            if not MAP[i][j] == 0:
                FINAL_STATES.append(pos_to_state(i,j))
    

def state_to_pos(state):
    y = state // TAM_MAPA  # linha
    x = state % TAM_MAPA   # coluna
    return x, y

def pos_to_state(x,y):
    return y * TAM_MAPA + x

def get_reward(x,y):
    return MAP[y][x]  

def is_done(state):
    return state in FINAL_STATES

def execute_action(state, act):
    x, y = state_to_pos(state)

    if act == action.BAIXO and y < TAM_MAPA-1:
        y += 1 
    elif act == action.ESQUERDA and x > 0:  
        x -= 1
    elif act == action.CIMA and y > 0:
        y -= 1
    elif act == action.DIREITA and x < TAM_MAPA-1:
        x += 1

    new_state = pos_to_state(x,y)
    reward = get_reward(x,y)
    
    # Adiciona uma pequena recompensa negativa para encorajar movimento eficiente
    if reward == 0:
        reward = -0.01
        
    done = is_done(new_state)

    return new_state, reward, done

def update_q_table(alpha, reward, gama, Q, state, new_state, action):
    peso_atual = Q[state][action]
    Q[state][action] = alpha * ( reward + gama* np.max(Q[new_state]) - peso_atual ) + peso_atual

def get_action(Q, state, epsilon):
    global ACTIONS
    if np.random.rand() < epsilon:
        return random.choice(ACTIONS)
    else:
        if np.all(Q[state] == 0):
            return random.choice(ACTIONS)
        return np.argmax(Q[state])

# ==== Mapa do ambiente ====

# Mapa de busca
MAP = np.zeros((TAM_MAPA,TAM_MAPA))

# ==== Áreas especiais ====
FINAL_STATES = []

# Área com recompensa valor 10
MAP[9][9] = 10  # Adicionar recompensa no estado alvo

# Adicionar penalidade nas armadilhas
# MAP[2][2] = -10
# MAP[3][3] = -10
MAP[4][3] = -10
MAP[4][5] = -10
MAP[4][6] = -10
# MAP[4][7] = -10
# MAP[4][8] = -10
# MAP[4][9] = -10
# MAP[6][0] = -10
# MAP[6][1] = -10
MAP[6][2] = -10
MAP[6][3] = -10

set_final_states()

# ==== Q-Table ====
Q = np.zeros((NMR_ESTADOS,NMR_ACOES)) 
print(Q[0])


# ==== Que comecem os jogos - e que a sorte esteja sempre a seu favor ====

print("="*23, "Mapa", "="*23)
print(MAP)
print("="*60)

alpha = 0.1    # o quanto a nova experiencia é importante
gamma = 0.95   # o quanto o futuro importa
epsilon = 1.0  # exploração máxima no início
epsilon_min = 0.05  # Mínimo menor para mais exploração
decay = 0.9999  # Decay mais lento para explorar por mais tempo

for i in range(50_000): 
    state = 0
    done = False

    while not done:
        a = get_action(Q,state,epsilon)
        
        new_state, reward, done = execute_action(state, a)
        update_q_table(alpha,reward, gamma, Q, state, new_state, a)
        state = new_state

    epsilon = max(epsilon_min, epsilon*decay)

    if i % 1000 == 0:
        print(f"Episodio {i}: epsilon: {epsilon:.4f} ultimo estado: {state}")


print("Resultado final da tabela Q:")
print(Q)

state = 0
for i in range(30):
    if np.all(Q[state] == [0,0,0,0]): # Ok que isso não deveria acontecer em um agente bem treinado ams se estiver só meio terinado vai funcionar agora
        a = random.choice(ACTIONS)
    else:
        a = np.argmax(Q[state])
    new_state, _, done = execute_action(state,a)
    state = new_state
    print(f"Passo {i}: {state} done: {done}")
    if done: break