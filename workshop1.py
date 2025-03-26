import numpy as np


# 1. Creați o listă nouă cu ultimele cifre ale fiecărui număr din lista de mai jos
lista = [15, 29, 13, 27, 41]
sol = [x%10 for x in lista]
print(sol)


# 2. Fiind dată o listă de numere naturale, creați o altă listă în care pentru fiecare element din lista originală
# să rețină 'DA' dacă numărul este par, în caz contrar, 'NU'
lista = [10, 3, 8, 6, 5, 7]
sol = ["DA" if x%2==0 else "NU" for x in lista]
print(sol)


# 3. Pe baza unei liste, creați o altă listă care conține doar elementele aflate pe pozițiile de forma 3k + 2
lista = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sol = [lista[i] for i in range(len(lista)) if i%3==2]
print(sol)


# 4. Pentru a antrena un model, sunt necesare două dicționare: id2label și label2id
# Creați aceste două dicționare, știind că label2id are forma {label: poziția în șir},
# iar id2label are forma {poziția în șir: label}.

labels = ['cat', 'dog', 'car', 'duck', 'airplane', 'train', 'teddy bear']
label2id = {labels[i] : i for i in range(len(labels))}
id2label = {i : labels[i] for i in range(len(labels))}
print(label2id, id2label)


# 5. Dat fiind un șir de caractere, s,
# generați o listă, litere, care să conțină fiecare caracter din s cu majusculele și minusculele inversate.
# hint: Folosiți funcțiile islower() și isupper() pentru a determina tipul fiecărei litere
# și funcțiile lower() și upper() pentru a le transforma în minusculă, respectiv majusculă.

s = "Succes la NITRO NLP!"
litere = [x.lower() if x.isupper() else x.upper() for x  in s]
print(litere)
