import ast

with open('../text/sentences.txt', 'r') as f:
    lines = f.read().splitlines()

with open('../text/sentences.csv', 'w') as f:
    # opcjonalnie, jeżeli chcesz dodać nagłówki
    f.write('Text,Class\n')
    for line in lines:
        # Przekształć ciąg znaków na krotkę i przekształć z powrotem do ciągu znaków, ale z przecinkami zamiast przecinków i nawiasów
        row = ['"{}"'.format(row) for row in ast.literal_eval(line)]
        f.write(','.join(row) + '\n')
