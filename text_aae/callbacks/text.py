
def convert_texts(x, charset):
    for i in range(x.shape[1]):
        yield "".join(charset[c] for c in x[:, i])
