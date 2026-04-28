import seth

def main():
    X = seth.NamedSet(name = "X", elements = {1, 2, 3, 4, 5})
    eq = {(1,3), (2,4)}

    Q = seth.Setoid(X = X, eq = eq)
    print(Q.content())
    print(Q.representation())
    print(Q.closure())
    print(Q.equalities())
    print(Q.quotient())
    print(Q.quotient().content())
    print(Q.projection().display())

if __name__ == "__main__":
    main()