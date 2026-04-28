import seth
def main():
    X = seth.NamedSet(name = "X", elements = {1, 2, 3})
    Y = seth.NamedSet(name = "Y", elements = {"a", "b"})
    Z = seth.NamedSet(name = "Z", elements = {"x", "y", "z"})

    print(seth.HomSet(X,Y).display())
    print(seth.HomSet(X,Y).__len__())
    #print(seth.currying(X,Y,Z).display())
    print(seth.Bij(X,Z).display())
    print('is initial Bij(X,Y) ?' + str(seth.initial_detection(seth.Bij(X,Y))))

if __name__ == "__main__":
    main()