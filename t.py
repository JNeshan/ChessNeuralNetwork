# Return type BOOLEAN
# Parameter 1 STRING input
def stringCheck(input):
    if(len(input) != 22):
        return False
    sub = input[0:3]
    if sub.isdigit() == False:
        return False
    sub = input[3:5]
    if sub != "--":
        return False
    sub = input[5:8]
    if sub.isalpha() == False:
        return False
    sub = input[8:10]
    if sub != "--":
        return False
    sub = input[10:12]
    if sub[0].isdigit() == False:
        return False
    if sub[1].isalpha() == False:
        return False
    sub = input[12:14]
    if sub != "--":
        return False
    sub = input[14:17]
    if sub.isalpha() == False:
        return False
    sub = input[17:19]
    if sub != "--":
        return False
    sub = input[19:22]
    if sub.isdigit() == False:
        return False
    return True





def main():
    word = "431--vzx--1a--lak--9011"
    print("test")
    print(stringCheck(word))

if __name__ == "__main__":
    main()