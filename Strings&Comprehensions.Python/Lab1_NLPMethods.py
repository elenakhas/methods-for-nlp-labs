def hello_world():
    print("Hello World")

def print_tick_tack_toe1():
    a = "     |  |      "
    b = "- - - - - - - -"
    print(a, b, sep = '\n')
    print(a, b, sep = '\n')
    print(a)

def print_tick_tack_toe2():
    a = "     |  |      \n"
    b = "- - - - - - - -\n"
    print((a + b)*2, a,  sep = '\n')

def print_tick_tack_toe3():
    a = "|".join(" "*4 for i in range(3))
    b = "-".join(" " for i in range(8))
    # alternation of two lines
    lines = [a if i%2 == 0 else b for i in range (5)]

    result = "\n".join(lines)
    print (result, sep = "\n")


#print_tick_tack_toe3()

def snow_white (num_chants, max_sing):
    first = "heigh"
    second = "ho"
    last = "it's off to work we go"

    verse = [first if i%2 ==0 else second for i in range (num_chants)]
    song = "\n".join(verse) + "\n" + last + "\n"

    while (max_sing != 0):
        max_sing -= 1
        print (song, sep = "\n", end = "\n")

#snow_white(5, 2)

#parameters - number of repetitions
def printing_challenge(num_rep):
    # define the line with 3 squares (consists of 2 lines)
    line_one = "  |  |  "
    line_two = "--+--+--"
    between_line = "H"

    between_row = "="*8
    between_row_line = "+"

    #if it is not the first line, add H in the beginning
    #concatenate the lines if the number of repetitions >1 (or 0?)
    for i in range (num_rep):
        line1 = line_one
        line2 = line_two
        between = between_row

        if (i != 0):
            line1 = between_line + line1
            line2 = between_line + line2
            between = between_row_line + between

        odd = line_one + line1 * num_rep + '\n'
        even = line_two + line2 * num_rep + '\n'
        between2 = between_row + between * num_rep + '\n'

        # repeat it twice, add the 5th line

    print(((odd+even)*2 + odd + between2) * num_rep)
    #repeat the whole thing number times

#printing_challenge(3)

def print_two(a, b):
    print("Arguments: {0} and {1}".format(a, b))
#print_two() # NO - no Arguments
#print_two(4, 1) #YES
#print_two(41) #NO missing argument
#print_two(a=4, 1) #NO, should declare both
#print_two(4, 1, 1) #NO, can take only 2
#print_two(b=4, 1) #NO, b defined twice
#print_two(4, a=1) #NO, a defined twice
#print_two(a=4, b=1) #YES
#print_two(b=1, a=4) #YES
#print_two(1, a=1) #NO, multiple values for a
#print_two(4, 1, b=1) #NO, too many args
#print_two(a=5, c=1) #NO, unexpected argument
#print_two(None, b=1) #YES

def keyword_args(a, b=1, c='X', d=None):
    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("d:", d)

#keyword_args(5) #YES
#keyword_args(a=5) #YES
#keyword_args(5, 8) #YES, b=8
##keyword_args(5, 2, c=4) #YES
#keyword_args(5, 0, 1) #YES
#keyword_args(5,2, d=8, c=4) YES
#keyword_args(5,2, 0, 1, "") No, too many
#keyword_args(c=7, 1) #NO positional argument follows keyword argument
#keyword_args(c=7, a=1) YES
#keyword_args(5, 2, [], 5) YES
#keyword_args(1, 7, e=6) NO, unexpected argument e
#keyword_args(1, c=7) YES
#keyword_args(5, 2, b=4) NO, b has multiple values
#keyword_args(d=a, b=d, a=b, c=d) NO, names are not defined
#keyword_args(d=5, c='', b=None, a='Cat') YES

def variadic(*args, **kwargs):
    print("Positionsl:", args)
    print("Keyword", kwargs)

#variadic (2, 3, 5, 7) YES
#variadic (1, 1, n=1) YES
#variadic (n=1, 2, 3) positional arguments follow keyword
#variadic () YES
#variadic (cs="Computer Science", pd = "Product design")
