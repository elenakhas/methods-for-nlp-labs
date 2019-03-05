def comprehensions():
    print ([x for x in [1, 2, 3, 4]])
    print ([n-2 for n in range(10)])
    print ([k%10 for k in range(41) if k%3 == 0])
    print ([s.lower() for s in ['PythOn', 'iS', 'cOol'] if s[0] < s[-1]]) #why doesn't it print iS?
    arr = [[3, 2, 1], ['a', 'b', 'c'], [('do' ,), ['re'], 'mi']]
    print([el.append(el[0]*4) for el in arr])
    print(arr)
    print ([letter for letter in "pYthON" if letter.isupper()])
    print ({len(w) for w in ["its", "the", "remix", "to", "ignition"]})
    print ([i*2+1 for i in [0, 1, 2, 3]])
    print ([m[0].upper() for m in ['apple', 'orange', 'pear']])
    print ([word for word in ['apple', 'orange', 'pear'] if 'p' in word])
    print ([elem[1] for elem in [part.split('_') for part in ["TA_sam", "student_poohbear", "TA_guido", "student_htiek"]] if elem[0] == 'TA'])
    print ([(y, len(y)) for y in ['apple', 'orange', 'pear']])
    print ({y : len(y) for y in ['apple', 'orange', 'pear']})
    print ([c for c, v in enumerate(['apple', 'orange', 'pear'])])

#comprehensions()

def pascal(current):

    next = [] #a container for the row to generate
    for i, v in enumerate(current): #take indices from the current list
        if i == 0:
            next.append(current[i])
        else:
             next.append(current[i] + current[i-1])
    next.append(1)
    print (next)

#pascal([0])
#pascal([1])
#pascal([1,1])
#pascal([1, 2, 1])
#pascal([1, 3, 3, 1])
#pascal([1, 4, 6, 4, 1])

def generate_pascal_row(current): #same function using list comprehension
    next = []
    [next.append(current[i] + current[i-1]) if i != 0 else next.append(current[i]) for i, v in enumerate(current)]
    next.append(1)
    print(next)

#generate_pascal_row([1, 1])
import urllib
def is_triad_phrase(string):
    #split the string into words - result is a list
    list_words = string.split(" ")
    with open('/usr/share/dict/words') as f: #read the dictionary file into a list
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        f.close()
    #small_words = [] # a storage for the new words
    for word in list_words: #iterate over two words in the list
        odd = ""
        even = ""

        for i in range (len(word)): #iterate over characters in a words, build the new words
            if i % 2 == 0:
                even = even + word[i]
            else:
                odd = odd + word[i]
        return (False if odd not in lines or even not in lines else True)
        #small_words.append(odd) #append the new words to the list
        #small_words.append(even)
    #print (small_words)


    #print(lines[0: 15])
    #counter = 0
    #for w in small_words:
    #    if w in lines:
    #        counter += 0
    #    else:
    #        counter += 1
    #if counter == 0:
    #    print ("True")
    #else:
    #    print("False")


    #print("True" if w in lines else "False" for w in small_words)


#print ("The line" + string + "is made of triad words" if True else "The line is not made of triad words")
#print(is_triad_phrase("learned theorem"))
#print(is_triad_phrase("studied theories"))
#is_triad_phrase("wooded agrarians")
#is_triad_phrase("forrested farmers")
#is_triad_phrase("schooled oriole")
#is_triad_phrase("educated small bird")
#is_triad_phrase("a")
#is_triad_phrase("")

def is_surpassing_phrase(string):
    list_words = string.split(" ")
    for word in list_words:
        first = 0
        second = 0
        if len(word) == 1 or len(word) == 0:
            return True
        else:
            for i in range(1, len(word)-1): # get the values for every 2 characters
                first = abs(ord(word[i-1]) - ord(word[i])) #get the
                second = abs(ord(word[i+1]) - ord(word[i]))
        return (False if first > second else True)

#print (is_surpassing_phrase("superb subway"))
#print (is_surpassing_phrase("excellent train"))
#print (is_surpassing_phrase("porky hogs"))
#print (is_surpassing_phrase("plump pigs"))
#print (is_surpassing_phrase("turnip fields"))
#print (is_surpassing_phrase("root vegetable lands")) #why does it return true for this?
#print (is_surpassing_phrase("a"))
#print (is_surpassing_phrase(""))

def is_cyclone_phrase(string):
    list_words = string.split(" ")
    for word in list_words:
        for i in range ((len(word)-1)//2):
            curr_letter = word[i]
            next_letter = word[-i-1]
            if curr_letter > next_letter:
                return False
            curr_letter = word[-i-1]
            next_letter = word[i+1]
            if curr_letter > next_letter:
                return False
        if(len(word))%2 == 0 and len(word)>0:
            if word[len(word)//2 - 1] > word[len(word)//2]:
                return False
    return True

print(is_cyclone_phrase("adjourned"))
print(is_cyclone_phrase("settled"))
print(is_cyclone_phrase("all alone at noon"))
print(is_cyclone_phrase("by myself at twelve pm"))
print(is_cyclone_phrase("acb"))
print(is_cyclone_phrase(""))
