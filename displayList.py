def display_list(list_name, level):
    for list_element in  list_name :
        if isinstance(list_element, list) :
            display_list(list_element, level+1)
        else :
            for indent in range(level) :
                print("\t", end='')  
            print(list_element)
            
            
