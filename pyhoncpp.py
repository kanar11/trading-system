age = int(input("Enter your age: "))

if age < 0:
    print("Invalid age")
elif age < 13:
    print("Child ticket")
elif age < 18:
    print("Teen ticket")
elif age < 65:
    print("Adult ticket")
else:
    print("Senior ticket")