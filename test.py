# def friends(greet, *args, **kwargs):
#     for names in args:
#         print(f"{greet} to the Programming zone {names}")
#     print("\nI am Veronica and I would like to announce your roles:")
#     for key, value in kwargs.items():
#         print(f"{key} is a {value}")

# greet = "Welcome"
# names = ["Sachin", "Rishu", "Yashwant", "Abhishek"]
# roles = {"Sachin":"Chief Instructor", "Rishu":"Engineer",
#          "Yashwant":"Lab Technician", "Abhishek":"Marketing Manager"}

# friends(greet, *names, **roles)
# switch = {"valueA":functionA,"valueB":functionB,"valueC":functionC}
# try:
#　　switch["value"]() #执行相应的方法。
# except KeyError as e:
#       pass 或 functionX #执行default部分

switch = {
    "a":lambda x:x*2,
    "b":lambda x:x*3,
    "c":lambda x:x**x
}
try:
    switch["c"](6)
except KeyError as e:
    pass
