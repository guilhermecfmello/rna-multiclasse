import sys
from neural import Neural

# Arguments tratment function
def getArgs(args, param):
  try:
    argValue = False
    for a in args:
      if(a == '-h'):
        print('============= Help Menu =============')
        print("Exec format: python main.py [args] | python main.py -h for help")
        print("\n\nCommands:")
        print('-i "databaseName"')
        return False
    for i in range(1, len(args)):
      if(args[i] == param):
        argValue = args[i+1]
    if(argValue == False): raise Exception("Param " + str(param) + " couldn't be found")
    return argValue
  except Exception as e:
    print("Exec format: python main.py [args] | python main.py -h for help.\nError: " + str(e))
    exit()    

dbName = getArgs(sys.argv, '-i')

neural = Neural(dbName)
# print(neural.inputs)
# print(neural.outputs)


ages = 200
# Learning rate
lr = 0.1
neural.trainning(ages, lr)
neural.testDatabase('database/test/dificil_teste.csv')
# print(neural.inputsTest)
neural.compareTest()
# neural.plot_all()
# neural.plot_all()
# print('weights: ' + str(neural.weights))
# print('bias: ' + str(neural.bias))
# rights, wrongs = compare(inputs, outputs, weights, bias)

# plot_all(inputs, outputs, weights)


