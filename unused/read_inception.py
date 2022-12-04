import pickle

with open("./metrics/inception_Recipe1M_salad.pkl","rb") as fr:
    data = pickle.load(fr)
    mean = data['mean']
    cov = data['cov']
    dataset_name = data['dataset_name']
    print(list(mean))
    print(list(cov))
    print(dataset_name)


    # with open("inception.txt",'w') as wf:
    #     wf.write(str(mean))
    #     wf.write(str(cov))
    #     wf.write(str(dataset_name))