import numpy as np

# try:
#     from cppimport import imp_from_filepath
#     from os.path import join, dirname
#     path = join(dirname(__file__), "sources/sampling.cpp")
#     sampling = imp_from_filepath(path)
#     sampling.seed(config.seed)
#     sample_ext = True
# except:
#     display.color_print("Cpp extension not loaded")
#     sample_ext = False

def UniformSample_original(train_list, train_size, n_users, m_items, neg_ratio = 1):
    # allPos = dataloader.train_list
    # start = time()
    # if sample_ext:
    #     S = sampling.sample_negative(dataset.n_users, dataset.m_items,
    #                                  dataset.trainDataSize, allPos, neg_ratio)
    # else:
    #     S = UniformSample_original_python(dataset)
    S = UniformSample_original_python(train_list, train_size, n_users, m_items)
    return S

def UniformSample_original_python(train_list, train_size, n_users, m_items):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """

    user_num = train_size
    users = np.random.randint(0, n_users, user_num)
    allPos = train_list
    S = []
    for user in users:
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
    return np.array(S)
