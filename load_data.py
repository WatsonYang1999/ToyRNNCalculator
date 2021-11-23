

def load_dataset(datapath:str):
    features = []
    labels = []
    seq_lens = []
    with open(datapath,'r') as f:
        print('WTF')
        for idx,line in enumerate(f.readlines()):
            line = '@' + line[:-1] + '#'
            if idx % 2 == 0 :

                features.append(line)


            else :
                assert len(line) != 0
                seq_lens.append(len(line))
                labels.append(line)

    return features,labels,seq_lens

