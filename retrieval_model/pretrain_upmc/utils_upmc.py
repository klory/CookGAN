import os

def get_classes(data_dir):
    with open(os.path.join(data_dir, 'meta/classes.txt')) as f:
        lines = f.readlines()
    return [line.rstrip() for line in lines]

def gen_filelist(data_dir, part):
    assert part in ('train, test'), 'part should be train|test'
    classes = get_classes(data_dir)
    table = {c:i for i,c in enumerate(classes)}
    with open(os.path.join(data_dir, 'meta/'+part+'.txt'), 'r') as f:
        lines = f.readlines()
        output = ''
    for line in lines:
        c = line.split('/')[0]
        temp = line.rstrip() + '.jpg ' + str(table[c]) + '\n'
        output += temp
    with open(os.path.join(data_dir, part+'.txt'), 'w') as f1:
        f1.write(output)

if __name__ == '__main__':
    for part in('train', 'test'):
        gen_filelist('UPMC-Food-101', part)