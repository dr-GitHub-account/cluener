from operator import index
import random
import torch
import numpy as np

# object instantialization:
# train_loader = DatasetLoader(data=train_dataset, batch_size=args.batch_size,
#                                 shuffle=False, seed=args.seed, sort=True,
#                                 vocab = processor.vocab,label2id = args.label2id)
class DatasetLoader(object):
    def __init__(self, data, batch_size, shuffle, vocab,label2id,seed, sort=True):
        print("***********__init__() of class DatasetLoader called***********")
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.seed = seed
        self.sort = sort
        self.vocab = vocab
        self.label2id = label2id
        self.reset()

    # reset()在__init__()函数中被调用，preprocess()在reset()函数中被调用
    def reset(self):
        print("***********reset() of class DatasetLoader called***********")
        # data中一个句子的格式{'id': xx, 
                            # 'context': xx, 
                            # 'tag': xx, (表达成B-xx, I-xx)
                            # 'raw_context': xx}
        # preprocess()返回一个列表，列表中每个元素对应data中一个数据的(tokens, tag_ids, x_len, text_a, text_tag)
        # tokens：该句子的词转换成的id构成的列表，注意词转换成的id很多
        # tag_ids：该句子中的词对应的标签的id构成的列表，注意标签的id很少
        # x_len：该句子的tokens长度，即词个数
        # text_a：该句子'context'键对应的值
        # text_tag：该句子'tag'键对应的值
        self.examples = self.preprocess(self.data)
        # print("np.shape(self.examples):", np.shape(self.examples))
        # # np.shape(self.examples): (10748, 5)，一个大列表含10748个元组，每个元组含5个元素
        # 即10748条训练数据，对应10748个(tokens, tag_ids, x_len, text_a, text_tag)形式的元组
        # print("self.examples[0], self.examples[1]:", '\n', self.examples[0], '\n', self.examples[1])
        # # self.examples[0], self.examples[1]: 
        # # ([1616, 155, 36, 14, 713, 73, 94, 322, 62, 1233, 376, 1738, 213, 285, 445, 178, 692, 10, 35, 580, 216, 53, 588, 192, 234, 2296, 144, 14, 21, 298, 902, 7, 1233, 376, 1738, 365, 26, 5, 53, 119, 55, 20, 159, 155, 73, 36, 14, 123, 672, 5], [3, 13, 13, 13, 0, 0, 0, 0, 0, 7, 17, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 50, '浙 商 银 行 企 业 信 贷 部 叶 老 桂 博 士 则 从 另 一 个 角 度 对 五 道 门 槛 进 行 了 解 读 。 叶 老 桂 认 为 ， 对 目 前 国 内 商 业 银 行 而 言 ，', 'B-company I-company I-company I-company O O O O O B-name I-name I-name O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O') 
        # # ([87, 87, 24, 422, 255, 287, 332, 607, 87, 300, 1110, 1187, 320, 217, 2079, 808, 1110, 2142], [0, 0, 0, 0, 4, 14, 14, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 18, '生 生 不 息 C S O L 生 化 狂 潮 让 你 填 弹 狂 扫', 'O O O O B-game I-game I-game I-game O O O O O O O O O O')
        
        # sort=True
        if self.sort:
            # 对self.examples里的10748个元组，根据元组的第三个元素排序，即按照文本长度从长到短排序
            self.examples = sorted(self.examples, key=lambda x: x[2], reverse=True)
            # print("np.shape(self.examples) after sorted:", np.shape(self.examples))
            # # np.shape(self.examples) after sorted: (10748, 5)
            # print("self.examples[0], self.examples[-1]:", '\n', self.examples[0], '\n', self.examples[-1])
            # # ([1616, 155, 36, 14, 713, 73, 94, 322, 62, 1233, 376, 1738, 213, 285, 445, 178, 692, 10, 35, 580, 216, 53, 588, 192, 234, 2296, 144, 14, 21, 298, 902, 7, 1233, 376, 1738, 365, 26, 5, 53, 119, 55, 20, 159, 155, 73, 36, 14, 123, 672, 5], [3, 13, 13, 13, 0, 0, 0, 0, 0, 7, 17, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 50, '浙 商 银 行 企 业 信 贷 部 叶 老 桂 博 士 则 从 另 一 个 角 度 对 五 道 门 槛 进 行 了 解 读 。 叶 老 桂 认 为 ， 对 目 前 国 内 商 业 银 行 而 言 ，', 'B-company I-company I-company I-company O O O O O B-name I-name I-name O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O') 
            # # ([2416, 1813], [7, 17], 2, '涂 艳', 'B-name I-name')
        # shuffle=False
        if self.shuffle:
            indices = list(range(len(self.examples)))
            random.shuffle(indices)
            self.examples = [self.examples[i] for i in indices]
        
        # self.features是一个列表，列表中每个元素也是一个列表，是从self.examples中取的一个batch的数据
        self.features = [self.examples[i:i + self.batch_size] for i in range(0, len(self.examples), self.batch_size)]
        # print("np.shape(self.features):", np.shape(self.features))
        # # np.shape(self.features): (168,)
        # print("np.shape(self.features[-2]):", '\n', np.shape(self.features[-2]))
        # # np.shape(self.features[-2]): 
        # #  (64, 5)
        print(f"{len(self.features)} batches created")

    def preprocess(self, data):
        """ Preprocess the data and convert to ids. """
        print("***********preprocess() of class DatasetLoader called***********")
        processed = []
        for d in data:
            # data_processor.py中CluenerProcessor类的_create_examples()函数将键名转为了'context'、'tag'
            text_a = d['context']
            tokens = [self.vocab.to_index(w) for w in text_a.split(" ")]
            x_len = len(tokens)
            text_tag = d['tag']
            tag_ids = [self.label2id[tag] for tag in text_tag.split(" ")]
            processed.append((tokens, tag_ids, x_len, text_a, text_tag))
        return processed

    def get_long_tensor(self, tokens_list, batch_size, mask=None):
        """ Convert list of list of tokens to a padded LongTensor. """
        # print("***********get_long_tensor() of class DatasetLoader called***********")
        token_len = max(len(x) for x in tokens_list)
        tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        mask_ = torch.LongTensor(batch_size, token_len).fill_(0)
        for i, s in enumerate(tokens_list):
            tokens[i, :len(s)] = torch.LongTensor(s)
            if mask:
                mask_[i, :len(s)] = torch.tensor([1] * len(s), dtype=torch.long)
        if mask:
            return tokens, mask_
        return tokens

    def sort_all(self, batch, lens):
        """ Sort all fields by descending order of lens, and return the original indices. """
        # print("***********sort_all() of class DatasetLoader called***********")
        unsorted_all = [lens] + [range(len(lens))] + list(batch)
        sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
        return sorted_all[2:], sorted_all[1]

    def __len__(self):
        print("***********__len__() of class DatasetLoader called***********")
        # return 50
        return len(self.features)

    # 有168个batches时，index在[0, 167之间取值]
    def __getitem__(self, index):
        """ Get a batch with index. """
        # print("***********__getitem__() of class DatasetLoader called***********")
        if not isinstance(index, int):
            raise TypeError
        if index < 0 or index >= len(self.features):
            raise IndexError
        # 把一个批量的数据存到变量batch中，例如本脚本上面试验中的self.features[-2]
        batch = self.features[index]
        # 当前批量大小，除了最后一个批量，其它批量的大小基本都一样
        batch_size = len(batch)
        # (tokens, tag_ids, x_len, text_a, text_tag)
        # 将batch变为含有五个元组的列表，每个元组有64个元素
        # 第一个元组含有64个句子的tokens(64个列表)
        # 第二个元组含有64个句子的tag_ids(64个列表)
        # 第三个元组含有64个句子的x_len(64个整形)
        # 第四个元组含有64个句子的text_a(64个字符串)
        # 第五个元组含有64个句子的text_tag(64个字符串)
        batch = list(zip(*batch))
        # print("np.shape(batch):", '\n', np.shape(batch))
        # # np.shape(batch): 
        # #  (5, 64)
        # lens为当前batch中所有句子长度构成的列表
        lens = [len(x) for x in batch[0]]
        # if index == 166:
        #     print(lens)
        #     # [10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        batch, orig_idx = self.sort_all(batch, lens)
        
        # chars是一个列表，列表中元素为该batch中所有句子转化成id后的结果(长度降序排列)
        chars = batch[0]
        # if index == 166:
        #     print("np.shape(chars):", np.shape(chars))
        #     # np.shape(chars): (64,)
        
        # Convert list of list of tokens to a padded LongTensor
        # input_ids, input_mask, label_ids的第一个维度为batch_size，第二个维度统一为当前batch中最长句子的长度，长度不足的，get_long_tensor()中用0填充
        input_ids, input_mask = self.get_long_tensor(chars, batch_size, mask=True)
        label_ids = self.get_long_tensor(batch[1], batch_size)
        input_lens = [len(x) for x in batch[0]]
        # if index == 166:
        #     print("input_ids:", '\n', input_ids)
        #     print("input_mask:", '\n', input_mask)
        #     print("label_ids:", '\n', label_ids)
        #     print("input_lens:", '\n', input_lens)
        # if index == 166:
        #     print("np.shape(input_ids):", '\n', np.shape(input_ids))
        #     print("np.shape(input_mask):", '\n', np.shape(input_mask))
        #     print("np.shape(label_ids):", '\n', np.shape(label_ids))
        #     print("np.shape(input_lens):", '\n', np.shape(input_lens))
        # # np.shape(input_ids): 
        # # torch.Size([64, 10])
        # # np.shape(input_mask): 
        # # torch.Size([64, 10])
        # # np.shape(label_ids): 
        # # torch.Size([64, 10])
        # # np.shape(input_lens): 
        # # (64,)
        # if index == 0:
        #     print("np.shape(input_ids):", '\n', np.shape(input_ids))
        #     print("np.shape(input_mask):", '\n', np.shape(input_mask))
        #     print("np.shape(label_ids):", '\n', np.shape(label_ids))
        #     print("np.shape(input_lens):", '\n', np.shape(input_lens))
        # # np.shape(input_ids): 
        # # torch.Size([64, 50])
        # # np.shape(input_mask): 
        # # torch.Size([64, 50])
        # # np.shape(label_ids): 
        # # torch.Size([64, 50])
        # # np.shape(input_lens): 
        # # (64,)
        return (input_ids, input_mask, label_ids, input_lens)