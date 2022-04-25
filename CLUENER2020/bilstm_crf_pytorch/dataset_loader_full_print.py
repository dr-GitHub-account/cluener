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
        # print("self.features[-2]:", '\n', self.features[-2]) # 最后一个列表只有60句，所以看倒数第二个，总共167(完整的batch)×64+1(最后一个batch)×60=10748个句子
        # # self.features[-2]: 
        # # [([312, 116, 42, 430, 861, 54, 399, 470, 791, 38], [0, 0, 0, 8, 18, 18, 18, 18, 18, 18], 10, '户 名 ： 深 圳 市 红 十 字 会', 'O O O B-organization I-organization I-organization I-organization I-organization I-organization I-organization'), ([2916, 59, 162, 115, 64, 1406, 600, 442, 2006, 2420], [0, 0, 0, 9, 19, 7, 17, 17, 9, 19], 10, '⊙ 本 报 记 者 唐 真 龙 邹 靓', 'O O O B-position I-position B-name I-name I-name B-position I-position'), ([296, 375, 235, 338, 29, 42, 1062, 1082, 302, 943], [4, 14, 14, 14, 14, 14, 14, 14, 14, 14], 10, '魔 兽 争 霸 3 ： 冰 封 王 座', 'B-game I-game I-game I-game I-game I-game I-game I-game I-game I-game'), ([279, 50, 148, 42, 13, 621, 397, 76, 1477, 12], [0, 0, 0, 0, 4, 14, 14, 14, 14, 14], 10, '原 作 品 ： 《 雷 神 之 锤 》', 'O O O O B-game I-game I-game I-game I-game I-game'), ([13, 399, 541, 732, 12, 658, 378, 502, 184, 380], [2, 12, 12, 12, 12, 0, 0, 0, 0, 0], 10, '《 红 楼 梦 》 初 回 限 定 版', 'B-book I-book I-book I-book I-book O O O O O'), ([440, 319, 1634, 6, 721, 189, 12, 187, 17, 394], [2, 12, 12, 12, 12, 12, 12, 0, 0, 0], 10, '列 那 狐 的 故 事 》 第 2 集', 'B-book I-book I-book I-book I-book I-book I-book O O O'), ([36, 139, 69, 187, 10, 170, 547, 39, 20, 234], [8, 18, 0, 0, 0, 0, 0, 0, 0, 0], 10, '银 联 卡 第 一 次 走 出 国 门', 'B-organization I-organization O O O O O O O O'), ([107, 339, 2144, 166, 844, 87, 302, 732, 1032], [7, 17, 17, 9, 19, 19, 7, 17, 17], 9, '方 传 柳 实 习 生 王 梦 菲', 'B-name I-name I-name B-position I-position I-position B-name I-name I-name'), ([340, 196, 1126, 36, 197, 632, 73, 224, 352], [0, 0, 3, 13, 0, 0, 0, 0, 0], 9, '接 手 荷 银 亚 洲 业 务 ？', 'O O B-company I-company O O O O O'), ([206, 393, 59, 819, 115, 64, 1864, 1373, 1373], [0, 0, 0, 0, 9, 19, 7, 17, 17], 9, '文 / 本 刊 记 者 冯 娜 娜', 'O O O O B-position I-position B-name I-name I-name'), ([318, 512, 285, 105, 1125, 713, 376, 925, 122], [0, 0, 0, 0, 9, 19, 19, 19, 0], 9, '张 女 士 （ 私 企 老 板 ）', 'O O O O B-position I-position I-position I-position O'), ([549, 91, 208, 272, 20, 176, 380, 12, 7], [4, 14, 14, 14, 14, 14, 14, 14, 0], 9, '完 美 世 界 国 际 版 》 。', 'B-game I-game I-game I-game I-game I-game I-game I-game O'), ([514, 336, 537, 271, 198, 388, 749, 26, 807], [3, 13, 9, 19, 19, 19, 7, 17, 17], 9, '易 兰 创 意 总 监 余 为 群', 'B-company I-company B-position I-position I-position I-position B-name I-name I-name'), ([65, 630, 355, 465, 198, 57, 102, 299, 101], [3, 13, 13, 13, 9, 19, 19, 7, 17], 9, '新 浪 乐 居 总 经 理 罗 军', 'B-company I-company I-company I-company B-position I-position I-position B-name I-name'), ([1570, 128, 101, 328, 562, 202, 258, 88, 7], [7, 17, 17, 0, 0, 3, 13, 13, 0], 9, '彭 小 军 告 诉 c b n 。', 'B-name I-name I-name O O B-company I-company I-company O'), ([240, 107, 576, 162, 115, 64, 995, 1141, 1794], [2, 12, 12, 12, 9, 19, 7, 17, 17], 9, '东 方 早 报 记 者 刘 秀 浩', 'B-book I-book I-book I-book B-position I-position B-name I-name I-name'), ([565, 875, 60, 270, 86, 1238, 1122, 416, 60], [8, 18, 18, 0, 0, 8, 18, 18, 18], 9, '切 沃 队 v s 锡 耶 纳 队', 'B-organization I-organization I-organization O O B-organization I-organization I-organization I-organization'), ([834, 456, 358, 195, 219, 1486, 22, 318, 826], [3, 13, 13, 13, 9, 19, 19, 7, 17], 9, '汇 石 投 资 合 伙 人 张 剑', 'B-company I-company I-company I-company B-position I-position I-position B-name I-name'), ([29, 114, 1156, 1072, 270, 86, 221, 177, 620], [0, 0, 8, 18, 0, 0, 8, 18, 18], 9, '3 . 拜 仁 v s 布 加 勒', 'O O B-organization I-organization O O B-organization I-organization I-organization'), ([651, 442, 394, 527, 853, 189, 127, 302, 1436], [3, 13, 13, 13, 9, 19, 19, 7, 17], 9, '富 龙 集 团 董 事 长 王 诚', 'B-company I-company I-company I-company B-position I-position I-position B-name I-name'), ([13, 399, 541, 732, 12, 532, 614, 380, 42], [2, 12, 12, 12, 12, 0, 0, 0, 0], 9, '《 红 楼 梦 》 标 准 版 ：', 'B-book I-book I-book I-book I-book O O O O'), ([928, 140, 482, 165, 41, 741, 1167, 812, 932], [6, 16, 16, 16, 16, 16, 16, 16, 16], 9, '哈 里 波 特 和 死 亡 圣 器', 'B-movie I-movie I-movie I-movie I-movie I-movie I-movie I-movie I-movie'), ([2169, 77, 397, 195, 679, 342, 16, 255, 211], [0, 6, 16, 0, 0, 0, 0, 0, 0], 9, '* 战 神 资 料 片 1 C D', 'O B-movie I-movie O O O O O O'), ([320, 575, 1546, 154, 840, 116, 2877, 3094, 7], [0, 10, 20, 20, 0, 0, 0, 0, 0], 9, '让 武 夷 山 闻 名 遐 迩 。', 'O B-scene I-scene I-scene O O O O O'), ([59, 162, 930, 167, 115, 64, 393, 956, 101], [0, 0, 9, 19, 19, 19, 0, 7, 17], 9, '本 报 摄 影 记 者 / 吴 军', 'O O B-position I-position I-position I-position O B-name I-name'), ([25, 75, 902, 64, 212, 32, 109, 109, 479], [1, 11, 0, 0, 7, 17, 17, 17, 17], 9, '上 海 读 者 m a r r y', 'B-address I-address O O B-name I-name I-name I-name I-name'), ([50, 64, 42, 1001, 426, 1019, 988, 2622, 371], [9, 19, 0, 7, 17, 17, 7, 17, 17], 9, '作 者 ： 夏 林 杰 严 钧 花', 'B-position I-position O B-name I-name I-name B-name I-name I-name'), ([275, 362, 54, 729, 1044, 129, 196, 257, 895], [1, 11, 11, 11, 11, 11, 11, 11, 11], 9, '广 州 市 绝 顶 高 手 网 吧', 'B-address I-address I-address I-address I-address I-address I-address I-address I-address'), ([571, 83, 1265, 563, 837, 442, 683, 154, 7], [0, 0, 0, 0, 10, 20, 20, 20, 0], 9, '即 可 遥 望 玉 龙 雪 山 。', 'O O O O B-scene I-scene I-scene I-scene O'), ([9, 17, 565, 875, 270, 86, 1238, 1122, 416], [0, 0, 8, 18, 0, 0, 8, 18, 18], 9, '0 2 切 沃 v s 锡 耶 纳', 'O O B-organization I-organization O O B-organization I-organization I-organization'), ([959, 132, 2285, 154, 989, 192, 448, 231, 394], [10, 20, 20, 20, 20, 0, 0, 0, 0], 9, '【 北 邙 山 】 道 具 收 集', 'B-scene I-scene I-scene I-scene I-scene O O O O'), ([1160, 189, 52, 224, 191, 93, 243, 988, 377], [0, 0, 9, 19, 19, 0, 0, 0, 0], 9, '涉 事 公 务 员 将 被 严 处', 'O O B-position I-position I-position O O O O'), ([34, 94, 121, 318, 229, 188, 522, 122, 7], [0, 0, 0, 7, 17, 0, 0, 0, 0], 9, '发 信 与 张 路 交 流 ） 。', 'O O O B-name I-name O O O O'), ([260, 666, 139, 709, 298, 113, 42, 128, 1776], [4, 14, 14, 14, 9, 19, 0, 0, 0], 9, '英 雄 联 盟 解 说 ： 小 苍', 'B-game I-game I-game I-game B-position I-position O O O'), ([526, 890, 6, 27, 146, 343, 849, 180, 37], [3, 13, 0, 0, 0, 0, 0, 0, 0], 9, '宝 钢 的 “ 两 种 输 法 ”', 'B-company I-company O O O O O O O'), ([417, 1066, 49, 162, 115, 64, 591, 2070, 1049], [2, 12, 12, 12, 9, 19, 7, 17, 17], 9, '证 券 时 报 记 者 黄 兆 隆', 'B-book I-book I-book I-book B-position I-position B-name I-name I-name'), ([728, 130, 1270, 1941, 394, 527, 403, 281, 1930], [3, 13, 13, 13, 13, 13, 0, 0, 0], 9, '百 胜 餐 饮 集 团 消 费 →', 'B-company I-company I-company I-company I-company I-company O O O'), ([417, 1066, 49, 162, 115, 64, 318, 1340, 1554], [2, 12, 12, 12, 9, 19, 7, 17, 17], 9, '证 券 时 报 记 者 张 若 斌', 'B-book I-book I-book I-book B-position I-position B-name I-name I-name'), ([206, 232, 477, 864, 158, 115, 703, 613, 718], [4, 14, 14, 14, 0, 0, 0, 0, 0], 9, '文 明 I V 全 记 录 套 装', 'B-game I-game I-game I-game O O O O O'), ([1984, 115, 64, 674, 855, 799, 1611, 2402, 609], [0, 9, 19, 7, 17, 17, 7, 17, 17], 9, '□ 记 者 何 丰 伦 孟 昭 丽', 'O B-position I-position B-name I-name I-name B-name I-name I-name'), ([25, 100, 61, 75, 226, 659, 1000, 789, 7], [0, 0, 0, 0, 0, 0, 10, 20, 0], 9, '上 天 下 海 玩 转 塞 班 。', 'O O O O O O B-scene I-scene O'), ([654, 291, 23, 199, 404, 1050, 302, 1057, 1350], [8, 18, 18, 18, 9, 19, 7, 17, 17], 9, '清 华 大 学 教 授 王 贵 祥', 'B-organization I-organization I-organization I-organization B-position I-position B-name I-name I-name'), ([417, 1066, 49, 162, 115, 64, 1738, 2121, 203], [2, 12, 12, 12, 9, 19, 7, 17, 17], 9, '证 券 时 报 记 者 桂 衍 民', 'B-book I-book I-book I-book B-position I-position B-name I-name I-name'), ([46, 214, 42, 20, 1020, 795, 331, 104, 230], [0, 0, 0, 1, 11, 11, 11, 11, 11], 9, '地 点 ： 国 贸 春 季 房 展', 'O O O B-address I-address I-address I-address I-address I-address'), ([206, 393, 118, 115, 64, 107, 134, 108], [0, 0, 0, 9, 19, 7, 17, 17], 8, '文 / 表 记 者 方 利 平', 'O O O B-position I-position B-name I-name I-name'), ([1767, 1049, 837, 932, 828, 773, 3259, 497], [7, 17, 0, 0, 0, 0, 0, 0], 8, '乾 隆 玉 器 独 占 鳌 头', 'B-name I-name O O O O O O'), ([308, 295, 247, 30, 153, 267, 22, 148], [0, 9, 19, 19, 0, 0, 0, 0], 8, '推 艺 术 家 看 重 人 品', 'O B-position I-position I-position O O O O'), ([208, 1760, 394, 527, 706, 102, 198, 1089], [3, 13, 13, 13, 9, 19, 19, 19], 8, '世 茂 集 团 助 理 总 裁', 'B-company I-company I-company I-company B-position I-position I-position I-position'), ([6, 570, 101, 325, 1567, 1059, 806, 7], [0, 5, 15, 0, 0, 0, 0, 0], 8, '的 陆 军 基 础 训 练 。', 'O B-government I-government O O O O O'), ([1164, 91, 92, 53, 115, 64, 113, 7], [7, 17, 17, 0, 9, 19, 0, 0], 8, '沈 美 成 对 记 者 说 。', 'B-name I-name I-name O B-position I-position O O'), ([985, 1088, 359, 18, 2192, 371, 1385, 7], [10, 20, 20, 0, 10, 20, 20, 0], 8, '宁 静 岛 、 芙 花 芬 。', 'B-scene I-scene I-scene O B-scene I-scene I-scene O'), ([9, 82, 647, 565, 270, 86, 299, 244], [0, 0, 8, 18, 0, 0, 8, 18], 8, '0 4 莱 切 v s 罗 马', 'O O B-organization I-organization O O B-organization I-organization'), ([1646, 2092, 1580, 769, 17, 381, 110, 42], [4, 14, 14, 14, 14, 0, 0, 0], 8, '狼 穴 尖 兵 2 专 区 ：', 'B-game I-game I-game I-game I-game O O O'), ([296, 375, 235, 338, 55, 174, 116, 42], [4, 14, 14, 14, 0, 0, 0, 0], 8, '魔 兽 争 霸 前 三 名 ：', 'B-game I-game I-game I-game O O O O'), ([2916, 59, 162, 115, 64, 183, 19, 26], [0, 0, 0, 9, 19, 7, 17, 17], 8, '⊙ 本 报 记 者 但 有 为', 'O O O B-position I-position B-name I-name I-name'), ([1065, 138, 355, 189, 626, 362, 957, 350], [0, 0, 0, 0, 10, 20, 20, 20], 8, '赏 心 乐 事 苏 州 庭 园', 'O O O O B-scene I-scene I-scene I-scene'), ([16, 29, 565, 875, 270, 86, 299, 244], [0, 0, 8, 18, 0, 0, 8, 18], 8, '1 3 切 沃 v s 罗 马', 'O O B-organization I-organization O O B-organization I-organization'), ([59, 175, 40, 261, 104, 1738, 1500, 122], [0, 0, 9, 19, 7, 17, 17, 0], 8, '本 期 主 持 房 桂 岭 ）', 'O O B-position I-position B-name I-name I-name O'), ([63, 114, 399, 420, 276, 2525, 355, 60], [0, 0, 8, 18, 18, 18, 18, 18], 8, '5 . 红 色 管 弦 乐 队', 'O O B-organization I-organization I-organization I-organization I-organization I-organization'), ([548, 116, 13, 600, 22, 726, 283, 12], [0, 0, 4, 14, 14, 14, 14, 14], 8, '又 名 《 真 人 快 打 》', 'O O B-game I-game I-game I-game I-game I-game'), ([200, 176, 235, 338, 55, 174, 116, 42], [4, 14, 14, 14, 0, 0, 0, 0], 8, '星 际 争 霸 前 三 名 ：', 'B-game I-game I-game I-game O O O O'), ([1207, 625, 496, 256, 190, 84, 62, 7], [4, 14, 14, 14, 0, 0, 0, 0], 8, '穿 越 火 线 等 分 部 。', 'B-game I-game I-game I-game O O O O'), ([13, 399, 541, 732, 12, 560, 210, 380], [4, 14, 14, 14, 14, 0, 0, 0], 8, '《 红 楼 梦 》 普 通 版', 'B-game I-game I-game I-game I-game O O O'), ([408, 397, 408, 22, 12, 76, 15, 7], [4, 14, 14, 14, 14, 0, 0, 0], 8, '半 神 半 人 》 之 中 。', 'B-game I-game I-game I-game I-game O O O')]
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
        # if index == 166:
        #     print("batch:", '\n', batch)
        #     # batch:
        #     #  [([312, 116, 42, 430, 861, 54, 399, 470, 791, 38], [2916, 59, 162, 115, 64, 1406, 600, 442, 2006, 2420], [296, 375, 235, 338, 29, 42, 1062, 1082, 302, 943], [279, 50, 148, 42, 13, 621, 397, 76, 1477, 12], [13, 399, 541, 732, 12, 658, 378, 502, 184, 380], [440, 319, 1634, 6, 721, 189, 12, 187, 17, 394], [36, 139, 69, 187, 10, 170, 547, 39, 20, 234], [107, 339, 2144, 166, 844, 87, 302, 732, 1032], [340, 196, 1126, 36, 197, 632, 73, 224, 352], [206, 393, 59, 819, 115, 64, 1864, 1373, 1373], [318, 512, 285, 105, 1125, 713, 376, 925, 122], [549, 91, 208, 272, 20, 176, 380, 12, 7], [514, 336, 537, 271, 198, 388, 749, 26, 807], [65, 630, 355, 465, 198, 57, 102, 299, 101], [1570, 128, 101, 328, 562, 202, 258, 88, 7], [240, 107, 576, 162, 115, 64, 995, 1141, 1794], [565, 875, 60, 270, 86, 1238, 1122, 416, 60], [834, 456, 358, 195, 219, 1486, 22, 318, 826], [29, 114, 1156, 1072, 270, 86, 221, 177, 620], [651, 442, 394, 527, 853, 189, 127, 302, 1436], [13, 399, 541, 732, 12, 532, 614, 380, 42], [928, 140, 482, 165, 41, 741, 1167, 812, 932], [2169, 77, 397, 195, 679, 342, 16, 255, 211], [320, 575, 1546, 154, 840, 116, 2877, 3094, 7], [59, 162, 930, 167, 115, 64, 393, 956, 101], [25, 75, 902, 64, 212, 32, 109, 109, 479], [50, 64, 42, 1001, 426, 1019, 988, 2622, 371], [275, 362, 54, 729, 1044, 129, 196, 257, 895], [571, 83, 1265, 563, 837, 442, 683, 154, 7], [9, 17, 565, 875, 270, 86, 1238, 1122, 416], [959, 132, 2285, 154, 989, 192, 448, 231, 394], [1160, 189, 52, 224, 191, 93, 243, 988, 377], [34, 94, 121, 318, 229, 188, 522, 122, 7], [260, 666, 139, 709, 298, 113, 42, 128, 1776], [526, 890, 6, 27, 146, 343, 849, 180, 37], [417, 1066, 49, 162, 115, 64, 591, 2070, 1049], [728, 130, 1270, 1941, 394, 527, 403, 281, 1930], [417, 1066, 49, 162, 115, 64, 318, 1340, 1554], [206, 232, 477, 864, 158, 115, 703, 613, 718], [1984, 115, 64, 674, 855, 799, 1611, 2402, 609], [25, 100, 61, 75, 226, 659, 1000, 789, 7], [654, 291, 23, 199, 404, 1050, 302, 1057, 1350], [417, 1066, 49, 162, 115, 64, 1738, 2121, 203], [46, 214, 42, 20, 1020, 795, 331, 104, 230], [206, 393, 118, 115, 64, 107, 134, 108], [1767, 1049, 837, 932, 828, 773, 3259, 497], [308, 295, 247, 30, 153, 267, 22, 148], [208, 1760, 394, 527, 706, 102, 198, 1089], [6, 570, 101, 325, 1567, 1059, 806, 7], [1164, 91, 92, 53, 115, 64, 113, 7], [985, 1088, 359, 18, 2192, 371, 1385, 7], [9, 82, 647, 565, 270, 86, 299, 244], [1646, 2092, 1580, 769, 17, 381, 110, 42], [296, 375, 235, 338, 55, 174, 116, 42], [2916, 59, 162, 115, 64, 183, 19, 26], [1065, 138, 355, 189, 626, 362, 957, 350], [16, 29, 565, 875, 270, 86, 299, 244], [59, 175, 40, 261, 104, 1738, 1500, 122], [63, 114, 399, 420, 276, 2525, 355, 60], [548, 116, 13, 600, 22, 726, 283, 12], [200, 176, 235, 338, 55, 174, 116, 42], [1207, 625, 496, 256, 190, 84, 62, 7], [13, 399, 541, 732, 12, 560, 210, 380], [408, 397, 408, 22, 12, 76, 15, 7]), ([0, 0, 0, 8, 18, 18, 18, 18, 18, 18], [0, 0, 0, 9, 19, 7, 17, 17, 9, 19], [4, 14, 14, 14, 14, 14, 14, 14, 14, 14], [0, 0, 0, 0, 4, 14, 14, 14, 14, 14], [2, 12, 12, 12, 12, 0, 0, 0, 0, 0], [2, 12, 12, 12, 12, 12, 12, 0, 0, 0], [8, 18, 0, 0, 0, 0, 0, 0, 0, 0], [7, 17, 17, 9, 19, 19, 7, 17, 17], [0, 0, 3, 13, 0, 0, 0, 0, 0], [0, 0, 0, 0, 9, 19, 7, 17, 17], [0, 0, 0, 0, 9, 19, 19, 19, 0], [4, 14, 14, 14, 14, 14, 14, 14, 0], [3, 13, 9, 19, 19, 19, 7, 17, 17], [3, 13, 13, 13, 9, 19, 19, 7, 17], [7, 17, 17, 0, 0, 3, 13, 13, 0], [2, 12, 12, 12, 9, 19, 7, 17, 17], [8, 18, 18, 0, 0, 8, 18, 18, 18], [3, 13, 13, 13, 9, 19, 19, 7, 17], [0, 0, 8, 18, 0, 0, 8, 18, 18], [3, 13, 13, 13, 9, 19, 19, 7, 17], [2, 12, 12, 12, 12, 0, 0, 0, 0], [6, 16, 16, 16, 16, 16, 16, 16, 16], [0, 6, 16, 0, 0, 0, 0, 0, 0], [0, 10, 20, 20, 0, 0, 0, 0, 0], [0, 0, 9, 19, 19, 19, 0, 7, 17], [1, 11, 0, 0, 7, 17, 17, 17, 17], [9, 19, 0, 7, 17, 17, 7, 17, 17], [1, 11, 11, 11, 11, 11, 11, 11, 11], [0, 0, 0, 0, 10, 20, 20, 20, 0], [0, 0, 8, 18, 0, 0, 8, 18, 18], [10, 20, 20, 20, 20, 0, 0, 0, 0], [0, 0, 9, 19, 19, 0, 0, 0, 0], [0, 0, 0, 7, 17, 0, 0, 0, 0], [4, 14, 14, 14, 9, 19, 0, 0, 0], [3, 13, 0, 0, 0, 0, 0, 0, 0], [2, 12, 12, 12, 9, 19, 7, 17, 17], [3, 13, 13, 13, 13, 13, 0, 0, 0], [2, 12, 12, 12, 9, 19, 7, 17, 17], [4, 14, 14, 14, 0, 0, 0, 0, 0], [0, 9, 19, 7, 17, 17, 7, 17, 17], [0, 0, 0, 0, 0, 0, 10, 20, 0], [8, 18, 18, 18, 9, 19, 7, 17, 17], [2, 12, 12, 12, 9, 19, 7, 17, 17], [0, 0, 0, 1, 11, 11, 11, 11, 11], [0, 0, 0, 9, 19, 7, 17, 17], [7, 17, 0, 0, 0, 0, 0, 0], [0, 9, 19, 19, 0, 0, 0, 0], [3, 13, 13, 13, 9, 19, 19, 19], [0, 5, 15, 0, 0, 0, 0, 0], [7, 17, 17, 0, 9, 19, 0, 0], [10, 20, 20, 0, 10, 20, 20, 0], [0, 0, 8, 18, 0, 0, 8, 18], [4, 14, 14, 14, 14, 0, 0, 0], [4, 14, 14, 14, 0, 0, 0, 0], [0, 0, 0, 9, 19, 7, 17, 17], [0, 0, 0, 0, 10, 20, 20, 20], [0, 0, 8, 18, 0, 0, 8, 18], [0, 0, 9, 19, 7, 17, 17, 0], [0, 0, 8, 18, 18, 18, 18, 18], [0, 0, 4, 14, 14, 14, 14, 14], [4, 14, 14, 14, 0, 0, 0, 0], [4, 14, 14, 14, 0, 0, 0, 0], [4, 14, 14, 14, 14, 0, 0, 0], [4, 14, 14, 14, 14, 0, 0, 0]), (10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8), ('户 名 ： 深 圳 市 红 十 字 会', '⊙ 本 报 记 者 唐 真 龙 邹 靓', '魔 兽 争 霸 3 ： 冰 封 王 座', '原 作 品 ： 《 雷 神 之 锤 》', '《 红 楼 梦 》 初 回 限 定 版', '列 那 狐 的 故 事 》 第 2 集', '银 联 卡 第 一 次 走 出 国 门', '方 传 柳 实 习 生 王 梦 菲', '接 手 荷 银 亚 洲 业 务 ？', '文 / 本 刊 记 者 冯 娜 娜', '张 女 士 （ 私 企 老 板 ）', '完 美 世 界 国 际 版 》 。', '易 兰 创 意 总 监 余 为 群', '新 浪 乐 居 总 经 理 罗 军', '彭 小 军 告 诉 c b n 。', '东 方 早 报 记 者 刘 秀 浩', '切 沃 队 v s 锡 耶 纳 队', '汇 石 投 资 合 伙 人 张 剑', '3 . 拜 仁 v s 布 加 勒', '富 龙 集 团 董 事 长 王 诚', '《 红 楼 梦 》 标 准 版 ：', '哈 里 波 特 和 死 亡 圣 器', '* 战 神 资 料 片 1 C D', '让 武 夷 山 闻 名 遐 迩 。', '本 报 摄 影 记 者 / 吴 军', '上 海 读 者 m a r r y', '作 者 ： 夏 林 杰 严 钧 花', '广 州 市 绝 顶 高 手 网 吧', '即 可 遥 望 玉 龙 雪 山 。', '0 2 切 沃 v s 锡 耶 纳', '【 北 邙 山 】 道 具 收 集', '涉 事 公 务 员 将 被 严 处', '发 信 与 张 路 交 流 ） 。', '英 雄 联 盟 解 说 ： 小 苍', '宝 钢 的 “ 两 种 输 法 ”', '证 券 时 报 记 者 黄 兆 隆', '百 胜 餐 饮 集 团 消 费 →', '证 券 时 报 记 者 张 若 斌', '文 明 I V 全 记 录 套 装', '□ 记 者 何 丰 伦 孟 昭 丽', '上 天 下 海 玩 转 塞 班 。', '清 华 大 学 教 授 王 贵 祥', '证 券 时 报 记 者 桂 衍 民', '地 点 ： 国 贸 春 季 房 展', '文 / 表 记 者 方 利 平', '乾 隆 玉 器 独 占 鳌 头', '推 艺 术 家 看 重 人 品', '世 茂 集 团 助 理 总 裁', '的 陆 军 基 础 训 练 。', '沈 美 成 对 记 者 说 。', '宁 静 岛 、 芙 花 芬 。', '0 4 莱 切 v s 罗 马', '狼 穴 尖 兵 2 专 区 ：', '魔 兽 争 霸 前 三 名 ：', '⊙ 本 报 记 者 但 有 为', '赏 心 乐 事 苏 州 庭 园', '1 3 切 沃 v s 罗 马', '本 期 主 持 房 桂 岭 ）', '5 . 红 色 管 弦 乐 队', '又 名 《 真 人 快 打 》', '星 际 争 霸 前 三 名 ：', '穿 越 火 线 等 分 部 。', '《 红 楼 梦 》 普 通 版', '半 神 半 人 》 之 中 。'), ('O O O B-organization I-organization I-organization I-organization I-organization I-organization I-organization', 'O O O B-position I-position B-name I-name I-name B-position I-position', 'B-game I-game I-game I-game I-game I-game I-game I-game I-game I-game', 'O O O O B-game I-game I-game I-game I-game I-game', 'B-book I-book I-book I-book I-book O O O O O', 'B-book I-book I-book I-book I-book I-book I-book O O O', 'B-organization I-organization O O O O O O O O', 'B-name I-name I-name B-position I-position I-position B-name I-name I-name', 'O O B-company I-company O O O O O', 'O O O O B-position I-position B-name I-name I-name', 'O O O O B-position I-position I-position I-position O', 'B-game I-game I-game I-game I-game I-game I-game I-game O', 'B-company I-company B-position I-position I-position I-position B-name I-name I-name', 'B-company I-company I-company I-company B-position I-position I-position B-name I-name', 'B-name I-name I-name O O B-company I-company I-company O', 'B-book I-book I-book I-book B-position I-position B-name I-name I-name', 'B-organization I-organization I-organization O O B-organization I-organization I-organization I-organization', 'B-company I-company I-company I-company B-position I-position I-position B-name I-name', 'O O B-organization I-organization O O B-organization I-organization I-organization', 'B-company I-company I-company I-company B-position I-position I-position B-name I-name', 'B-book I-book I-book I-book I-book O O O O', 'B-movie I-movie I-movie I-movie I-movie I-movie I-movie I-movie I-movie', 'O B-movie I-movie O O O O O O', 'O B-scene I-scene I-scene O O O O O', 'O O B-position I-position I-position I-position O B-name I-name', 'B-address I-address O O B-name I-name I-name I-name I-name', 'B-position I-position O B-name I-name I-name B-name I-name I-name', 'B-address I-address I-address I-address I-address I-address I-address I-address I-address', 'O O O O B-scene I-scene I-scene I-scene O', 'O O B-organization I-organization O O B-organization I-organization I-organization', 'B-scene I-scene I-scene I-scene I-scene O O O O', 'O O B-position I-position I-position O O O O', 'O O O B-name I-name O O O O', 'B-game I-game I-game I-game B-position I-position O O O', 'B-company I-company O O O O O O O', 'B-book I-book I-book I-book B-position I-position B-name I-name I-name', 'B-company I-company I-company I-company I-company I-company O O O', 'B-book I-book I-book I-book B-position I-position B-name I-name I-name', 'B-game I-game I-game I-game O O O O O', 'O B-position I-position B-name I-name I-name B-name I-name I-name', 'O O O O O O B-scene I-scene O', 'B-organization I-organization I-organization I-organization B-position I-position B-name I-name I-name', 'B-book I-book I-book I-book B-position I-position B-name I-name I-name', 'O O O B-address I-address I-address I-address I-address I-address', 'O O O B-position I-position B-name I-name I-name', 'B-name I-name O O O O O O', 'O B-position I-position I-position O O O O', 'B-company I-company I-company I-company B-position I-position I-position I-position', 'O B-government I-government O O O O O', 'B-name I-name I-name O B-position I-position O O', 'B-scene I-scene I-scene O B-scene I-scene I-scene O', 'O O B-organization I-organization O O B-organization I-organization', 'B-game I-game I-game I-game I-game O O O', 'B-game I-game I-game I-game O O O O', 'O O O B-position I-position B-name I-name I-name', 'O O O O B-scene I-scene I-scene I-scene', 'O O B-organization I-organization O O B-organization I-organization', 'O O B-position I-position B-name I-name I-name O', 'O O B-organization I-organization I-organization I-organization I-organization I-organization', 'O O B-game I-game I-game I-game I-game I-game', 'B-game I-game I-game I-game O O O O', 'B-game I-game I-game I-game O O O O', 'B-game I-game I-game I-game I-game O O O', 'B-game I-game I-game I-game I-game O O O')]
        #     print("np.shape(batch):", '\n', np.shape(batch))
        #     # np.shape(batch): 
        #     #  (5, 64)
        # lens为当前batch中所有句子长度构成的列表
        lens = [len(x) for x in batch[0]]
        # if index == 166:
        #     print(lens)
        #     # [10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        batch, orig_idx = self.sort_all(batch, lens)
        # if index == 166:
        #     print("batch after sort_all():", '\n', batch)
        # # batch after sort_all():    
        # #  [[[36, 139, 69, 187, 10, 170, 547, 39, 20, 234], [440, 319, 1634, 6, 721, 189, 12, 187, 17, 394], [13, 399, 541, 732, 12, 658, 378, 502, 184, 380], [279, 50, 148, 42, 13, 621, 397, 76, 1477, 12], [296, 375, 235, 338, 29, 42, 1062, 1082, 302, 943], [2916, 59, 162, 115, 64, 1406, 600, 442, 2006, 2420], [312, 116, 42, 430, 861, 54, 399, 470, 791, 38], [46, 214, 42, 20, 1020, 795, 331, 104, 230], [417, 1066, 49, 162, 115, 64, 1738, 2121, 203], [654, 291, 23, 199, 404, 1050, 302, 1057, 1350], [25, 100, 61, 75, 226, 659, 1000, 789, 7], [1984, 115, 64, 674, 855, 799, 1611, 2402, 609], [206, 232, 477, 864, 158, 115, 703, 613, 718], [417, 1066, 49, 162, 115, 64, 318, 1340, 1554], [728, 130, 1270, 1941, 394, 527, 403, 281, 1930], [417, 1066, 49, 162, 115, 64, 591, 2070, 1049], [526, 890, 6, 27, 146, 343, 849, 180, 37], [260, 666, 139, 709, 298, 113, 42, 128, 1776], [34, 94, 121, 318, 229, 188, 522, 122, 7], [1160, 189, 52, 224, 191, 93, 243, 988, 377], [959, 132, 2285, 154, 989, 192, 448, 231, 394], [9, 17, 565, 875, 270, 86, 1238, 1122, 416], [571, 83, 1265, 563, 837, 442, 683, 154, 7], [275, 362, 54, 729, 1044, 129, 196, 257, 895], [50, 64, 42, 1001, 426, 1019, 988, 2622, 371], [25, 75, 902, 64, 212, 32, 109, 109, 479], [59, 162, 930, 167, 115, 64, 393, 956, 101], [320, 575, 1546, 154, 840, 116, 2877, 3094, 7], [2169, 77, 397, 195, 679, 342, 16, 255, 211], [928, 140, 482, 165, 41, 741, 1167, 812, 932], [13, 399, 541, 732, 12, 532, 614, 380, 42], [651, 442, 394, 527, 853, 189, 127, 302, 1436], [29, 114, 1156, 1072, 270, 86, 221, 177, 620], [834, 456, 358, 195, 219, 1486, 22, 318, 826], [565, 875, 60, 270, 86, 1238, 1122, 416, 60], [240, 107, 576, 162, 115, 64, 995, 1141, 1794], [1570, 128, 101, 328, 562, 202, 258, 88, 7], [65, 630, 355, 465, 198, 57, 102, 299, 101], [514, 336, 537, 271, 198, 388, 749, 26, 807], [549, 91, 208, 272, 20, 176, 380, 12, 7], [318, 512, 285, 105, 1125, 713, 376, 925, 122], [206, 393, 59, 819, 115, 64, 1864, 1373, 1373], [340, 196, 1126, 36, 197, 632, 73, 224, 352], [107, 339, 2144, 166, 844, 87, 302, 732, 1032], [408, 397, 408, 22, 12, 76, 15, 7], [13, 399, 541, 732, 12, 560, 210, 380], [1207, 625, 496, 256, 190, 84, 62, 7], [200, 176, 235, 338, 55, 174, 116, 42], [548, 116, 13, 600, 22, 726, 283, 12], [63, 114, 399, 420, 276, 2525, 355, 60], [59, 175, 40, 261, 104, 1738, 1500, 122], [16, 29, 565, 875, 270, 86, 299, 244], [1065, 138, 355, 189, 626, 362, 957, 350], [2916, 59, 162, 115, 64, 183, 19, 26], [296, 375, 235, 338, 55, 174, 116, 42], [1646, 2092, 1580, 769, 17, 381, 110, 42], [9, 82, 647, 565, 270, 86, 299, 244], [985, 1088, 359, 18, 2192, 371, 1385, 7], [1164, 91, 92, 53, 115, 64, 113, 7], [6, 570, 101, 325, 1567, 1059, 806, 7], [208, 1760, 394, 527, 706, 102, 198, 1089], [308, 295, 247, 30, 153, 267, 22, 148], [1767, 1049, 837, 932, 828, 773, 3259, 497], [206, 393, 118, 115, 64, 107, 134, 108]], [[8, 18, 0, 0, 0, 0, 0, 0, 0, 0], [2, 12, 12, 12, 12, 12, 12, 0, 0, 0], [2, 12, 12, 12, 12, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 14, 14, 14, 14, 14], [4, 14, 14, 14, 14, 14, 14, 14, 14, 14], [0, 0, 0, 9, 19, 7, 17, 17, 9, 19], [0, 0, 0, 8, 18, 18, 18, 18, 18, 18], [0, 0, 0, 1, 11, 11, 11, 11, 11], [2, 12, 12, 12, 9, 19, 7, 17, 17], [8, 18, 18, 18, 9, 19, 7, 17, 17], [0, 0, 0, 0, 0, 0, 10, 20, 0], [0, 9, 19, 7, 17, 17, 7, 17, 17], [4, 14, 14, 14, 0, 0, 0, 0, 0], [2, 12, 12, 12, 9, 19, 7, 17, 17], [3, 13, 13, 13, 13, 13, 0, 0, 0], [2, 12, 12, 12, 9, 19, 7, 17, 17], [3, 13, 0, 0, 0, 0, 0, 0, 0], [4, 14, 14, 14, 9, 19, 0, 0, 0], [0, 0, 0, 7, 17, 0, 0, 0, 0], [0, 0, 9, 19, 19, 0, 0, 0, 0], [10, 20, 20, 20, 20, 0, 0, 0, 0], [0, 0, 8, 18, 0, 0, 8, 18, 18], [0, 0, 0, 0, 10, 20, 20, 20, 0], [1, 11, 11, 11, 11, 11, 11, 11, 11], [9, 19, 0, 7, 17, 17, 7, 17, 17], [1, 11, 0, 0, 7, 17, 17, 17, 17], [0, 0, 9, 19, 19, 19, 0, 7, 17], [0, 10, 20, 20, 0, 0, 0, 0, 0], [0, 6, 16, 0, 0, 0, 0, 0, 0], [6, 16, 16, 16, 16, 16, 16, 16, 16], [2, 12, 12, 12, 12, 0, 0, 0, 0], [3, 13, 13, 13, 9, 19, 19, 7, 17], [0, 0, 8, 18, 0, 0, 8, 18, 18], [3, 13, 13, 13, 9, 19, 19, 7, 17], [8, 18, 18, 0, 0, 8, 18, 18, 18], [2, 12, 12, 12, 9, 19, 7, 17, 17], [7, 17, 17, 0, 0, 3, 13, 13, 0], [3, 13, 13, 13, 9, 19, 19, 7, 17], [3, 13, 9, 19, 19, 19, 7, 17, 17], [4, 14, 14, 14, 14, 14, 14, 14, 0], [0, 0, 0, 0, 9, 19, 19, 19, 0], [0, 0, 0, 0, 9, 19, 7, 17, 17], [0, 0, 3, 13, 0, 0, 0, 0, 0], [7, 17, 17, 9, 19, 19, 7, 17, 17], [4, 14, 14, 14, 14, 0, 0, 0], [4, 14, 14, 14, 14, 0, 0, 0], [4, 14, 14, 14, 0, 0, 0, 0], [4, 14, 14, 14, 0, 0, 0, 0], [0, 0, 4, 14, 14, 14, 14, 14], [0, 0, 8, 18, 18, 18, 18, 18], [0, 0, 9, 19, 7, 17, 17, 0], [0, 0, 8, 18, 0, 0, 8, 18], [0, 0, 0, 0, 10, 20, 20, 20], [0, 0, 0, 9, 19, 7, 17, 17], [4, 14, 14, 14, 0, 0, 0, 0], [4, 14, 14, 14, 14, 0, 0, 0], [0, 0, 8, 18, 0, 0, 8, 18], [10, 20, 20, 0, 10, 20, 20, 0], [7, 17, 17, 0, 9, 19, 0, 0], [0, 5, 15, 0, 0, 0, 0, 0], [3, 13, 13, 13, 9, 19, 19, 19], [0, 9, 19, 19, 0, 0, 0, 0], [7, 17, 0, 0, 0, 0, 0, 0], [0, 0, 0, 9, 19, 7, 17, 17]], [10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], ['银 联 卡 第 一 次 走 出 国 门', '列 那 狐 的 故 事 》 第 2 集', '《 红 楼 梦 》 初 回 限 定 版', '原 作 品 ： 《 雷 神 之 锤 》', '魔 兽 争 霸 3 ： 冰 封 王 座', '⊙ 本 报 记 者 唐 真 龙 邹 靓', '户 名 ： 深 圳 市 红 十 字 会', '地 点 ： 国 贸 春 季 房 展', '证 券 时 报 记 者 桂 衍 民', '清 华 大 学 教 授 王 贵 祥', '上 天 下 海 玩 转 塞 班 。', '□ 记 者 何 丰 伦 孟 昭 丽', '文 明 I V 全 记 录 套 装', '证 券 时 报 记 者 张 若 斌', '百 胜 餐 饮 集 团 消 费 →', '证 券 时 报 记 者 黄 兆 隆', '宝 钢 的 “ 两 种 输 法 ”', '英 雄 联 盟 解 说 ： 小 苍', '发 信 与 张 路 交 流 ） 。', '涉 事 公 务 员 将 被 严 处', '【 北 邙 山 】 道 具 收 集', '0 2 切 沃 v s 锡 耶 纳', '即 可 遥 望 玉 龙 雪 山 。', '广 州 市 绝 顶 高 手 网 吧', '作 者 ： 夏 林 杰 严 钧 花', '上 海 读 者 m a r r y', '本 报 摄 影 记 者 / 吴 军', '让 武 夷 山 闻 名 遐 迩 。', '* 战 神 资 料 片 1 C D', '哈 里 波 特 和 死 亡 圣 器', '《 红 楼 梦 》 标 准 版 ：', '富 龙 集 团 董 事 长 王 诚', '3 . 拜 仁 v s 布 加 勒', '汇 石 投 资 合 伙 人 张 剑', '切 沃 队 v s 锡 耶 纳 队', '东 方 早 报 记 者 刘 秀 浩', '彭 小 军 告 诉 c b n 。', '新 浪 乐 居 总 经 理 罗 军', '易 兰 创 意 总 监 余 为 群', '完 美 世 界 国 际 版 》 。', '张 女 士 （ 私 企 老 板 ）', '文 / 本 刊 记 者 冯 娜 娜', '接 手 荷 银 亚 洲 业 务 ？', '方 传 柳 实 习 生 王 梦 菲', '半 神 半 人 》 之 中 。', '《 红 楼 梦 》 普 通 版', '穿 越 火 线 等 分 部 。', '星 际 争 霸 前 三 名 ：', '又 名 《 真 人 快 打 》', '5 . 红 色 管 弦 乐 队', '本 期 主 持 房 桂 岭 ）', '1 3 切 沃 v s 罗 马', '赏 心 乐 事 苏 州 庭 园', '⊙ 本 报 记 者 但 有 为', '魔 兽 争 霸 前 三 名 ：', '狼 穴 尖 兵 2 专 区 ：', '0 4 莱 切 v s 罗 马', '宁 静 岛 、 芙 花 芬 。', '沈 美 成 对 记 者 说 。', '的 陆 军 基 础 训 练 。', '世 茂 集 团 助 理 总 裁', '推 艺 术 家 看 重 人 品', '乾 隆 玉 器 独 占 鳌 头', '文 / 表 记 者 方 利 平'], ['B-organization I-organization O O O O O O O O', 'B-book I-book I-book I-book I-book I-book I-book O O O', 'B-book I-book I-book I-book I-book O O O O O', 'O O O O B-game I-game I-game I-game I-game I-game', 'B-game I-game I-game I-game I-game I-game I-game I-game I-game I-game', 'O O O B-position I-position B-name I-name I-name B-position I-position', 'O O O B-organization I-organization I-organization I-organization I-organization I-organization I-organization', 'O O O B-address I-address I-address I-address I-address I-address', 'B-book I-book I-book I-book B-position I-position B-name I-name I-name', 'B-organization I-organization I-organization I-organization B-position I-position B-name I-name I-name', 'O O O O O O B-scene I-scene O', 'O B-position I-position B-name I-name I-name B-name I-name I-name', 'B-game I-game I-game I-game O O O O O', 'B-book I-book I-book I-book B-position I-position B-name I-name I-name', 'B-company I-company I-company I-company I-company I-company O O O', 'B-book I-book I-book I-book B-position I-position B-name I-name I-name', 'B-company I-company O O O O O O O', 'B-game I-game I-game I-game B-position I-position O O O', 'O O O B-name I-name O O O O', 'O O B-position I-position I-position O O O O', 'B-scene I-scene I-scene I-scene I-scene O O O O', 'O O B-organization I-organization O O B-organization I-organization I-organization', 'O O O O B-scene I-scene I-scene I-scene O', 'B-address I-address I-address I-address I-address I-address I-address I-address I-address', 'B-position I-position O B-name I-name I-name B-name I-name I-name', 'B-address I-address O O B-name I-name I-name I-name I-name', 'O O B-position I-position I-position I-position O B-name I-name', 'O B-scene I-scene I-scene O O O O O', 'O B-movie I-movie O O O O O O', 'B-movie I-movie I-movie I-movie I-movie I-movie I-movie I-movie I-movie', 'B-book I-book I-book I-book I-book O O O O', 'B-company I-company I-company I-company B-position I-position I-position B-name I-name', 'O O B-organization I-organization O O B-organization I-organization I-organization', 'B-company I-company I-company I-company B-position I-position I-position B-name I-name', 'B-organization I-organization I-organization O O B-organization I-organization I-organization I-organization', 'B-book I-book I-book I-book B-position I-position B-name I-name I-name', 'B-name I-name I-name O O B-company I-company I-company O', 'B-company I-company I-company I-company B-position I-position I-position B-name I-name', 'B-company I-company B-position I-position I-position I-position B-name I-name I-name', 'B-game I-game I-game I-game I-game I-game I-game I-game O', 'O O O O B-position I-position I-position I-position O', 'O O O O B-position I-position B-name I-name I-name', 'O O B-company I-company O O O O O', 'B-name I-name I-name B-position I-position I-position B-name I-name I-name', 'B-game I-game I-game I-game I-game O O O', 'B-game I-game I-game I-game I-game O O O', 'B-game I-game I-game I-game O O O O', 'B-game I-game I-game I-game O O O O', 'O O B-game I-game I-game I-game I-game I-game', 'O O B-organization I-organization I-organization I-organization I-organization I-organization', 'O O B-position I-position B-name I-name I-name O', 'O O B-organization I-organization O O B-organization I-organization', 'O O O O B-scene I-scene I-scene I-scene', 'O O O B-position I-position B-name I-name I-name', 'B-game I-game I-game I-game O O O O', 'B-game I-game I-game I-game I-game O O O', 'O O B-organization I-organization O O B-organization I-organization', 'B-scene I-scene I-scene O B-scene I-scene I-scene O', 'B-name I-name I-name O B-position I-position O O', 'O B-government I-government O O O O O', 'B-company I-company I-company I-company B-position I-position I-position I-position', 'O B-position I-position I-position O O O O', 'B-name I-name O O O O O O', 'O O O B-position I-position B-name I-name I-name']]
        
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
        # # input_ids:
        # #  tensor([[  36,  139,   69,  187,   10,  170,  547,   39,   20,  234],
        # #         [ 440,  319, 1634,    6,  721,  189,   12,  187,   17,  394],
        # #         [  13,  399,  541,  732,   12,  658,  378,  502,  184,  380],
        # #         [ 279,   50,  148,   42,   13,  621,  397,   76, 1477,   12],
        # #         [ 296,  375,  235,  338,   29,   42, 1062, 1082,  302,  943],
        # #         [2916,   59,  162,  115,   64, 1406,  600,  442, 2006, 2420],
        # #         [ 312,  116,   42,  430,  861,   54,  399,  470,  791,   38],
        # #         [  46,  214,   42,   20, 1020,  795,  331,  104,  230,    0],
        # #         [ 417, 1066,   49,  162,  115,   64, 1738, 2121,  203,    0],
        # #         [ 654,  291,   23,  199,  404, 1050,  302, 1057, 1350,    0],
        # #         [  25,  100,   61,   75,  226,  659, 1000,  789,    7,    0],
        # #         [1984,  115,   64,  674,  855,  799, 1611, 2402,  609,    0],
        # #         [ 206,  232,  477,  864,  158,  115,  703,  613,  718,    0],
        # #         [ 417, 1066,   49,  162,  115,   64,  318, 1340, 1554,    0],
        # #         [ 728,  130, 1270, 1941,  394,  527,  403,  281, 1930,    0],
        # #         [ 417, 1066,   49,  162,  115,   64,  591, 2070, 1049,    0],
        # #         [ 526,  890,    6,   27,  146,  343,  849,  180,   37,    0],
        # #         [ 260,  666,  139,  709,  298,  113,   42,  128, 1776,    0],
        # #         [  34,   94,  121,  318,  229,  188,  522,  122,    7,    0],
        # #         [1160,  189,   52,  224,  191,   93,  243,  988,  377,    0],
        # #         [ 959,  132, 2285,  154,  989,  192,  448,  231,  394,    0],
        # #         [   9,   17,  565,  875,  270,   86, 1238, 1122,  416,    0],
        # #         [ 571,   83, 1265,  563,  837,  442,  683,  154,    7,    0],
        # #         [ 275,  362,   54,  729, 1044,  129,  196,  257,  895,    0],
        # #         [  50,   64,   42, 1001,  426, 1019,  988, 2622,  371,    0],
        # #         [  25,   75,  902,   64,  212,   32,  109,  109,  479,    0],
        # #         [  59,  162,  930,  167,  115,   64,  393,  956,  101,    0],
        # #         [ 320,  575, 1546,  154,  840,  116, 2877, 3094,    7,    0],
        # #         [2169,   77,  397,  195,  679,  342,   16,  255,  211,    0],
        # #         [ 928,  140,  482,  165,   41,  741, 1167,  812,  932,    0],
        # #         [  13,  399,  541,  732,   12,  532,  614,  380,   42,    0],
        # #         [ 651,  442,  394,  527,  853,  189,  127,  302, 1436,    0],
        # #         [  29,  114, 1156, 1072,  270,   86,  221,  177,  620,    0],
        # #         [ 834,  456,  358,  195,  219, 1486,   22,  318,  826,    0],
        # #         [ 565,  875,   60,  270,   86, 1238, 1122,  416,   60,    0],
        # #         [ 240,  107,  576,  162,  115,   64,  995, 1141, 1794,    0],
        # #         [1570,  128,  101,  328,  562,  202,  258,   88,    7,    0],
        # #         [  65,  630,  355,  465,  198,   57,  102,  299,  101,    0],
        # #         [ 514,  336,  537,  271,  198,  388,  749,   26,  807,    0],
        # #         [ 549,   91,  208,  272,   20,  176,  380,   12,    7,    0],
        # #         [ 318,  512,  285,  105, 1125,  713,  376,  925,  122,    0],
        # #         [ 206,  393,   59,  819,  115,   64, 1864, 1373, 1373,    0],
        # #         [ 340,  196, 1126,   36,  197,  632,   73,  224,  352,    0],
        # #         [ 107,  339, 2144,  166,  844,   87,  302,  732, 1032,    0],
        # #         [ 408,  397,  408,   22,   12,   76,   15,    7,    0,    0],
        # #         [  13,  399,  541,  732,   12,  560,  210,  380,    0,    0],
        # #         [1207,  625,  496,  256,  190,   84,   62,    7,    0,    0],
        # #         [ 200,  176,  235,  338,   55,  174,  116,   42,    0,    0],
        # #         [ 548,  116,   13,  600,   22,  726,  283,   12,    0,    0],
        # #         [  63,  114,  399,  420,  276, 2525,  355,   60,    0,    0],
        # #         [  59,  175,   40,  261,  104, 1738, 1500,  122,    0,    0],
        # #         [  16,   29,  565,  875,  270,   86,  299,  244,    0,    0],
        # #         [1065,  138,  355,  189,  626,  362,  957,  350,    0,    0],
        # #         [2916,   59,  162,  115,   64,  183,   19,   26,    0,    0],
        # #         [ 296,  375,  235,  338,   55,  174,  116,   42,    0,    0],
        # #         [1646, 2092, 1580,  769,   17,  381,  110,   42,    0,    0],
        # #         [   9,   82,  647,  565,  270,   86,  299,  244,    0,    0],
        # #         [ 985, 1088,  359,   18, 2192,  371, 1385,    7,    0,    0],
        # #         [1164,   91,   92,   53,  115,   64,  113,    7,    0,    0],
        # #         [   6,  570,  101,  325, 1567, 1059,  806,    7,    0,    0],
        # #         [ 208, 1760,  394,  527,  706,  102,  198, 1089,    0,    0],
        # #         [ 308,  295,  247,   30,  153,  267,   22,  148,    0,    0],
        # #         [1767, 1049,  837,  932,  828,  773, 3259,  497,    0,    0],
        # #         [ 206,  393,  118,  115,   64,  107,  134,  108,    0,    0]])
        # # input_mask: 
        # #  tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
        # # label_ids: 
        # #  tensor([[ 8, 18,  0,  0,  0,  0,  0,  0,  0,  0],
        # #         [ 2, 12, 12, 12, 12, 12, 12,  0,  0,  0],
        # #         [ 2, 12, 12, 12, 12,  0,  0,  0,  0,  0],
        # #         [ 0,  0,  0,  0,  4, 14, 14, 14, 14, 14],
        # #         [ 4, 14, 14, 14, 14, 14, 14, 14, 14, 14],
        # #         [ 0,  0,  0,  9, 19,  7, 17, 17,  9, 19],
        # #         [ 0,  0,  0,  8, 18, 18, 18, 18, 18, 18],
        # #         [ 0,  0,  0,  1, 11, 11, 11, 11, 11,  0],
        # #         [ 2, 12, 12, 12,  9, 19,  7, 17, 17,  0],
        # #         [ 8, 18, 18, 18,  9, 19,  7, 17, 17,  0],
        # #         [ 0,  0,  0,  0,  0,  0, 10, 20,  0,  0],
        # #         [ 0,  9, 19,  7, 17, 17,  7, 17, 17,  0],
        # #         [ 4, 14, 14, 14,  0,  0,  0,  0,  0,  0],
        # #         [ 2, 12, 12, 12,  9, 19,  7, 17, 17,  0],
        # #         [ 3, 13, 13, 13, 13, 13,  0,  0,  0,  0],
        # #         [ 2, 12, 12, 12,  9, 19,  7, 17, 17,  0],
        # #         [ 3, 13,  0,  0,  0,  0,  0,  0,  0,  0],
        # #         [ 4, 14, 14, 14,  9, 19,  0,  0,  0,  0],
        # #         [ 0,  0,  0,  7, 17,  0,  0,  0,  0,  0],
        # #         [ 0,  0,  9, 19, 19,  0,  0,  0,  0,  0],
        # #         [10, 20, 20, 20, 20,  0,  0,  0,  0,  0],
        # #         [ 0,  0,  8, 18,  0,  0,  8, 18, 18,  0],
        # #         [ 0,  0,  0,  0, 10, 20, 20, 20,  0,  0],
        # #         [ 1, 11, 11, 11, 11, 11, 11, 11, 11,  0],
        # #         [ 9, 19,  0,  7, 17, 17,  7, 17, 17,  0],
        # #         [ 1, 11,  0,  0,  7, 17, 17, 17, 17,  0],
        # #         [ 0,  0,  9, 19, 19, 19,  0,  7, 17,  0],
        # #         [ 0, 10, 20, 20,  0,  0,  0,  0,  0,  0],
        # #         [ 0,  6, 16,  0,  0,  0,  0,  0,  0,  0],
        # #         [ 6, 16, 16, 16, 16, 16, 16, 16, 16,  0],
        # #         [ 2, 12, 12, 12, 12,  0,  0,  0,  0,  0],
        # #         [ 3, 13, 13, 13,  9, 19, 19,  7, 17,  0],
        # #         [ 0,  0,  8, 18,  0,  0,  8, 18, 18,  0],
        # #         [ 3, 13, 13, 13,  9, 19, 19,  7, 17,  0],
        # #         [ 8, 18, 18,  0,  0,  8, 18, 18, 18,  0],
        # #         [ 2, 12, 12, 12,  9, 19,  7, 17, 17,  0],
        # #         [ 7, 17, 17,  0,  0,  3, 13, 13,  0,  0],
        # #         [ 3, 13, 13, 13,  9, 19, 19,  7, 17,  0],
        # #         [ 3, 13,  9, 19, 19, 19,  7, 17, 17,  0],
        # #         [ 4, 14, 14, 14, 14, 14, 14, 14,  0,  0],
        # #         [ 0,  0,  0,  0,  9, 19, 19, 19,  0,  0],
        # #         [ 0,  0,  0,  0,  9, 19,  7, 17, 17,  0],
        # #         [ 0,  0,  3, 13,  0,  0,  0,  0,  0,  0],
        # #         [ 7, 17, 17,  9, 19, 19,  7, 17, 17,  0],
        # #         [ 4, 14, 14, 14, 14,  0,  0,  0,  0,  0],
        # #         [ 4, 14, 14, 14, 14,  0,  0,  0,  0,  0],
        # #         [ 4, 14, 14, 14,  0,  0,  0,  0,  0,  0],
        # #         [ 4, 14, 14, 14,  0,  0,  0,  0,  0,  0],
        # #         [ 0,  0,  4, 14, 14, 14, 14, 14,  0,  0],
        # #         [ 0,  0,  8, 18, 18, 18, 18, 18,  0,  0],
        # #         [ 0,  0,  9, 19,  7, 17, 17,  0,  0,  0],
        # #         [ 0,  0,  8, 18,  0,  0,  8, 18,  0,  0],
        # #         [ 0,  0,  0,  0, 10, 20, 20, 20,  0,  0],
        # #         [ 0,  0,  0,  9, 19,  7, 17, 17,  0,  0],
        # #         [ 4, 14, 14, 14,  0,  0,  0,  0,  0,  0],
        # #         [ 4, 14, 14, 14, 14,  0,  0,  0,  0,  0],
        # #         [ 0,  0,  8, 18,  0,  0,  8, 18,  0,  0],
        # #         [10, 20, 20,  0, 10, 20, 20,  0,  0,  0],
        # #         [ 7, 17, 17,  0,  9, 19,  0,  0,  0,  0],
        # #         [ 0,  5, 15,  0,  0,  0,  0,  0,  0,  0],
        # #         [ 3, 13, 13, 13,  9, 19, 19, 19,  0,  0],
        # #         [ 0,  9, 19, 19,  0,  0,  0,  0,  0,  0],
        # #         [ 7, 17,  0,  0,  0,  0,  0,  0,  0,  0],
        # #         [ 0,  0,  0,  9, 19,  7, 17, 17,  0,  0]])
        # # input_lens: 
        # #  [10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
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