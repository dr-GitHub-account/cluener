import json
import torch
import argparse
import torch.nn as nn
import numpy as np
from torch import optim
import config
from model import NERModel
from dataset_loader import DatasetLoader
from progressbar import ProgressBar
from ner_metrics import SeqEntityScore
from data_processor import CluenerProcessor
from lr_scheduler import ReduceLROnPlateau
from utils_ner import get_entities
from common import (init_logger,
                    logger,
                    json_to_text,
                    load_model,
                    AverageMeter,
                    seed_everything)

# 调用：train(args,model,processor)
def train(args,model,processor):
    # 训练集
    # 对训练集，load_and_cache_examples()返回的是data_processor.py中get_train_examples()的结果
    # get_train_examples()主要是将原始train.json中的数据转换成[{'id': xx, 
                                                            # 'context': xx, 
                                                            # 'tag': xx, (表达成B-xx, I-xx)
                                                            # 'raw_context': xx}
                                                            # ...]
                                                            # 的形式
    train_dataset = load_and_cache_examples(args, processor, data_type='train')
    # train_loader是DatasetLoader类的可迭代对象
    # args.batch_size即一个batch包含train.json中的行数，默认为32，可在train.sh里面修改
    train_loader = DatasetLoader(data=train_dataset, batch_size=args.batch_size,
                                 shuffle=False, seed=args.seed, sort=True,
                                 vocab = processor.vocab,label2id = args.label2id)
    # 所有可训练参数
    parameters = [p for p in model.parameters() if p.requires_grad]
    # 优化器
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    # Reduce learning rate when a metric has stopped improving.
    # Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. 
    # This scheduler reads a metrics quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                  verbose=1, epsilon=1e-4, cooldown=0, min_lr=0, eps=1e-8)
    best_f1 = 0
    # 开始遍历每个epoch
    for epoch in range(1, 1 + args.epochs):
        print(f"Epoch {epoch}/{args.epochs}")
        # 训练时打印的进度条，例如：[Training] 168/168 [==============================] 75.4ms/step  loss: 1.0134  
        pbar = ProgressBar(n_total=len(train_loader), desc='Training')
        # AverageMeter computes and stores the average and current value
        train_loss = AverageMeter()
        # set model to train mode
        model.train()
        assert model.training
        # 开始遍历每个batch
        for step, batch in enumerate(train_loader):
            # class DatasetLoader的__getitem__(self, index)返回(input_ids, input_mask, label_ids, input_lens)
            # 具体一个batch中，input_ids, input_mask, input_tags, input_lens见下面的输出
            input_ids, input_mask, input_tags, input_lens = batch
            # if step == 1:
            #     print('\n', 'input_ids', '\n', input_ids, '\n', np.shape(input_ids), '\n',
            #           'input_mask', '\n', input_mask, '\n', np.shape(input_mask), '\n',
            #           'input_tags', '\n', input_tags, '\n', np.shape(input_tags), '\n',
            #           'input_lens', '\n', input_lens, '\n', np.shape(input_lens))
            #     # input_ids 
            #     # tensor([[  36,   14,   71,  ...,  215,   49,    5],
            #     #         [ 373,   30,   79,  ...,  990,  433,    7],
            #     #         [ 237,   28,    6,  ...,   76,  267,    7],
            #     #         ...,
            #     #         [ 189,  385,    6,  ...,  101,   32,  280],
            #     #         [  93,  168,  583,  ...,  737,  515,    5],
            #     #         [1783, 1811,  506,  ...,  106,  126,  280]]) 
            #     # torch.Size([64, 50]) 
            #     # input_mask 
            #     # tensor([[1, 1, 1,  ..., 1, 1, 1],
            #     #         [1, 1, 1,  ..., 1, 1, 1],
            #     #         [1, 1, 1,  ..., 1, 1, 1],
            #     #         ...,
            #     #         [1, 1, 1,  ..., 1, 1, 1],
            #     #         [1, 1, 1,  ..., 1, 1, 1],
            #     #         [1, 1, 1,  ..., 1, 1, 1]]) 
            #     # torch.Size([64, 50]) 
            #     # input_tags 
            #     # tensor([[ 0,  0,  0,  ...,  0,  0,  0],
            #     #         [ 9, 19,  0,  ...,  0,  0,  0],
            #     #         [ 0,  0,  0,  ...,  0,  0,  0],
            #     #         ...,
            #     #         [ 0,  0,  0,  ..., 15,  0,  0],
            #     #         [ 0,  0,  0,  ...,  0,  0,  0],
            #     #         [10, 20, 20,  ...,  0,  0,  0]]) 
            #     # torch.Size([64, 50]) 
            #     # input_lens 
            #     # [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50] 
            #     # (64,)
            # if step == 166:
            #     print('\n', 'input_ids', '\n', input_ids, '\n', np.shape(input_ids), '\n',
            #           'input_mask', '\n', input_mask, '\n', np.shape(input_mask), '\n',
            #           'input_tags', '\n', input_tags, '\n', np.shape(input_tags), '\n',
            #           'input_lens', '\n', input_lens, '\n', np.shape(input_lens))
            # 由于按长度降序排列的基础上分批，一轮中的
            # 第1个batch，input_ids, input_mask, input_tags的维度都是torch.Size([64, 50])
            # 第166个batch，input_ids, input_mask, input_tags的维度都是torch.Size([64, 10])
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)
            # step为166时，features维度torch.Size([64, 10, 33]) # 该批64个句子，每个句子包含10个词，每个词都有33个分类的值
            # loss就是一个数值，例如tensor(8.9871, device='cuda:0', grad_fn=<MeanBackward0>) 
            # loss的计算用到CRF的原理，最终调用的是crf._calculate_loss_old()，详细分析见https://www.jianshu.com/p/566c6faace64
            if step == 1:
                features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
                break
            # features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
#             if step == 166:
#                 print('\n', "features:", features, '\n', np.shape(features))
#                 print("loss:", loss, '\n', np.shape(loss))
# #                  features: tensor([[[ 2.6459e+00, -4.4215e-01, -1.9535e+00,  ..., -4.0014e+00,
#     #           -3.4442e-01,  6.9062e-02],
#     #          [ 1.4129e+00, -2.2106e+00, -3.0510e+00,  ..., -4.2903e+00,
#     #           -4.6297e-01,  2.2698e-01],
#     #          [ 3.4138e+00, -2.6776e+00, -3.5085e+00,  ..., -5.8684e+00,
#     #           -6.4630e-01, -3.2454e-02],
#     #          ...,
#     #          [ 8.0761e+00,  1.9655e+00, -2.0424e+00,  ..., -5.8393e+00,
#     #           -5.6531e-01, -4.5488e-01],
#     #          [ 5.8832e+00,  2.4703e+00, -2.1461e+00,  ..., -6.4830e+00,
#     #           -5.0603e-01, -6.0104e-01],
#     #          [ 4.5271e+00, -7.9451e-01, -3.1045e+00,  ..., -5.7839e+00,
#     #           -5.5761e-02, -8.6590e-01]],

#     #         [[ 6.1318e-01, -2.5042e+00,  1.5772e+00,  ..., -3.1423e+00,
#     #           -1.8321e-02,  1.1504e+00],
#     #          [ 9.0930e-01, -3.3000e+00,  1.4157e+00,  ..., -3.2337e+00,
#     #           -1.8074e-02,  1.2435e+00],
#     #          [ 1.5743e+00, -3.2882e+00,  1.0156e+00,  ..., -3.6433e+00,
#     #            1.8683e-01,  1.0125e+00],
#     #          ...,
#     #          [ 6.5505e+00, -1.6395e+00, -1.9536e+00,  ..., -6.6046e+00,
#     #           -2.9757e-01,  8.0718e-01],
#     #          [ 8.1896e+00, -1.6847e+00, -2.1852e+00,  ..., -6.8262e+00,
#     #           -3.3698e-01,  2.6191e-01],
#     #          [ 5.8177e+00,  1.2469e-01, -2.7436e+00,  ..., -6.9567e+00,
#     #           -5.0716e-01,  9.6944e-02]],

#     #         [[ 7.4268e-01, -2.6285e+00,  5.5740e+00,  ..., -2.2732e+00,
#     #            3.2179e-01, -1.1364e+00],
#     #          [-1.1681e+00, -4.3445e+00, -6.5679e-01,  ..., -2.8339e+00,
#     #            4.1109e-01, -3.5601e-01],
#     #          [ 7.5065e-01, -5.1111e+00, -3.1458e+00,  ..., -3.8628e+00,
#     #            1.1751e-01,  1.3915e-01],
#     #          ...,
#     #          [ 7.7705e+00, -3.0391e+00, -3.2330e+00,  ..., -6.6410e+00,
#     #           -3.8742e-01,  5.7752e-01],
#     #          [ 7.7844e+00, -2.0402e+00, -2.7205e+00,  ..., -6.7492e+00,
#     #           -2.4519e-01,  5.7911e-01],
#     #          [ 6.9569e+00, -1.2678e+00, -1.4939e+00,  ..., -6.4389e+00,
#     #           -4.6978e-01,  3.3389e-01]],

#     #         ...,

#     #         [[ 6.8296e+00, -3.8270e-01,  2.9954e+00,  ..., -5.8579e+00,
#     #            1.5678e-01, -8.1627e-01],
#     #          [ 3.4616e+00, -8.7779e-02,  3.0976e+00,  ..., -4.5443e+00,
#     #           -1.3139e-01, -5.5757e-01],
#     #          [ 7.0825e-01, -1.1624e+00,  1.3783e+00,  ..., -4.2287e+00,
#     #            8.4069e-03, -3.3586e-01],
#     #          ...,
#     #          [ 1.0317e+01, -2.0642e+00, -1.5293e+00,  ..., -7.8535e+00,
#     #           -5.2397e-02,  1.1710e-01],
#     #          [ 8.2443e+00,  1.0014e+00, -7.2722e-01,  ..., -8.4937e+00,
#     #           -4.2934e-01, -2.0850e-02],
#     #          [ 7.6935e+00, -5.9680e-01, -2.1351e+00,  ..., -8.5967e+00,
#     #           -3.8235e-01,  1.7272e-02]],

#     #         [[ 2.5192e+00,  1.8870e+00,  4.6041e-01,  ..., -5.2421e+00,
#     #           -1.3803e+00, -6.4533e-01],
#     #          [ 3.3240e+00,  1.8385e+00, -6.3419e-01,  ..., -6.3506e+00,
#     #           -9.9950e-01, -4.3637e-02],
#     #          [ 3.7435e+00,  9.1744e-01, -7.1899e-01,  ..., -6.8895e+00,
#     #           -1.1168e+00, -1.4139e-01],
#     #          ...,
#     #          [ 5.8671e+00, -1.4043e+00, -2.0876e+00,  ..., -8.3154e+00,
#     #           -5.6581e-01, -3.2507e-01],
#     #          [ 6.3212e+00, -1.7019e+00, -2.8040e+00,  ..., -8.8277e+00,
#     #           -4.2596e-01, -3.9155e-01],
#     #          [ 5.9585e+00, -1.8938e+00, -3.4605e+00,  ..., -8.3822e+00,
#     #           -3.5375e-01, -3.4236e-01]],

#     #         [[ 2.7880e+00,  1.4274e+00,  3.1136e+00,  ..., -6.1672e+00,
#     #           -6.0084e-01, -2.9435e-01],
#     #          [ 6.3901e+00, -5.0615e-01,  1.7425e+00,  ..., -8.0054e+00,
#     #           -5.4850e-01, -5.5964e-01],
#     #          [ 6.6285e+00, -1.8432e-02,  1.7194e+00,  ..., -7.6677e+00,
#     #           -5.1384e-01, -1.0702e-01],
#     #          ...,
#     #          [ 7.2458e+00, -1.4230e+00, -1.4955e+00,  ..., -8.2561e+00,
#     #           -4.1321e-01, -1.3946e-01],
#     #          [ 6.6179e+00,  5.7376e-01, -1.9511e+00,  ..., -8.7059e+00,
#     #           -6.2238e-01, -3.8805e-01],
#     #          [ 5.8160e+00,  4.0310e-01, -2.6053e+00,  ..., -8.4230e+00,
#     #           -6.0084e-01, -5.0822e-01]]], device='cuda:0', grad_fn=<AddBackward0>) 
#                 #  torch.Size([64, 10, 33]) # 该批64个句子，每个句子包含10个词，每个词都有33个分类的值
                # # 33是label2id字典中含键值对的数目
#                 # loss: tensor(8.9871, device='cuda:0', grad_fn=<MeanBackward0>) 
#                 #  torch.Size([])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pbar(step=step, info={'loss': loss.item()})
            train_loss.update(loss.item(), n=1)
        print(" ")
        # 该epoch各batches的平均loss
        train_log = {'loss': train_loss.avg}
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        eval_log, class_info = evaluate(args,model,processor)
        logs = dict(train_log, **eval_log)
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        logger.info(show_info)
        scheduler.epoch_step(logs['eval_f1'], epoch)
        if logs['eval_f1'] > best_f1:
            logger.info(f"\nEpoch {epoch}: eval_f1 improved from {best_f1} to {logs['eval_f1']}")
            logger.info("save model to disk.")
            best_f1 = logs['eval_f1']
            if isinstance(model, nn.DataParallel):
                model_stat_dict = model.module.state_dict()
            else:
                model_stat_dict = model.state_dict()
            state = {'epoch': epoch, 'arch': args.arch, 'state_dict': model_stat_dict}
            model_path = args.output_dir / 'best-model.bin'
            torch.save(state, str(model_path))
            print("Eval Entity Score: ")
            for key, value in class_info.items():
                info = f"Subject: {key} - Acc: {value['acc']} - Recall: {value['recall']} - F1: {value['f1']}"
                logger.info(info)

def evaluate(args,model,processor):
    eval_dataset = load_and_cache_examples(args,processor, data_type='dev')
    eval_dataloader = DatasetLoader(data=eval_dataset, batch_size=args.batch_size,
                                 shuffle=False, seed=args.seed, sort=False,
                                 vocab=processor.vocab, label2id=args.label2id)
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    metric = SeqEntityScore(args.id2label,markup=args.markup)
    eval_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            input_ids, input_mask, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)
            features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
            eval_loss.update(val=loss.item(), n=input_ids.size(0))
            tags, _ = model.crf._obtain_labels(features, args.id2label, input_lens)
            input_tags = input_tags.cpu().numpy()
            target = [input_[:len_] for input_, len_ in zip(input_tags, input_lens)]
            metric.update(pred_paths=tags, label_paths=target)
            pbar(step=step)
    print(" ")
    eval_info, class_info = metric.result()
    eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
    result = {'eval_loss': eval_loss.avg}
    result = dict(result, **eval_info)
    return result, class_info

def predict(args,model,processor):
    # model_path = args.output_dir / 'best-model.bin'
    # model = load_model(model, model_path=str(model_path))
    test_data = []
    with open(str(args.data_dir / "test.json"), 'r') as f:
        idx = 0
        for line in f:
            json_d = {}
            line = json.loads(line.strip())
            text = line['text']
            words = list(text)
            labels = ['O'] * len(words)
            json_d['id'] = idx
            json_d['context'] = " ".join(words)
            json_d['tag'] = " ".join(labels)
            json_d['raw_context'] = "".join(words)
            idx += 1
            test_data.append(json_d)
    pbar = ProgressBar(n_total=len(test_data))
    results = []
    for step, line in enumerate(test_data):
        token_a = line['context'].split(" ")
        input_ids = [processor.vocab.to_index(w) for w in token_a]
        input_mask = [1] * len(token_a)
        input_lens = [len(token_a)]
        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            input_mask = torch.tensor([input_mask], dtype=torch.long)
            input_lens = torch.tensor([input_lens], dtype=torch.long)
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            # 参数input_tags=None，model.forward_loss返回的features是model.forward的结果
            # 即将bilstm层->Norm层结果通过一个nn.Linear(hidden_size * 2,len(label2id))层，得到维度为len(label2id)的分类结果
            features = model.forward_loss(input_ids, input_mask, input_lens, input_tags=None)
            # 得到tags序列中单个字的预测的标签，后续" ".join(tags[0])将其拼接为一整个序列的预测的标签
            tags, _ = model.crf._obtain_labels(features, args.id2label, input_lens)
        label_entities = get_entities(tags[0], args.id2label)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join(tags[0])
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step=step)
    print(" ")
    output_predic_file = str(args.output_dir / "test_prediction.json")
    output_submit_file = str(args.output_dir / "test_submit.json")
    with open(output_predic_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    test_text = []
    with open(str(args.data_dir / 'test.json'), 'r') as fr:
        for line in fr:
            test_text.append(json.loads(line))
    test_submit = []
    for x, y in zip(test_text, results):
        json_d = {}
        json_d['id'] = x['id']
        json_d['label'] = {}
        entities = y['entities']
        words = list(x['text'])
        if len(entities) != 0:
            for subject in entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                word = "".join(words[start:end + 1])
                if tag in json_d['label']:
                    if word in json_d['label'][tag]:
                        json_d['label'][tag][word].append([start, end])
                    else:
                        json_d['label'][tag][word] = [[start, end]]
                else:
                    json_d['label'][tag] = {}
                    json_d['label'][tag][word] = [[start, end]]
        test_submit.append(json_d)
    json_to_text(output_submit_file, test_submit)

def load_and_cache_examples(args,processor, data_type='train'):
    # Load data features from cache or dataset file
    cached_examples_file = args.data_dir / 'cached_crf-{}_{}_{}'.format(
        data_type,
        args.arch,
        str(args.task_name))
    if cached_examples_file.exists():
        logger.info("Loading features from cached file %s", cached_examples_file)
        examples = torch.load(cached_examples_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if data_type == 'train':
            examples = processor.get_train_examples()
        elif data_type == 'dev':
            examples = processor.get_dev_examples()
        logger.info("Saving features into cached file %s", cached_examples_file)
        torch.save(examples, str(cached_examples_file))
    return examples

def main():
    
    # *******************************args*******************************
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--do_train", default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')
    parser.add_argument("--do_predict", default=False, action='store_true')
    parser.add_argument('--markup', default='bios', type=str, choices=['bios', 'bio'])
    parser.add_argument("--arch",default='bilstm_crf',type=str)
    parser.add_argument('--learning_rate',default=0.001,type=float)
    parser.add_argument('--seed',default=1234,type=int)
    parser.add_argument('--gpu',default='0',type=str)
    # parser.add_argument('--epochs',default=50,type=int)
    parser.add_argument('--epochs',default=5,type=int)
    parser.add_argument('--batch_size',default=32,type=int)
    parser.add_argument('--embedding_size',default=128,type=int)
    parser.add_argument('--hidden_size',default=384,type=int)
    parser.add_argument("--grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--task_name", type=str, default='ner')
    parser.add_argument("--output_time", type=str, default='default')
    args = parser.parse_args()
    args.data_dir = config.data_dir
    if not config.output_dir.exists():
        # args.output_dir.mkdir()
        config.output_dir.mkdir()
    args.output_dir = config.output_dir / '{}'.format(args.output_time)
    if not args.output_dir.exists():
        args.output_dir.mkdir()
    # **************************************************************
 
        
    # *******************************log*******************************
    init_logger(log_file=str(args.output_dir / '{}-{}.log'.format(args.arch, args.task_name)))
    # **************************************************************
    
    
    # *******************************seed*******************************
    seed_everything(args.seed)
    # **************************************************************
    
    
    # *******************************device*******************************
    if args.gpu!='':
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")
    # **************************************************************
        
        
    # *******************************mapping between ids and labels*******************************
    # args.id2label and args.label2id are both dictionaries
    args.id2label = {i: label for i, label in enumerate(config.label2id)}
    args.label2id = config.label2id
    # **************************************************************
    
    
    # *******************************processor*******************************
    # processor is a <data_processor.CluenerProcessor object>
    processor = CluenerProcessor(data_dir=config.data_dir)
    # get_vocab() builds and saves processor.vocab to 'self.data_dir / 'vocab.pkl''
    # processor.vocab is a <vocabulary.Vocabulary object>
    processor.get_vocab()
    # **************************************************************
    
    
    # *******************************model*******************************
    # class init:
    # __init__(self, vocab_size, embedding_size,
            # hidden_size, label2id, device, drop_p = 0.1)
            
    # embedding_size = 128, hidden_size = 384, label2id is a dictionary
    # embedding_size是词向量的维度
    # hidden_size是LSTM中隐层的维度
    model = NERModel(vocab_size=len(processor.vocab), embedding_size=args.embedding_size,
                     hidden_size=args.hidden_size,device=args.device,label2id=args.label2id)
    # put model to device
    model.to(args.device)
    # **************************************************************
    
    
    # *******************************train/eval/predict*******************************
    if args.do_train:
        train(args,model,processor)
    if args.do_eval:
        # model_path = args.output_dir / 'best-model.bin'
        model_path = '/home/user/xiongdengrui/cluener/CLUENER2020/bilstm_crf_pytorch/outputs/20220415170122/best-model.bin'
        model = load_model(model, model_path=str(model_path))
        evaluate(args,model,processor)
    if args.do_predict:
        model_path = '/home/user/xiongdengrui/cluener/CLUENER2020/bilstm_crf_pytorch/outputs/20220415170122/best-model.bin'
        model = load_model(model, model_path=str(model_path))
        predict(args,model,processor)
    # **************************************************************

if __name__ == "__main__":
    main()