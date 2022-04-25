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
            input_ids, input_mask, input_tags, input_lens = batch
            # 由于按长度降序排列的基础上分批，一轮中的
            # 第1个batch，input_ids, input_mask, input_tags的维度都是torch.Size([64, 50])
            # 第166个batch，input_ids, input_mask, input_tags的维度都是torch.Size([64, 10])
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)
            # step为166时，features维度torch.Size([64, 10, 33]) # 该批64个句子，每个句子包含10个词，每个词都有33个分类的值
            # loss就是一个数值，例如tensor(8.9871, device='cuda:0', grad_fn=<MeanBackward0>) 
            # loss的计算用到CRF的原理，最终调用的是crf._calculate_loss_old()，详细分析见https://www.jianshu.com/p/566c6faace64
            # 计算loss的时候，也是要用到真实标签y的，因为设计最小化loss目的是最大化似然概率p(y|X)，这是一个由softmax定义的正确预测在所有预测中的概率
            features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            pbar(step=step, info={'loss': loss.item()})
            train_loss.update(loss.item(), n=1)
        print(" ")
        # 该epoch各batches的平均loss
        train_log = {'loss': train_loss.avg}
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        # 在一个epoch的结尾，调用evaluate()进行验证
        # evaluate() returns result, class_info
        # result例如{'eval_loss': 22.0380, 'eval_acc': 0.4548, 'eval_recall': 0.3223, 'eval_f1': 0.3772}
        # class_info例如{'address': {Acc: 0.2897, Recall: 0.1126, F1: 0.1622}, 
                        # 'position': {Acc: 0.704, Recall: 0.4065, F1: 0.5154},
                        # ...}
        eval_log, class_info = evaluate(args,model,processor)
        logs = dict(train_log, **eval_log)
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        logger.info(show_info)
        # 有关lr调整的函数
        scheduler.epoch_step(logs['eval_f1'], epoch)
        if logs['eval_f1'] > best_f1:
            logger.info(f"\nEpoch {epoch}: eval_f1 improved from {best_f1} to {logs['eval_f1']}")
            logger.info("save model to disk.")
            best_f1 = logs['eval_f1']
            if isinstance(model, nn.DataParallel):
                model_stat_dict = model.module.state_dict()
            else:
                model_stat_dict = model.state_dict()
            # 保存模型
            state = {'epoch': epoch, 'arch': args.arch, 'state_dict': model_stat_dict}
            model_path = args.output_dir / 'best-model.bin'
            torch.save(state, str(model_path))
            print("Eval Entity Score: ")
            for key, value in class_info.items():
                info = f"Subject: {key} - Acc: {value['acc']} - Recall: {value['recall']} - F1: {value['f1']}"
                logger.info(info)

def evaluate(args,model,processor):
    # 对验证集，load_and_cache_examples()返回的是data_processor.py中get_dev_examples()的结果
    # get_dev_examples()主要是将原始dev.json中的数据转换成[{'id': xx, 
                                                            # 'context': xx, 
                                                            # 'tag': xx, (表达成B-xx, I-xx)
                                                            # 'raw_context': xx}
                                                            # ...]
                                                            # 的形式
    eval_dataset = load_and_cache_examples(args,processor, data_type='dev')
    # eval_dataloader是DatasetLoader类的可迭代对象
    # args.batch_size即一个batch包含dev.json中的行数，默认为32，可在train.sh里面修改
    eval_dataloader = DatasetLoader(data=eval_dataset, batch_size=args.batch_size,
                                 shuffle=False, seed=args.seed, sort=False,
                                 vocab=processor.vocab, label2id=args.label2id)
    # 验证过程的进度条，例如[Evaluating] 21/21 [==============================] 840.9ms/step 
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    # 评价指标metric，是SeqEntityScore类对象
    metric = SeqEntityScore(args.id2label,markup=args.markup)
    # AverageMeter computes and stores the average and current value
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
            # tags是预测标签，_obtain_labels()预测过程用到维特比解码
            tags, _ = model.crf._obtain_labels(features, args.id2label, input_lens)
            # 真实标签
            input_tags = input_tags.cpu().numpy()
            # batch size为64的情况下，每一个batch的target是一个包含64个元素的列表，每个元素为一个array，表示该句子的真实标签
            # target就是input_tags处理了一下后得到的
            target = [input_[:len_] for input_, len_ in zip(input_tags, input_lens)]
            # 预测标签与真实标签存入metric
            metric.update(pred_paths=tags, label_paths=target)
            pbar(step=step)
    print(" ")
    # metric.result() returns {'acc': precision, 'recall': recall, 'f1': f1}, class_info
    eval_info, class_info = metric.result()
    eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
    result = {'eval_loss': eval_loss.avg}
    result = dict(result, **eval_info)
    return result, class_info

def predict(args,model,processor):
    # model_path = args.output_dir / 'best-model.bin'
    # model = load_model(model, model_path=str(model_path))
    # 处理测试数据
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
    # 进度条
    pbar = ProgressBar(n_total=len(test_data))
    # 所有句子的结果，放在一个大列表results中
    results = []
    # 测试不分批，直接逐句测试
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
            # _obtain_labels函数会调用到_viterbi_decode()
            tags, _ = model.crf._obtain_labels(features, args.id2label, input_lens)
        label_entities = get_entities(tags[0], args.id2label)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join(tags[0])
        json_d['entities'] = label_entities
        # 将当前句子的结果字典json_d加入列表results中
        results.append(json_d)
        pbar(step=step)
    print(" ")
    # 预测结果文件output_predic_file与提交文件output_submit_file的保存路径
    output_predic_file = str(args.output_dir / "test_prediction.json")
    output_submit_file = str(args.output_dir / "test_submit.json")
    # 向预测结果文件output_predic_file中写入结果
    with open(output_predic_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    # 将test.json中每一行都作为一个元素加入列表test_text中
    test_text = []
    with open(str(args.data_dir / 'test.json'), 'r') as fr:
        for line in fr:
            test_text.append(json.loads(line))
    # 对列表test_text中待预测的句子，与列表results中预测结果进行相应的处理
    # 得到符合提交格式的结果，放入列表test_submit中，进而存到路径output_submit_file
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
    parser.add_argument('--epochs',default=50,type=int)
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
        model_path = '/home/user/xiongdengrui/cluener/CLUENER2020/bilstm_crf_pytorch/outputs/20220424162108/best-model.bin'
        model = load_model(model, model_path=str(model_path))
        evaluate(args,model,processor)
    if args.do_predict:
        model_path = '/home/user/xiongdengrui/cluener/CLUENER2020/bilstm_crf_pytorch/outputs/20220424162108/best-model.bin'
        model = load_model(model, model_path=str(model_path))
        predict(args,model,processor)
    # **************************************************************

if __name__ == "__main__":
    main()
