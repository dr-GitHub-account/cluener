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
            if step == 1:
                print('\n', 'input_ids[0]', '\n', input_ids[0], '\n',
                      'input_ids[1]', '\n', input_ids[1])
                print('\n', 'input_tags[0]', '\n', input_tags[0], '\n',
                      'input_tags[1]', '\n', input_tags[1])
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)
            features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pbar(step=step, info={'loss': loss.item()})
            train_loss.update(loss.item(), n=1)
        print(" ")
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
