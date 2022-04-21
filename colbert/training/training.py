import os
import random
import time
import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW
from colbert.utils.runs import Run
from colbert.utils.amp import MixedPrecisionManager

from colbert.training.lazy_batcher import LazyBatcher
from colbert.training.eager_batcher import EagerBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints

# LOG_STEP = 100 #?@ debugging
LOG_STEP = 5000 #!@ custom


def train(args):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    if args.distributed:
        torch.cuda.manual_seed_all(12345)

    assert args.similarity == 'l2'
    assert args.kd_objective == 'ce'

    #!@ custom
    # sanity check
    if args.kd_query_expansion:
        for kd_expansion_pt in args.kd_expansion_pt_list:
            assert os.path.exists(kd_expansion_pt)

        #!@ custom: hard-coded, averaging KD losses with equal weights.
        args.kd_lambda_list = [1 / len(args.kd_expansion_pt_list) for _ in range(len(args.kd_expansion_pt_list))]
    
    if args.distributed:
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)

    if args.lazy:
        reader = LazyBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)
    else:
        reader = EagerBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)

    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)
    
    #!@ custom
    # Knowledge Distillation
    if args.knowledge_distillation:
        # Load teacher's checkpoint, for warm-start
        assert args.teacher_checkpoint is not None
        print_message(f"#> Load teacher checkpoint from {args.teacher_checkpoint}.")
        _teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location='cpu')

        print_message(f'#> Instantiate teacher (ColBERT that uses full tokens).')
        teacher = ColBERT.from_pretrained('bert-base-uncased',
                                    query_maxlen=_teacher_checkpoint["arguments"]["query_maxlen"],
                                    doc_maxlen=_teacher_checkpoint["arguments"]["doc_maxlen"],
                                    dim=_teacher_checkpoint["arguments"]["dim"],
                                    similarity_metric=_teacher_checkpoint["arguments"]["similarity"],
                                    mask_punctuation=_teacher_checkpoint["arguments"]["mask_punctuation"],)
        teacher.load_state_dict(_teacher_checkpoint['model_state_dict'])

    if args.checkpoint is not None:

        # assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
        
        #!@ custom
        if not args.resume_optimizer:
            print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")
        else:
            print_message(f"#> Starting from checkpoint {args.checkpoint} along with the optimizer!")

        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        try:
            colbert.load_state_dict(checkpoint['model_state_dict'])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)
        



    if args.rank == 0:
        torch.distributed.barrier()

    colbert = colbert.to(DEVICE)
    colbert.train()

    if args.distributed:
        colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[args.rank],
                                                            output_device=args.rank,
                                                            find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()

    #!@ custom
    if args.checkpoint is not None:
        if args.resume_optimizer:
            print_message(f"#> Resume optimizer!")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer.zero_grad()
            print(optimizer)

    #!@ custom
    if args.knowledge_distillation:
            
        teacher = teacher.to(DEVICE)
        print_message(f'#> Set eval mode for teacher.')
        teacher.eval()

        if args.distributed:
            teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.rank],
                                                        output_device=args.rank,
                                                        find_unused_parameters=True)
    
    #?@ debugging
    # print(f'optimizer={optimizer}')
    
    amp = MixedPrecisionManager(args.amp)
    criterion = nn.CrossEntropyLoss()
    
    #!@ custom: in-batch negatives
    labels = torch.arange(args.bsize, dtype=torch.long, device=DEVICE)
    
    #!@ original: pair-wise negatives
    # labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE) 

    start_time = time.time()
    train_loss = 0.0

    start_batch_idx = 0

    if args.resume:
        assert args.checkpoint is not None
        start_batch_idx = checkpoint['batch']

        reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    for batch_idx, BatchSteps in zip(range(start_batch_idx, args.maxsteps), reader):
        this_batch_loss = 0.0

        for queries, passages, queries_exp, pairwise_scores in BatchSteps:
        
            with amp.context():
                
                #?@ debugging
                # print(f'queries[0]=\n\t{queries[0]} ({queries[0].shape})') # input_ids
                # print(f'queries[1]=\n\t{queries[1]} ({queries[1].shape})') # attention_mask
                # print('\n')
                # print(f'passages[0]=\n\t{passages[0]} ({passages[0].shape})') # input_ids
                # print(f'passages[1]=\n\t{passages[1]} ({passages[1].shape})') # attention_mask
                # print('\n')
                # print(f'queries_exp[0]=\n\t{queries_exp[0]} ({queries_exp[0].shape})') # qexp_embs
                # print(f'queries_exp[1]=\n\t{queries_exp[1]} ({queries_exp[1].shape})') # qexp_wts
                # print(f'queries_exp[0]=\n\t{queries_exp[0]} ({queries_exp[0][0].shape})') # qexp_embs
                # print(f'queries_exp[1]=\n\t{queries_exp[1]} ({queries_exp[1][0].shape})') # qexp_wts
                # print(f'training.py: train: exit');exit()

                #!@ custom: in-batch negatives
                scores = colbert(queries, passages)
                # scores: float tensor (bsize, (k+1) * bsize)
                scores = scores[0]

                #!@ original: pair-wise negatives
                # scores = colbert(queries, passages).view(2, -1).permute(1, 0)
                # scores: float tensor (bsize, 2)

                #?@ debugging
                # print(f'scores ({scores.shape})')
                # print(f'scores: \n\t{scores}')
                # print(f'training.py: train: exit');exit()

                #!@ custom
                if args.knowledge_distillation:
                    with torch.no_grad():
                        teacher_scores_list = teacher(queries, passages, Q_exp = queries_exp)
                    
                    loss = 0
                    for teacher_index, teacher_scores in enumerate(teacher_scores_list):
                        teacher_scores[torch.arange(len(teacher_scores)), torch.arange(len(teacher_scores))] = teacher_scores.max(dim=1).values
                        teacher_scores = teacher_scores / args.kd_temperature
                        soft_labels = torch.nn.functional.softmax(teacher_scores, dim=-1)
                        log_soft_labels = torch.nn.functional.log_softmax(teacher_scores, dim=-1)

                        _loss = soft_labels * (log_soft_labels - torch.nn.functional.log_softmax(scores, dim=-1))
                        _loss = _loss.sum(-1) 
                        # _loss: float tensor (bsize)
                        _loss = _loss.mean(0)

                        #?@ debugging
                        # print()
                        # print(f'scores=\n\t{scores} ({scores.size()})')
                        # print(f'soft_labels=\n\t{soft_labels} ({soft_labels.size()})')
                        # print(f'_loss={_loss.item()}')
                        # print(f'args.kd_lambda_list[teacher_index]={args.kd_lambda_list[teacher_index]}')
                        # print(f'training.py: train: exit');exit()
                        
                        loss = loss + _loss * args.kd_lambda_list[teacher_index]

                else:
                    loss = criterion(scores, labels[:scores.size(0)])
                
                if args.static_supervision and (pairwise_scores is not None):
                    static_p_scores, static_n_scores = pairwise_scores
                    static_p_scores = static_p_scores.to(loss.device) # (N)
                    static_n_scores = static_n_scores.to(loss.device) # (N, n_negatives)

                    static_scores = torch.cat((static_p_scores.unsqueeze(1), static_n_scores), dim=1)
                    # static_scores: float tensor (bsize, (n_negatives+1))
                    static_scores = static_scores / args.kd_temperature
                    static_soft_labels = torch.nn.functional.softmax(static_scores, dim=1)
                    static_log_soft_labels = torch.nn.functional.log_softmax(static_scores, dim=-1)

                    _bsize = scores.size(0)
                    _n_negatives = (scores.size(1) // _bsize) - 1
                    assert _bsize == static_scores.size(0)
                    assert _n_negatives == static_scores.size(1) - 1
                    pairwise_scores = torch.stack([
                        scores[bidx, torch.arange(bidx, scores.size(1), _bsize)]
                        for bidx in range(_bsize)
                    ], dim=0)
                    # pairwise_scores: float tensor (bsize, 1 + n_negatives)
                    assert tuple(pairwise_scores.size()) == tuple(static_soft_labels.size())
                    
                    static_kd_loss = static_soft_labels * (static_log_soft_labels - torch.nn.functional.log_softmax(pairwise_scores, dim=-1))
                    static_kd_loss = static_kd_loss.sum(1) # float tensor (bsize,)
                    static_kd_loss = static_kd_loss.mean(0)
                    
                    loss = static_kd_loss + loss * args.dual_supervision_lambda

                loss = loss / args.accumsteps

                #?@ debugging
                # print()
                # print(f'scores=\n\t{scores} ({scores.size()})')
                # print(f'soft_labels=\n\t{soft_labels} ({soft_labels.size()})')
                # print(f'loss={loss.item()}')
                # print(f'training.py: train: exit');exit()


            #!@ custom: comment
            # if args.rank < 1:
            #     print_progress(scores)

            amp.backward(loss)

            train_loss += loss.item()
            this_batch_loss += loss.item()

        amp.step(colbert, optimizer)

        if args.rank < 1:
            avg_loss = train_loss / (batch_idx+1)

            num_examples_seen = (batch_idx - start_batch_idx) * args.bsize * args.nranks
            elapsed = float(time.time() - start_time)

            # log_to_mlflow = (batch_idx % 20 == 0) #!@ original
            log_to_mlflow = ((batch_idx+1) % LOG_STEP == 0) #!@ custom

            Run.log_metric('train/avg_loss', avg_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            # Run.log_metric('train/batch_loss', this_batch_loss, step=batch_idx, log_to_mlflow=log_to_mlflow) #!@ custom
            Run.log_metric('train/examples', num_examples_seen, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/throughput', num_examples_seen / elapsed, step=batch_idx, log_to_mlflow=log_to_mlflow)

            #!@ original
            # print_message(batch_idx, avg_loss)
            #!@ custom
            if (batch_idx + 1) % LOG_STEP == 0:
                print_message(batch_idx+1, avg_loss)

            manage_checkpoints(args, colbert, optimizer, batch_idx+1)
