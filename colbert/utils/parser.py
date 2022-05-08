import os
import copy
import faiss

from argparse import ArgumentParser

import colbert.utils.distributed as distributed
from colbert.utils.runs import Run
from colbert.utils.utils import print_message, timestamp, create_directory


class Arguments():
    def __init__(self, description):
        self.parser = ArgumentParser(description=description)
        self.checks = []

        self.add_argument('--root', dest='root', default='experiments')
        self.add_argument('--experiment', dest='experiment', default='dirty')
        self.add_argument('--run', dest='run', default=Run.name)

        self.add_argument('--local_rank', dest='rank', default=-1, type=int)

    def add_model_parameters(self):
        # Core Arguments
        self.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'])
        self.add_argument('--dim', dest='dim', default=128, type=int)
        self.add_argument('--query_maxlen', dest='query_maxlen', default=32, type=int)
        self.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int)

        # Filtering-related Arguments
        self.add_argument('--mask-punctuation', dest='mask_punctuation', default=False, action='store_true')

    def add_model_training_parameters(self):
        self.add_argument('--knowledge_distillation', action='store_true',
                        help="Whether to use knowledge distillation, \
                            where the student is ColBERT \
                            and the teacher can be either ColBERT using original query or ColBERT using expanded query.")
        self.add_argument('--teacher_checkpoint', type=str, help="Path to ColBERT teacher's checkpoint.")
        
        self.add_argument('--static_supervision', type=str, help="Path to the pre-computed pairwise pseudo-labels by a cross-encoder or an ensemble of cross-encoders.")
        self.add_argument('--dual_supervision_lambda', type=float, default=0.75, help="balancing factor for dual supervision; loss_static (KD from cross-encoder) + lambda * loss_dynamic (KD from collective feedback encoder).")

        
        self.add_argument('--kd_query_expansion', action='store_true', help="Whether to use query expansion for knowledge distillation from teacher.")
        
        self.add_argument('--kd_expansion_pt_list', type=str, help="Path to expansion.pt", nargs="+")
        self.add_argument('--kd_lambda_list', type=float, help="KD loss weight for each expansion.pt", nargs="+")
        
        self.add_argument('--kd_penalty', type=float, default=0.0, help="[DEPRECATED] penalty for labeled positive passage.")
        
        self.add_argument('--kd_objective', type=str, default="ce", choices=['ce', 'mse'], 
                            help="knowledge distillation objective: \
                                ``ce`` from TCT-ColBERT (https://aclanthology.org/2021.repl4nlp-1.17.pdf), \
                                and ``mse`` from TAS-Balanced (https://dl.acm.org/doi/pdf/10.1145/3404835.3462891) and cross-architecture KD (https://arxiv.org/pdf/2010.02666.pdf).")
        
        # for kd_objective==ce
        self.add_argument('--kd_temperature', type=float, default=0.25)
        # for kd_objective==mse
        self.add_argument('--kd_maxmargin', type=float, default=32.0)
                                



        # NOTE: Providing a checkpoint is one thing, --resume is another, --resume_optimizer is yet another.
        self.add_argument('--resume', dest='resume', default=False, action='store_true')
        self.add_argument('--resume_optimizer', dest='resume_optimizer', default=False, action='store_true')
        self.add_argument('--checkpoint', dest='checkpoint', default=None, required=False)

        self.add_argument('--lr', dest='lr', default=3e-06, type=float)
        self.add_argument('--maxsteps', dest='maxsteps', default=400000, type=int)
        self.add_argument('--bsize', dest='bsize', default=32, type=int)
        self.add_argument('--accum', dest='accumsteps', default=2, type=int)
        self.add_argument('--amp', dest='amp', default=False, action='store_true')

    def add_model_inference_parameters(self):
        self.add_argument('--checkpoint', dest='checkpoint', required=True)
        self.add_argument('--bsize', dest='bsize', default=128, type=int)
        self.add_argument('--amp', dest='amp', default=False, action='store_true')

    def add_training_input(self):
        self.add_argument('--triples', dest='triples', required=True)
        self.add_argument('--queries', dest='queries', default=None)
        self.add_argument('--collection', dest='collection', default=None)

        def check_training_input(args):
            assert (args.collection is None) == (args.queries is None), \
                "For training, both (or neither) --collection and --queries must be supplied." \
                "If neither is supplied, the --triples file must contain texts (not PIDs)."

        self.checks.append(check_training_input)

    def add_ranking_input(self):
        self.add_argument('--queries', dest='queries', default=None)
        self.add_argument('--collection', dest='collection', default=None)
        self.add_argument('--qrels', dest='qrels', default=None)

    def add_reranking_input(self):
        self.add_ranking_input()
        self.add_argument('--topk', dest='topK', required=True)
        self.add_argument('--shortcircuit', dest='shortcircuit', default=False, action='store_true')

    def add_indexing_input(self):
        self.add_argument('--collection', dest='collection', required=True)
        self.add_argument('--index_root', dest='index_root', required=True)
        self.add_argument('--index_name', dest='index_name', required=True)

    def add_index_use_input(self):
        self.add_argument('--index_root', dest='index_root', required=True)
        self.add_argument('--index_name', dest='index_name', required=True)
        self.add_argument('--partitions', dest='partitions', default=None, type=int)

    def add_retrieval_input(self):
        self.add_index_use_input()
        self.add_argument('--nprobe', dest='nprobe', default=10, type=int)
        self.add_argument('--retrieve_only', dest='retrieve_only', default=False, action='store_true')

    def add_argument(self, *args, **kw_args):
        return self.parser.add_argument(*args, **kw_args)

    def check_arguments(self, args):
        for check in self.checks:
            check(args)

    def parse(self):
        args = self.parser.parse_args()
        self.check_arguments(args)

        args.input_arguments = copy.deepcopy(args)

        args.nranks, args.distributed = distributed.init(args.rank)

        args.nthreads = int(max(os.cpu_count(), faiss.omp_get_max_threads()) * 0.8)
        args.nthreads = max(1, args.nthreads // args.nranks)

        if args.nranks > 1:
            print_message(f"#> Restricting number of threads for FAISS to {args.nthreads} per process",
                          condition=(args.rank == 0))
            faiss.omp_set_num_threads(args.nthreads)

        Run.init(args.rank, args.root, args.experiment, args.run)
        Run._log_args(args)
        Run.info(args.input_arguments.__dict__, '\n')

        return args
