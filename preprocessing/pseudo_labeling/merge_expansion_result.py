from argparse import ArgumentParser
import torch

if __name__=='__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--pts', nargs="+", required=True)
    parser.add_argument('--output', required=True)
    
    args = parser.parse_args()

    print(args)
    
    exp_dict = {'qid_to_embs':{}, 'qid_to_weights':{}, 'qid_to_tokens':{}}
    for pt in args.pts:
        _dict = torch.load(pt)
        
        assert len(_dict['qid_to_embs']) == len(_dict['qid_to_weights']) == len(_dict['qid_to_tokens'])
        print(f'#> Load {pt} ({len(_dict["qid_to_embs"])})', end=" ")
        # Dict{'qid_to_embs':qid_to_embs, 'qid_to_weights': qid_to_weights, 'qid_to_tokens': qid_to_tokens}
        
        # exp_dict['qid_to_embs'] = dict(exp_dict['qid_to_embs'], **_dict['qid_to_embs'])
        # exp_dict['qid_to_weights'] = dict(exp_dict['qid_to_weights'], **_dict['qid_to_weights'])
        # exp_dict['qid_to_tokens'] = dict(exp_dict['qid_to_tokens'], **_dict['qid_to_tokens'])
        exp_dict['qid_to_embs'].update(_dict['qid_to_embs'])
        exp_dict['qid_to_weights'].update(_dict['qid_to_weights'])
        exp_dict['qid_to_tokens'].update(_dict['qid_to_tokens'])

        print(f'==> {len(exp_dict["qid_to_embs"])}')

    torch.save(exp_dict, args.output)
    print(f'\n\t\t{args.output}')