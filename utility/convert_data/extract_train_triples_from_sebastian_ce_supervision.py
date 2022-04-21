import argparse
import json
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')

    args = parser.parse_args()

    # Read sebastian's training triples with pseudo-labels from a cross-encoder or an ensemble of cross-encoders
    # the files are described in here: https://github.com/sebastian-hofstaetter/neural-ranking-kd#teacher-training-files-msmarco-passage
    # and are uploaded in here: https://zenodo.org/record/4068216
    # The input file format is: [pos psg score] \t [neg psg score] \t [qid] \t [pos pid] \t [neg pid]
    # The output file format is: JsonLine [qid, pos pid, neg pid]
    print(f'input : {args.input}')
    print(f'output: {args.output}')
    with open(args.input) as ifile, open(args.output, 'w') as ofile:
        
        for line_idx, line in enumerate(ifile):
            
            pos_score, neg_score, qid, pos_pid, neg_pid = line.strip().split('\t')
            
            outline = json.dumps([int(qid), int(pos_pid), int(neg_pid)])
            ofile.write(f'{outline}\n')
            
