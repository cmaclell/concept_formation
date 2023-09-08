import csv
import gensim.downloader
import os

PRINT_ALL = False
PENALIZE_NONES = True

w2v = gensim.downloader.load('word2vec-google-news-300')

# files = [x for x in os.listdir('.') if 'csv' in x]
# files = ['word2vec_rocstories_out.csv']

# files = ['cobweb_freq_5_rocstories_out-v2.csv']
# outfile = "partial_out_cobweb_mi.csv"

files = ['cobweb_no_stop_roc_story_out.csv']
prefix = "partial_out_cobweb_"

for i, typ in enumerate(['normal', 'basic', 'best']):
    outfile = ''.join([prefix, typ, '.csv'])
    with open(outfile, 'w') as fout:
        fout.write("{},{},{}\n".format("Actual", "Predicted", "Similarity"))

for fn in files:
    scores = []
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # Skip the field name row
        for row in reader:
            for i, typ in enumerate(['normal', 'basic', 'best']):
                outfile = ''.join([prefix, typ, '.csv'])
                (corr, pred) = (row[3].strip(), row[7 + (3 * i)].strip())
                if pred == 'NONE' and PENALIZE_NONES:
                    scores.append(0.0)
                    with open(outfile, 'a') as fout:
                        fout.write('{},{},{}\n'.format(corr, pred, 0.0))
                    if PRINT_ALL:
                        print('%s\t%s\t%.2f' % (corr, pred, 0.0))
                elif pred == corr:
                    scores.append(1.0)
                    with open(outfile, 'a') as fout:
                        fout.write('{},{},{}\n'.format(corr, pred, 1.0))
                    if PRINT_ALL:
                        print('%s\t%s\t%.2f' % (corr, pred, 1.0))
                else:
                    try:
                        assert pred != 'NONE'
                        sim = w2v.similarity(corr, pred)
                        scores.append(sim)
                        with open(outfile, 'a') as fout:
                            fout.write('{},{},{}\n'.format(corr, pred, sim))
                        if PRINT_ALL:
                            print('%s\t%s\t%.2f' % (corr, pred, sim))
                    except:
                        sim = 0.0
                        with open(outfile, 'a') as fout:
                            fout.write('{},{},{}\n'.format(corr, pred, sim))

    print('%s: %.2f' % (fn, sum(scores) * 1.0 / len(scores)))
