''' Here I use machine learning to synthesize 
novel cell penetrating peptides'''

# import and cleanup the data
import pandas as pd
from Bio import SeqIO

# empty lists to hold the sequences
names = []
sequences = []

# load sequences from fasta files using Biopython
for seq_record in SeqIO.parse("cpp.fa", "fasta"):
    names.append(seq_record.id)
    sequences.append(str(seq_record.seq))
for seq_record in SeqIO.parse("not_cpp.fa", "fasta"):
    names.append(seq_record.id)
    sequences.append(str(seq_record.seq))

# format the data correctly for the next steps
data = pd.DataFrame(list(zip(names, sequences)), columns =['id', 'sequence'])
labels = [1] * 738 + [0] * 854
data['labels'] = labels
data = data.sample(frac=1) # shuffle the sequences randomly
del names, seq_record, sequences, labels

# keep the labels, but drop them from the dataframe
labels = data.labels
data.drop(['labels'], axis=1, inplace=True)

# needs to be lists of amino acids for graph transform
data['seq_list'] = data['sequence'].apply(lambda x: list(x))
data.drop(['sequence'], axis=1, inplace=True)
data.columns = ['id', 'sequence']


# sequence graph transform
# convert sequences into graph representation (uniform length numerical vectors)
from sgt import SGT
alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 
			 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 
			 'R', 'S', 'T', 'V', 'W', 'Y']

sgt = SGT(alphabets=alphabet, 
		  kappa=1, 
          flatten=True, 
          lengthsensitive=True, 
          mode='default')

# feature embeddings (graph) from sequences
# will be used as features for training machine learning algorithm
embedding = sgt.fit_transform(data)
embedding = embedding.set_index('id')



# creating a Random Forest classifier model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 50, 
                             criterion = 'entropy',
                             random_state = 16)

# train and evaluate model using k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = clf,
                             X = embedding, 
                             y = labels, 
                             cv = 5)
print("The mean accuracy is: ", accuracies.mean())
print("The standard deviation is: ", accuracies.std())

# finally fit to the entire data set
clf.fit(embedding, labels)


### Using the trained model to synthesize novel CPP sequences

# generate random peptide sequences from the defined alphabet
import random
def random_peptide(length):
    return [random.choice(alphabet) for x in range(length)]

# function to synthesize novel CPP's
# number of tries, peptide length, predicted probability cutoff
def synthesize(n_iter, pep_length, probability):
    for i in range(n_iter):
        rand_pep = random_peptide(pep_length)
        pred = clf.predict_proba(sgt.transform(pd.DataFrame({'id': [9999], 'sequence': [rand_pep]})).drop(['id'], axis=1))[0][1]
        if pred > probability:
            print(pred, ''.join(rand_pep))

# synthesize novel CPP's
synthesize(10000, 20, 0.8)

# test to make sure our novel peptides are not in the training data!
"YYPTFQRKRRYMSTCQMWWP" in list(data.sequence)

